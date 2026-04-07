//! GGUF file parser — reads metadata and tensor info without loading weights into memory.
//! Uses memory-mapped I/O so tensor data can be read on-demand.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian (bytes: G G U F)

// ── GGUF value types ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }
}

// ── Quantization types ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::IQ2XXS,
            16 => Self::IQ2XS,
            17 => Self::IQ3XXS,
            18 => Self::IQ1S,
            19 => Self::IQ4NL,
            20 => Self::IQ3S,
            21 => Self::IQ2S,
            22 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Block size (number of elements per block) for quantized types
    pub fn block_size(&self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 |
            Self::Q8_0 | Self::Q8_1 | Self::IQ4NL => 32,
            // K-quants use QK_K = 256
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S | Self::IQ3XXS | Self::IQ3S |
            Self::IQ1S | Self::IQ1M | Self::IQ4XS => 256,
            Self::Unknown(_) => 32,
        }
    }

    /// Bytes per block (exact, matches GGML's sizeof(block_xxx))
    pub fn type_size(&self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            // Non-K quants (block_size=32)
            Self::Q4_0 => 2 + 32 / 2,          // 18: half d + 16 bytes of nibbles
            Self::Q4_1 => 2 + 2 + 32 / 2,      // 20: half d + half m + 16 bytes
            Self::Q5_0 => 2 + 4 + 32 / 2,      // 22: half d + 4 bytes high bits + 16 bytes
            Self::Q5_1 => 2 + 2 + 4 + 32 / 2,  // 24: half d + half m + 4 bytes + 16 bytes
            Self::Q8_0 => 2 + 32,               // 34: half d + 32 bytes
            Self::Q8_1 => 4 + 4 + 32,           // 40: float d + float s + 32 bytes
            Self::IQ4NL => 2 + 16,              // 18: half d + 16 bytes
            // K-quants (QK_K=256)
            Self::Q2K => 256 / 16 + 256 / 4 + 2 + 2,  // 84: scales + qs + d + dmin
            Self::Q3K => 256 / 4 + 256 / 8 + 12 + 2,  // 110: hmask + qs + scales + d
            Self::Q4K => 2 + 2 + 12 + 256 / 2,         // 144: d + dmin + scales + qs
            Self::Q5K => 2 + 2 + 12 + 256 / 2 + 256 / 8, // 176: d + dmin + scales + qs + qh
            Self::Q6K => 256 / 2 + 256 / 4 + 256 / 16 + 2, // 210: ql + qh + scales + d
            // IQ types (approximate — less common, using known sizes)
            Self::IQ2XXS => 2 + 2 * 256 / 8,   // 66
            Self::IQ2XS => 2 + 2 * 256 / 8 + 256 / 32, // 74
            Self::IQ2S => 2 + 256 / 4 + 256 / 16, // 82
            Self::IQ3XXS => 2 + 3 * 256 / 8,   // 98
            Self::IQ3S => 2 + 256 / 4 + 256 / 8 + 256 / 32, // 110
            Self::IQ1S => 2 + 256 / 8 + 256 / 16, // 50
            Self::IQ1M => 256 / 8 + 256 / 16 + 256 / 32, // 56
            Self::IQ4XS => 2 + 2 + 256 / 2,    // 132
            Self::Unknown(_) => 18, // assume Q4_0-like
        }
    }

    /// Exact byte count for n_elements of this type
    pub fn nbytes(&self, n_elements: u64) -> u64 {
        let bs = self.block_size();
        let n_blocks = (n_elements + bs - 1) / bs;
        n_blocks * self.type_size()
    }

    /// Bits per element (approximate, for display only)
    pub fn bits_per_element(&self) -> f64 {
        self.type_size() as f64 * 8.0 / self.block_size() as f64
    }
}

// ── Tensor info ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset into the data section of the GGUF file
    pub data_offset: u64,
    /// Size in bytes
    pub size_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }
}

// ── GGUF File ───────────────────────────────────────────────────

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    _mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = 0usize;

        // Header
        let magic = read_u32(&mmap, &mut cursor);
        if magic != GGUF_MAGIC {
            bail!("Not a GGUF file (magic: {:#x})", magic);
        }

        let version = read_u32(&mmap, &mut cursor);
        if version < 2 || version > 3 {
            bail!("Unsupported GGUF version: {}", version);
        }

        let n_tensors = read_u64(&mmap, &mut cursor) as usize;
        let n_kv = read_u64(&mmap, &mut cursor) as usize;

        // Metadata KV pairs
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mmap, &mut cursor);
            let value = read_value(&mmap, &mut cursor);
            metadata.insert(key, value);
        }

        // Tensor infos
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mmap, &mut cursor);
            let n_dims = read_u32(&mmap, &mut cursor) as usize;
            let shape: Vec<u64> = (0..n_dims).map(|_| read_u64(&mmap, &mut cursor)).collect();
            let dtype = GgmlType::from_u32(read_u32(&mmap, &mut cursor));
            let offset = read_u64(&mmap, &mut cursor);

            let n_elements: u64 = shape.iter().product();
            // For stacked tensors [d1, d2, n_expert], the innermost dims are
            // contiguous blocks. Size = product(inner_dims) computed per-block,
            // then multiplied by outermost dim.
            // For simple tensors, this is just dtype.nbytes(n_elements).
            let size_bytes = if shape.len() >= 2 {
                // Inner elements = product of all dims except last
                let inner: u64 = shape[..shape.len()-1].iter().product();
                let outer = *shape.last().unwrap();
                dtype.nbytes(inner) * outer
            } else {
                dtype.nbytes(n_elements)
            };

            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                data_offset: offset,
                size_bytes,
            });
        }

        // Data section starts after alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        let data_offset = (cursor + alignment - 1) / alignment * alignment;

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset: data_offset as u64,
            _mmap: mmap,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn n_layers(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.block_count", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn n_experts(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.expert_count", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn n_experts_used(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.expert_used_count", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn embedding_dim(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.embedding_length", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.context_length", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn total_size_mb(&self) -> f64 {
        self.tensors.iter().map(|t| t.size_mb()).sum()
    }

    /// Read raw bytes for a tensor from the mmap'd file.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = self.data_offset as usize + tensor.data_offset as usize;
        let end = start + tensor.size_bytes as usize;
        &self._mmap[start..end]
    }

    /// Find a tensor by name
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

// ── Binary readers ──────────────────────────────────────────────

fn read_u8(data: &[u8], cursor: &mut usize) -> u8 {
    let v = data[*cursor];
    *cursor += 1;
    v
}

fn read_u16(data: &[u8], cursor: &mut usize) -> u16 {
    let v = u16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    v
}

fn read_u32(data: &[u8], cursor: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    v
}

fn read_i32(data: &[u8], cursor: &mut usize) -> i32 {
    let v = i32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    v
}

fn read_u64(data: &[u8], cursor: &mut usize) -> u64 {
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    v
}

fn read_i64(data: &[u8], cursor: &mut usize) -> i64 {
    let v = i64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    v
}

fn read_f32(data: &[u8], cursor: &mut usize) -> f32 {
    let v = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    v
}

fn read_f64(data: &[u8], cursor: &mut usize) -> f64 {
    let v = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    v
}

fn read_string(data: &[u8], cursor: &mut usize) -> String {
    let len = read_u64(data, cursor) as usize;
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .unwrap_or("<invalid utf8>")
        .to_string();
    *cursor += len;
    s
}

fn read_value(data: &[u8], cursor: &mut usize) -> GgufValue {
    let vtype = read_u32(data, cursor);
    read_typed_value(data, cursor, vtype)
}

fn read_typed_value(data: &[u8], cursor: &mut usize, vtype: u32) -> GgufValue {
    match vtype {
        0 => GgufValue::U8(read_u8(data, cursor)),
        1 => GgufValue::I8(read_u8(data, cursor) as i8),
        2 => GgufValue::U16(read_u16(data, cursor)),
        3 => GgufValue::I16(read_u16(data, cursor) as i16),
        4 => GgufValue::U32(read_u32(data, cursor)),
        5 => GgufValue::I32(read_i32(data, cursor)),
        6 => GgufValue::F32(read_f32(data, cursor)),
        7 => {
            let b = read_u8(data, cursor);
            GgufValue::Bool(b != 0)
        }
        8 => GgufValue::String(read_string(data, cursor)),
        9 => {
            // Array
            let atype = read_u32(data, cursor);
            let alen = read_u64(data, cursor) as usize;
            let values: Vec<GgufValue> = (0..alen)
                .map(|_| read_typed_value(data, cursor, atype))
                .collect();
            GgufValue::Array(values)
        }
        10 => GgufValue::U64(read_u64(data, cursor)),
        11 => GgufValue::I64(read_i64(data, cursor)),
        12 => GgufValue::F64(read_f64(data, cursor)),
        _ => {
            // Unknown type, try to skip
            GgufValue::U32(0)
        }
    }
}
