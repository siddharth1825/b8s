//! Tensor loader — loads GGUF tensors into candle QTensors with device placement awareness.
//!
//! Uses candle's built-in GGUF parser for actual tensor loading (handles all quant formats),
//! while our custom parser provides the categorization and device placement logic.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::Device;
use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;

use crate::ops::{categorize_tensors, CategorizedTensor, DevicePlacement};

/// A loaded model split across devices
pub struct LoadedModel {
    /// Tensors placed on the fast GPU (attention, router, SSM, norms)
    pub fast_gpu: HashMap<String, Arc<QTensor>>,
    /// Tensors placed on the big memory device (experts, shared FFN)
    pub big_memory: HashMap<String, Arc<QTensor>>,
    /// Tensors replicated on both (embeddings, output)
    pub replicated: HashMap<String, Arc<QTensor>>,
    /// Tensor categorization metadata
    pub categories: Vec<CategorizedTensor>,
    /// Model metadata
    pub n_layers: u32,
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub embedding_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub device: Device,
    pub vocab_size: u32,
}

impl LoadedModel {
    /// Load a GGUF model with auto-detected device (Metal if available, else CPU).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let device = Self::best_device();
        Self::load_with_device(path, &device)
    }

    /// Load only FastGpu + Replicated tensors onto Metal GPU.
    /// BigMemory (expert) tensors are skipped — they live on the remote worker.
    /// This is for split mode where experts are offloaded via gRPC.
    pub fn load_split_local(path: impl AsRef<Path>) -> Result<Self> {
        let device = Self::best_device();
        Self::load_with_device_split(path, &device)
    }

    /// Detect the best available device
    pub fn best_device() -> Device {
        match Device::new_metal(0) {
            Ok(d) => {
                tracing::info!("Using Metal GPU device");
                d
            }
            Err(_) => {
                tracing::info!("Metal not available, using CPU");
                Device::Cpu
            }
        }
    }

    /// Load a GGUF model onto a specific device, categorizing tensors by placement.
    pub fn load_with_device(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!(path = %path.display(), device = ?device, "Loading GGUF model");

        // Use our parser for metadata + categorization
        let our_gguf = crate::gguf::GgufFile::open(path)?;
        let categories = categorize_tensors(&our_gguf);

        let n_layers = our_gguf.n_layers().unwrap_or(1);
        let n_experts = our_gguf.n_experts().unwrap_or(1);
        let n_experts_used = our_gguf.n_experts_used().unwrap_or(1);
        let embedding_dim = our_gguf.embedding_dim().unwrap_or(2048);

        let arch = our_gguf.architecture().unwrap_or("unknown");
        let n_heads = our_gguf.metadata
            .get(&format!("{}.attention.head_count", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(16);
        let n_kv_heads = our_gguf.metadata
            .get(&format!("{}.attention.head_count_kv", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(2);
        let head_dim = our_gguf.metadata
            .get(&format!("{}.attention.key_length", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(embedding_dim / n_heads);
        let rms_norm_eps = our_gguf.metadata
            .get(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let vocab_size = our_gguf.metadata
            .get(&format!("tokenizer.ggml.tokens", ))
            .map(|v| match v {
                crate::gguf::GgufValue::Array(a) => a.len() as u32,
                _ => 248320,
            })
            .unwrap_or(248320);

        // Use candle's parser for actual tensor loading
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF with candle")?;

        // Build placement lookup from our categorization
        let placement_map: HashMap<String, DevicePlacement> = categories
            .iter()
            .map(|c| (c.name.clone(), c.placement))
            .collect();

        let mut fast_gpu = HashMap::new();
        let mut big_memory = HashMap::new();
        let mut replicated = HashMap::new();

        let total = content.tensor_infos.len();
        tracing::info!(total, "Loading tensors");

        Self::load_tensors_filtered(&content, &mut file, device, &placement_map,
            &mut fast_gpu, &mut big_memory, &mut replicated, false)?;

        tracing::info!(
            fast_gpu = fast_gpu.len(),
            big_memory = big_memory.len(),
            replicated = replicated.len(),
            "Model loaded"
        );

        Ok(Self {
            fast_gpu,
            big_memory,
            replicated,
            categories,
            n_layers,
            n_experts,
            n_experts_used,
            embedding_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            vocab_size,
            device: device.clone(),
        })
    }

    /// Get a tensor by name, searching all device placements
    pub fn tensor(&self, name: &str) -> Option<&Arc<QTensor>> {
        self.fast_gpu.get(name)
            .or_else(|| self.big_memory.get(name))
            .or_else(|| self.replicated.get(name))
    }

    /// Summary of memory usage by device
    pub fn memory_summary(&self) -> String {
        let fast_mb: f64 = self.fast_gpu.values()
            .map(|t| t.storage_size_in_bytes() as f64 / (1024.0 * 1024.0))
            .sum();
        let big_mb: f64 = self.big_memory.values()
            .map(|t| t.storage_size_in_bytes() as f64 / (1024.0 * 1024.0))
            .sum();
        let rep_mb: f64 = self.replicated.values()
            .map(|t| t.storage_size_in_bytes() as f64 / (1024.0 * 1024.0))
            .sum();

        format!(
            "FastGpu: {:.0} MB ({} tensors), BigMemory: {:.0} MB ({} tensors), Replicated: {:.0} MB ({} tensors)",
            fast_mb, self.fast_gpu.len(),
            big_mb, self.big_memory.len(),
            rep_mb, self.replicated.len(),
        )
    }

    /// Load only FastGpu + Replicated tensors (for split mode where experts are remote)
    pub fn load_with_device_split(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!(path = %path.display(), device = ?device, "Loading GGUF (split mode — skipping expert weights)");

        let our_gguf = crate::gguf::GgufFile::open(path)?;
        let categories = categorize_tensors(&our_gguf);

        let n_layers = our_gguf.n_layers().unwrap_or(1);
        let n_experts = our_gguf.n_experts().unwrap_or(1);
        let n_experts_used = our_gguf.n_experts_used().unwrap_or(1);
        let embedding_dim = our_gguf.embedding_dim().unwrap_or(2048);
        let arch = our_gguf.architecture().unwrap_or("unknown");
        let n_heads = our_gguf.metadata.get(&format!("{}.attention.head_count", arch)).and_then(|v| v.as_u32()).unwrap_or(16);
        let n_kv_heads = our_gguf.metadata.get(&format!("{}.attention.head_count_kv", arch)).and_then(|v| v.as_u32()).unwrap_or(2);
        let head_dim = our_gguf.metadata.get(&format!("{}.attention.key_length", arch)).and_then(|v| v.as_u32()).unwrap_or(embedding_dim / n_heads);
        let rms_norm_eps = our_gguf.metadata.get(&format!("{}.attention.layer_norm_rms_epsilon", arch)).and_then(|v| v.as_f32()).unwrap_or(1e-6);
        let vocab_size = our_gguf.metadata.get("tokenizer.ggml.tokens").map(|v| match v { crate::gguf::GgufValue::Array(a) => a.len() as u32, _ => 248320 }).unwrap_or(248320);

        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).context("Failed to parse GGUF")?;

        let placement_map: HashMap<String, DevicePlacement> = categories.iter().map(|c| (c.name.clone(), c.placement)).collect();

        let mut fast_gpu = HashMap::new();
        let mut big_memory = HashMap::new();
        let mut replicated = HashMap::new();

        Self::load_tensors_filtered(&content, &mut file, device, &placement_map,
            &mut fast_gpu, &mut big_memory, &mut replicated, true)?;

        tracing::info!(
            fast_gpu = fast_gpu.len(),
            big_memory = big_memory.len(),
            replicated = replicated.len(),
            "Model loaded (split mode)"
        );

        Ok(Self {
            fast_gpu, big_memory, replicated, categories,
            n_layers, n_experts, n_experts_used, embedding_dim,
            n_heads, n_kv_heads, head_dim, rms_norm_eps, vocab_size,
            device: device.clone(),
        })
    }

    fn load_tensors_filtered(
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        placement_map: &HashMap<String, DevicePlacement>,
        fast_gpu: &mut HashMap<String, Arc<QTensor>>,
        big_memory: &mut HashMap<String, Arc<QTensor>>,
        replicated: &mut HashMap<String, Arc<QTensor>>,
        skip_big_memory: bool,
    ) -> Result<()> {
        let total = content.tensor_infos.len();
        let mut loaded = 0;
        let mut skipped = 0;

        for (name, _) in content.tensor_infos.iter() {
            let placement = placement_map.get(name.as_str()).copied().unwrap_or(DevicePlacement::BigMemory);

            if skip_big_memory && placement == DevicePlacement::BigMemory {
                skipped += 1;
                continue;
            }

            let tensor = content.tensor(file, name, device)
                .with_context(|| format!("Failed to load tensor: {}", name))?;
            let tensor = Arc::new(tensor);

            match placement {
                DevicePlacement::FastGpu => { fast_gpu.insert(name.clone(), tensor); }
                DevicePlacement::BigMemory => { big_memory.insert(name.clone(), tensor); }
                DevicePlacement::Replicated => { replicated.insert(name.clone(), tensor); }
            }
            loaded += 1;
        }

        tracing::info!(loaded, skipped, total, "Tensor loading complete");
        Ok(())
    }
}
