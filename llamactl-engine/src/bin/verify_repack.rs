//! Verify repacked expert data matches the original GGUF.
//! Reads expert 0 layer 0 from both sources and compares byte-for-byte.

use anyhow::{Context, Result};
use llamactl_engine::GgufFile;
use std::io::Read;

fn main() -> Result<()> {
    let gguf_path = std::env::args().nth(1).expect("Usage: verify_repack <model.gguf> <experts_dir>");
    let experts_dir = std::env::args().nth(2).expect("Usage: verify_repack <model.gguf> <experts_dir>");

    println!("GGUF: {}", gguf_path);
    println!("Experts: {}\n", experts_dir);

    let gguf = GgufFile::open(&gguf_path)?;
    let n_experts = gguf.n_experts().unwrap_or(0) as usize;

    // Read expert tensor info
    let tensor_names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

    for tensor_name in &tensor_names {
        let full_name = format!("blk.0.{}.weight", tensor_name);
        let tensor = gguf.find_tensor(&full_name)
            .with_context(|| format!("Tensor not found: {}", full_name))?;

        let gguf_data = gguf.tensor_data(tensor);
        let per_expert = gguf_data.len() / n_experts;

        println!("Tensor: {} | shape: {:?} | dtype: {:?}", full_name, tensor.shape, tensor.dtype);
        println!("  Total bytes: {} | Per expert: {} | n_experts: {}", gguf_data.len(), per_expert, n_experts);

        // Expert 0 data from GGUF: first per_expert bytes
        let gguf_expert0 = &gguf_data[0..per_expert];

        // Read first bytes of expert 0 from GGUF
        println!("  GGUF expert 0 first 32 bytes: {:?}", &gguf_expert0[..32]);
        println!("  GGUF expert 0 last 32 bytes:  {:?}", &gguf_expert0[per_expert-32..]);

        println!();
    }

    // Now read from repacked file
    println!("--- Repacked layer_000.bin ---\n");

    // Read manifest for layout
    let manifest_str = std::fs::read_to_string(format!("{}/manifest.json", experts_dir))?;
    let layer_meta_str = std::fs::read_to_string(format!("{}/layer_000.json", experts_dir))?;

    // Parse per_expert_bytes and tensor sizes from layer metadata
    // Simple extraction
    let per_expert_bytes: usize = {
        let pos = layer_meta_str.find("\"per_expert_bytes\"").unwrap();
        let colon = layer_meta_str[pos..].find(':').unwrap();
        let start = pos + colon + 1;
        let end = layer_meta_str[start..].find(|c: char| c == ',' || c == '}').unwrap();
        layer_meta_str[start..start+end].trim().parse()?
    };

    // Parse tensor byte sizes
    let mut tensor_sizes: Vec<usize> = Vec::new();
    {
        let pos = layer_meta_str.find("\"expert_tensors\"").unwrap();
        let arr_start = layer_meta_str[pos..].find('[').unwrap() + pos;
        let arr_end = layer_meta_str[arr_start..].find(']').unwrap() + arr_start;
        let arr = &layer_meta_str[arr_start..=arr_end];
        let mut cursor = 0;
        while let Some(p) = arr[cursor..].find("\"bytes\"") {
            let colon = arr[cursor+p..].find(':').unwrap();
            let start = cursor + p + colon + 1;
            let end = arr[start..].find(|c: char| c == ',' || c == '}').unwrap();
            let bytes: usize = arr[start..start+end].trim().parse()?;
            tensor_sizes.push(bytes);
            cursor = start + end;
        }
    }

    println!("Per expert bytes: {}", per_expert_bytes);
    println!("Tensor sizes: {:?}\n", tensor_sizes);

    // Read repacked file
    let mut file = std::fs::File::open(format!("{}/layer_000.bin", experts_dir))?;
    let mut repacked = vec![0u8; per_expert_bytes];
    file.read_exact(&mut repacked)?;

    // Compare expert 0 tensors
    let mut offset = 0;
    for (i, tensor_name) in tensor_names.iter().enumerate() {
        let size = tensor_sizes[i];
        let repacked_slice = &repacked[offset..offset + size];

        let full_name = format!("blk.0.{}.weight", tensor_name);
        let tensor = gguf.find_tensor(&full_name).unwrap();
        let gguf_data = gguf.tensor_data(tensor);
        let per_expert = gguf_data.len() / n_experts;
        let gguf_expert0 = &gguf_data[0..per_expert];

        // Compare sizes
        println!("{}:", tensor_name);
        println!("  Repacked size: {} | GGUF per-expert: {}", size, per_expert);

        if size != per_expert {
            println!("  SIZE MISMATCH!");
        }

        let min_len = std::cmp::min(size, per_expert);
        let mut mismatches = 0;
        for j in 0..min_len {
            if repacked_slice[j] != gguf_expert0[j] {
                mismatches += 1;
                if mismatches <= 3 {
                    println!("  Mismatch at byte {}: repacked={:#04x} gguf={:#04x}", j, repacked_slice[j], gguf_expert0[j]);
                }
            }
        }
        if mismatches == 0 {
            println!("  MATCH: all {} bytes identical", min_len);
        } else {
            println!("  {} mismatches out of {} bytes", mismatches, min_len);
        }

        // Also check expert 1
        let gguf_expert1 = &gguf_data[per_expert..2*per_expert];
        // Read expert 1 from repacked
        let mut expert1_buf = vec![0u8; per_expert_bytes];
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(per_expert_bytes as u64))?;
        file.read_exact(&mut expert1_buf)?;
        let repacked_e1 = &expert1_buf[offset..offset + size];

        let min_len = std::cmp::min(size, per_expert);
        let mut mismatches = 0;
        for j in 0..min_len {
            if repacked_e1[j] != gguf_expert1[j] { mismatches += 1; }
        }
        println!("  Expert 1: {} mismatches out of {} bytes", mismatches, min_len);

        offset += size;
        println!();
    }

    // Also verify: what does GGML's to_float produce for expert 0 gate?
    println!("--- Dequant verification ---\n");
    {
        let full_name = "blk.0.ffn_gate_exps.weight";
        let tensor = gguf.find_tensor(full_name).unwrap();
        let gguf_data = gguf.tensor_data(tensor);
        let per_expert = gguf_data.len() / n_experts;

        println!("Gate tensor: {:?} dtype={:?}", tensor.shape, tensor.dtype);
        println!("GGUF shape convention: shape[0]={} (innermost/contiguous), shape[1]={}, shape[2]={} (outermost)",
            tensor.shape[0], tensor.shape[1], tensor.shape[2]);
        println!("In memory: [{} experts] × [{} rows] × [{} cols]",
            tensor.shape[2], tensor.shape[1], tensor.shape[0]);
        println!("Per expert: {} rows of {} elements = {} elements total",
            tensor.shape[1], tensor.shape[0], tensor.shape[0] * tensor.shape[1]);
        println!("Per expert bytes: {} (quantized), expected F32: {} bytes",
            per_expert, tensor.shape[0] * tensor.shape[1] * 4);

        // For ggml_mul_mat(gate_w, x):
        // gate_w GGUF shape [n_embd, n_ff] = memory [n_ff rows, n_embd cols]
        // x shape [n_embd]
        // output shape [n_ff]
        // Computation: out[i] = dot(gate_w_row[i], x) for i in 0..n_ff
        //
        // In the stacked tensor [n_embd, n_ff, n_expert]:
        // Expert e's gate_w in memory: rows [e*n_ff .. (e+1)*n_ff], each row has n_embd elements
        // But with quantization, each row of n_embd elements is stored as ceil(n_embd/256) blocks
        //
        // The question is: does expert e's data start at byte offset e * per_expert_bytes?
        // Let's verify: in GGUF, the stacked tensor is [n_embd, n_ff, n_expert]
        // Memory layout (C row-major, GGUF reversed): [n_expert][n_ff][n_embd]
        // So expert e starts at: e * n_ff * row_bytes
        // where row_bytes = ceil(n_embd / 256) * block_size
        let n_embd = tensor.shape[0] as usize;
        let n_ff = tensor.shape[1] as usize;
        let _n_exp = tensor.shape[2] as usize;

        println!("\nFor matmul: out_dim={} (n_ff), in_dim={} (n_embd)", n_ff, n_embd);
        println!("Each row: {} elements quantized in {} blocks of 256",
            n_embd, n_embd / 256);
    }

    // Test dequant on a non-zero expert (down_exps expert 0 has data)
    println!("\n--- Dequant test: down_exps expert 0 ---\n");
    {
        let full_name = "blk.0.ffn_down_exps.weight";
        let tensor = gguf.find_tensor(full_name).unwrap();
        let gguf_data = gguf.tensor_data(tensor);
        let per_expert = gguf_data.len() / n_experts;
        let expert0 = &gguf_data[0..per_expert];

        // GGUF shape [n_ff=1024, n_embd=3072, n_expert=256]
        // In memory: [256 experts][3072 rows][1024 cols]
        // Wait — shape is [1024, 3072, 256] in GGUF
        // GGUF shape[0]=1024 (innermost), shape[1]=3072, shape[2]=256 (outermost)
        // Memory: [256 experts][3072 rows][1024 cols per row]
        let n_inner = tensor.shape[0] as usize; // 1024
        let n_rows = tensor.shape[1] as usize;  // 3072

        println!("Shape: [{}, {}, {}]", n_inner, n_rows, n_experts);
        println!("Per expert: {} rows of {} elements", n_rows, n_inner);

        // Q5K block: 256 elements per block, 176 bytes per block
        // Each row of 1024 elements = 4 blocks = 704 bytes
        let block_size = 256usize;
        let block_bytes = 176usize; // Q5K
        let blocks_per_row = n_inner / block_size;
        let row_bytes = blocks_per_row * block_bytes;
        println!("Row: {} blocks × {} bytes = {} bytes/row", blocks_per_row, block_bytes, row_bytes);
        println!("Total: {} rows × {} bytes = {} bytes (actual: {})",
            n_rows, row_bytes, n_rows * row_bytes, per_expert);

        // The issue might be HERE. If n_rows * row_bytes != per_expert, the rows are quantized
        // differently than we think.
        //
        // Actually for Q5K, ggml stores: blocks are [QK_K=256 elements], but the quantization
        // operates on ROWS. In ggml, the weight matrix [ne0, ne1] has ne1 rows of ne0 elements.
        // Each row of ne0 elements is quantized into ceil(ne0/256) blocks.
        //
        // For down_exps expert 0: ne0=1024, ne1=3072
        // So 3072 rows of 1024 elements, each row = 4 blocks of Q5K = 704 bytes
        // Total = 3072 * 704 = 2,162,688 — matches!

        // Now: our Metal shader computes out[row] = dot(dequant(W[row,:]), x)
        // For down_proj: out_dim=n_embd=3072, in_dim=n_ff=1024
        // So we have 3072 output rows, each row has 1024 elements (4 Q5K blocks)
        // This is correct.

        // Let's manually dequant first block of first row and check values
        println!("\nFirst block of expert 0, row 0 (Q5K, 256 elements):");
        let block = &expert0[0..block_bytes];
        println!("  Raw bytes [0..32]: {:?}", &block[0..32]);

        // Q5K format: d(half=2B), dmin(half=2B), scales(12B), qs(128B), qh(32B) = 176B
        let d_bytes = [block[0], block[1]];
        let dmin_bytes = [block[2], block[3]];
        let d = half::f16::from_le_bytes(d_bytes).to_f32();
        let dmin = half::f16::from_le_bytes(dmin_bytes).to_f32();
        println!("  d = {} (raw: {:?})", d, d_bytes);
        println!("  dmin = {} (raw: {:?})", dmin, dmin_bytes);

        // First few quantized values
        let qs = &block[16..144]; // qs starts at offset 4+12=16, 128 bytes
        let qh = &block[144..176]; // qh at offset 16+128=144, 32 bytes
        println!("  First 8 qs bytes: {:?}", &qs[0..8]);
        println!("  First 4 qh bytes: {:?}", &qh[0..4]);
    }

    Ok(())
}
