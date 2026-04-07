//! Test expert FFN computation against reference.
//! Reads one expert from repacked file, dequantizes, computes gate@x, and prints output.

use anyhow::Result;
use llamactl_engine::GgufFile;

fn main() -> Result<()> {
    let gguf_path = std::env::args().nth(1).expect("Usage: test_expert_compute <model.gguf> <experts_dir>");
    let experts_dir = std::env::args().nth(2).expect("Usage: test_expert_compute <model.gguf> <experts_dir>");

    let gguf = GgufFile::open(&gguf_path)?;
    let n_experts = gguf.n_experts().unwrap_or(0) as usize;
    let n_embd = gguf.embedding_dim().unwrap_or(0) as usize;

    // Find a non-zero expert. From verification, gate/up expert 0 is all zeros.
    // Let's use down_exps expert 0 which has data.
    // Actually let's check which expert in gate_exps is non-zero
    let gate_tensor = gguf.find_tensor("blk.0.ffn_gate_exps.weight").unwrap();
    let gate_data = gguf.tensor_data(gate_tensor);
    let per_expert = gate_data.len() / n_experts;
    let n_ff = gate_tensor.shape[1] as usize;

    println!("Gate tensor: shape={:?} dtype={:?}", gate_tensor.shape, gate_tensor.dtype);
    println!("n_embd={}, n_ff={}, per_expert={} bytes\n", n_embd, n_ff, per_expert);

    // Find first non-zero expert
    let mut test_expert = 0;
    for e in 0..n_experts {
        let start = e * per_expert;
        let end = start + per_expert;
        let sum: u64 = gate_data[start..end].iter().map(|&b| b as u64).sum();
        if sum > 0 {
            test_expert = e;
            println!("First non-zero expert in gate_exps layer 0: expert {}", e);
            break;
        }
    }

    let expert_data = &gate_data[test_expert * per_expert..(test_expert + 1) * per_expert];
    println!("Expert {} gate data first 64 bytes: {:?}", test_expert, &expert_data[..64]);

    // Dequantize using GGML's reference implementation
    // We can't call GGML C functions from Rust directly, but we can implement Q3K dequant
    // Let's check if the data makes sense by computing row sums

    // Q3K: block_size=256, type_size=110
    // Each row of n_embd=3072 elements = 12 blocks = 1320 bytes
    let block_bytes = 110;
    let blocks_per_row = n_embd / 256;
    let row_bytes = blocks_per_row * block_bytes;

    println!("\nRow layout: {} blocks/row × {} bytes/block = {} bytes/row", blocks_per_row, block_bytes, row_bytes);
    println!("Expected total: {} rows × {} bytes = {} (actual: {})", n_ff, row_bytes, n_ff * row_bytes, per_expert);

    if n_ff * row_bytes != per_expert {
        println!("MISMATCH! per_expert doesn't equal n_ff * row_bytes");
        println!("This means the rows might not be what we think");
    }

    // Check: could it be stored as [n_embd rows of n_ff cols] instead?
    let alt_blocks_per_row = n_ff / 256;  // 1024/256 = 4
    let alt_row_bytes = alt_blocks_per_row * block_bytes;  // 4 * 110 = 440
    let alt_total = n_embd * alt_row_bytes;  // 3072 * 440 = 1,351,680
    println!("\nAlternative: {} rows of {} cols ({} blocks/row = {} bytes/row)",
        n_embd, n_ff, alt_blocks_per_row, alt_row_bytes);
    println!("Alternative total: {} (actual: {})", alt_total, per_expert);

    if alt_total == per_expert {
        println!("*** MATCH! The actual layout is [n_embd rows of n_ff cols], NOT [n_ff rows of n_embd cols] ***");
        println!("This means out_dim=n_embd={}, in_dim=n_ff={} for gate matmul", n_embd, n_ff);
        println!("But we were computing with out_dim=n_ff, in_dim=n_embd — WRONG DIMENSIONS!");
    }

    Ok(())
}
