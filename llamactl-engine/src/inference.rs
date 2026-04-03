//! Inference operations for heterogeneous split execution.
//!
//! Implements Qwen3.5 MoE hybrid architecture:
//! - SSM (Mamba/Gated Delta Net) layers: 30 of 40 layers
//! - Full self-attention layers: 10 of 40 layers (at indices 3,7,11,...,39)
//! - MoE FFN with 256 experts (8 active) on every layer

use anyhow::Result;
use candle_core::{DType, IndexOp, Module, Tensor};
use candle_core::quantized::QMatMul;
use std::sync::Arc;

use crate::loader::LoadedModel;

// ── Which layers use attention vs SSM ───────────────────────────

/// Attention layers for Qwen3.5: every 4th layer starting at 3
const ATTENTION_LAYERS: [u32; 10] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39];

pub fn is_attention_layer(layer: u32) -> bool {
    ATTENTION_LAYERS.contains(&layer)
}

// ── RMS Norm ────────────────────────────────────────────────────

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = (&x_f32 * &x_f32)?.mean_keepdim(candle_core::D::Minus1)?;
    let x_norm = x_f32.broadcast_div(&(variance + eps as f64)?.sqrt()?)?;
    let out = x_norm.to_dtype(dtype)?.broadcast_mul(weight)?;
    Ok(out)
}

// ── KV Cache (only for attention layers) ────────────────────────

pub struct KvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn append(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            Some(prev) => Tensor::cat(&[prev, &k], 2)?,
            None => k,
        };
        let v = match &self.v {
            Some(prev) => Tensor::cat(&[prev, &v], 2)?,
            None => v,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

// ── Self-Attention (GQA, for attention layers only) ─────────────

/// Full self-attention with GQA, Q/K norms, and gated Q
/// Qwen3.5 attention: Q is [2048 → 8192] which includes gate (16 heads * 256 * 2)
pub fn self_attention(
    x: &Tensor,
    model: &LoadedModel,
    layer: u32,
    kv_cache: &mut KvCache,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);
    let (batch, seq_len, _hidden) = x.dims3()?;
    let n_heads = model.n_heads as usize;
    let n_kv_heads = model.n_kv_heads as usize;
    let head_dim = model.head_dim as usize;

    // Q projection: [batch, seq, 2048] → [batch, seq, 8192]
    // The 8192 is 2 * n_heads * head_dim = gated Q (split into Q and gate)
    let q_w = make_qmatmul(model, &format!("{}.attn_q.weight", prefix))?;
    let q_proj = q_w.forward(x)?; // [batch, seq, 8192]

    // Split into Q and Q_gate
    let q_dim = n_heads * head_dim; // 4096
    let q = q_proj.narrow(candle_core::D::Minus1, 0, q_dim)?;
    let q_gate = q_proj.narrow(candle_core::D::Minus1, q_dim, q_dim)?;

    // K, V projections
    let k_w = make_qmatmul(model, &format!("{}.attn_k.weight", prefix))?;
    let v_w = make_qmatmul(model, &format!("{}.attn_v.weight", prefix))?;
    let k = k_w.forward(x)?; // [batch, seq, 512]
    let v = v_w.forward(x)?; // [batch, seq, 512]

    // Q/K norms (per-head RMS norm)
    let q_norm_w = model.tensor(&format!("{}.attn_q_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing attn_q_norm"))?;
    let k_norm_w = model.tensor(&format!("{}.attn_k_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing attn_k_norm"))?;

    // Reshape for multi-head
    let q = q.reshape((batch, seq_len, n_heads, head_dim))?;
    let k = k.reshape((batch, seq_len, n_kv_heads, head_dim))?;

    // Apply per-head norm
    let q_norm = q_norm_w.dequantize(x.device())?;
    let k_norm = k_norm_w.dequantize(x.device())?;
    let q = rms_norm(&q, &q_norm, model.rms_norm_eps)?;
    let k = rms_norm(&k, &k_norm, model.rms_norm_eps)?;

    // Transpose to [batch, heads, seq, head_dim]
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.reshape((batch, seq_len, n_kv_heads, head_dim))?.transpose(1, 2)?;

    // GQA: repeat KV heads
    let repeats = n_heads / n_kv_heads;
    let k = if repeats > 1 { k.repeat(&[1, repeats, 1, 1])? } else { k };
    let v = if repeats > 1 { v.repeat(&[1, repeats, 1, 1])? } else { v };

    // KV cache
    let (k, v) = kv_cache.append(k, v)?;

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

    // Causal mask
    let seq_len_q = attn_weights.dim(2)?;
    let seq_len_k = attn_weights.dim(3)?;
    let attn_weights = if seq_len_k > 1 {
        let mut mask_data = vec![0f32; seq_len_q * seq_len_k];
        let offset = seq_len_k.saturating_sub(seq_len_q);
        for i in 0..seq_len_q {
            for j in (i + offset + 1)..seq_len_k {
                mask_data[i * seq_len_k + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len_q, seq_len_k), x.device())?;
        attn_weights.broadcast_add(&mask)?
    } else {
        attn_weights
    };

    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    let attn_out = attn_weights.matmul(&v)?;
    let attn_out = attn_out.transpose(1, 2)?.reshape((batch, seq_len, n_heads * head_dim))?;

    // Apply Q gate (sigmoid gate on attention output)
    let gate = candle_nn::ops::sigmoid(&q_gate)?;
    let gated = (attn_out * gate)?;

    // Output projection
    let out_w = make_qmatmul(model, &format!("{}.attn_output.weight", prefix))?;
    let out = out_w.forward(&gated)?;

    Ok(out)
}

// ── Mamba / Gated Delta Net SSM Block ───────────────────────────

/// SSM (State-Space Model) block for non-attention layers.
///
/// Simplified Mamba-2 forward pass:
/// 1. in_proj: hidden → 2*inner (split into x, z gate)
/// 2. conv1d on x
/// 3. SSM recurrence (simplified: skip for autoregressive, use linear scan)
/// 4. Gated output: gate(z) * ssm_output
/// 5. out_proj: inner → hidden
pub fn ssm_block(
    x: &Tensor,
    model: &LoadedModel,
    layer: u32,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);
    let (batch, seq_len, hidden) = x.dims3()?;
    let inner_size = 4096usize; // ssm_inner_size from metadata

    // 1. Input projection: [batch, seq, 2048] → [batch, seq, 8192]
    let in_proj = make_qmatmul(model, &format!("{}.attn_qkv.weight", prefix))?;
    let projected = in_proj.forward(x)?; // [batch, seq, 8192]

    // Split into main path and gate: each [batch, seq, 4096]
    let xz = projected.narrow(candle_core::D::Minus1, 0, inner_size)?;
    let gate = projected.narrow(candle_core::D::Minus1, inner_size, inner_size)?;

    // 2. Conv1d + SiLU activation
    // Phase 2: simplified — apply SiLU directly (full conv1d with state tracking in Phase 3)
    let conv_out = candle_nn::ops::silu(&xz)?;

    // 3. SSM recurrence (simplified for Phase 2 — full Gated Delta Net in Phase 3)
    // conv_out: [batch, seq, 4096] (inner_size)
    // GGUF dims are reversed, so:
    //   ssm_out GGUF [4096, 2048] → candle QMatMul input=4096, output=2048 → output projection
    //   attn_gate GGUF [2048, 4096] → candle QMatMul input=2048, output=4096 → gate from residual

    // Gate: silu(z) * conv_out → gated inner representation
    let gated = (conv_out * candle_nn::ops::silu(&gate)?)?; // [batch, seq, 4096]

    // Output projection: inner(4096) → hidden(2048)
    let out_proj = make_qmatmul(model, &format!("{}.ssm_out.weight", prefix))?;
    let combined = out_proj.forward(&gated)?; // [batch, seq, 2048]

    Ok(combined)
}

// ── MoE Router ──────────────────────────────────────────────────

pub fn moe_route(
    x: &Tensor,
    gate_weight: &QMatMul,
    n_experts_used: u32,
) -> Result<(Tensor, Tensor)> {
    let logits = gate_weight.forward(x)?;
    let k = n_experts_used as usize;

    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let sorted_indices = probs.arg_sort_last_dim(false)?;
    let top_k_indices = sorted_indices.narrow(candle_core::D::Minus1, 0, k)?;
    let top_k_weights = probs.gather(&top_k_indices, candle_core::D::Minus1)?;
    let weight_sum = top_k_weights.sum_keepdim(candle_core::D::Minus1)?;
    let top_k_weights = top_k_weights.broadcast_div(&weight_sum)?;

    Ok((top_k_indices, top_k_weights))
}

// ── Expert FFN (sparse) ─────────────────────────────────────────

pub fn expert_ffn(
    x: &Tensor,
    expert_indices: &Tensor,
    expert_weights: &Tensor,
    model: &LoadedModel,
    layer: u32,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);
    let (batch, seq_len, hidden) = x.dims3()?;
    let k = expert_indices.dim(2)?;

    let gate_exps = model.tensor(&format!("{}.ffn_gate_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_gate_exps"))?;
    let up_exps = model.tensor(&format!("{}.ffn_up_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_up_exps"))?;
    let down_exps = model.tensor(&format!("{}.ffn_down_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_down_exps"))?;

    // Dequantize expert weights
    let gate_f = gate_exps.dequantize(x.device())?; // [hidden, intermediate, n_experts]
    let up_f = up_exps.dequantize(x.device())?;
    let down_f = down_exps.dequantize(x.device())?;

    let indices_vec = expert_indices.to_dtype(DType::U32)?.to_vec3::<u32>()?;
    let weights_vec = expert_weights.to_vec3::<f32>()?;

    let mut output_data = vec![0f32; batch * seq_len * hidden];

    for b in 0..batch {
        for s in 0..seq_len {
            let token = x.narrow(0, b, 1)?.narrow(1, s, 1)?.squeeze(0)?.squeeze(0)?;
            let token_2d = token.unsqueeze(0)?; // [1, hidden]

            for ki in 0..k {
                let expert_idx = indices_vec[b][s][ki] as usize;
                let weight = weights_vec[b][s][ki];

                // GGUF [hidden, intermediate, n_experts] → candle [n_experts, intermediate, hidden]
                // Index expert on dim 0 → [intermediate, hidden]
                let gate_w = gate_f.i(expert_idx)?; // [intermediate, hidden]
                let up_w = up_f.i(expert_idx)?;
                let down_w = down_f.i(expert_idx)?; // [hidden, intermediate]

                // SwiGLU: down(silu(token @ gate^T) * (token @ up^T))
                let gate_out = token_2d.matmul(&gate_w.t()?)?; // [1, intermediate]
                let up_out = token_2d.matmul(&up_w.t()?)?;
                let activated = (candle_nn::ops::silu(&gate_out)? * up_out)?;
                let expert_out = activated.matmul(&down_w.t()?)?.squeeze(0)?; // [hidden]
                let expert_vec = expert_out.to_vec1::<f32>()?;

                let offset = (b * seq_len + s) * hidden;
                for i in 0..hidden {
                    output_data[offset + i] += expert_vec[i] * weight;
                }
            }
        }
    }

    Tensor::from_vec(output_data, (batch, seq_len, hidden), x.device()).map_err(Into::into)
}

// ── Shared Expert FFN ───────────────────────────────────────────

pub fn shared_expert_ffn(
    x: &Tensor,
    model: &LoadedModel,
    layer: u32,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);

    let gate_w = make_qmatmul(model, &format!("{}.ffn_gate_shexp.weight", prefix))?;
    let up_w = make_qmatmul(model, &format!("{}.ffn_up_shexp.weight", prefix))?;
    let down_w = make_qmatmul(model, &format!("{}.ffn_down_shexp.weight", prefix))?;

    let gate_out = gate_w.forward(x)?;
    let up_out = up_w.forward(x)?;
    let activated = (candle_nn::ops::silu(&gate_out)? * up_out)?;
    let out = down_w.forward(&activated)?;

    // Shared expert gating
    if let Some(gate_inp) = model.tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix)) {
        let gate_val = candle_nn::ops::sigmoid(&gate_inp.dequantize(x.device())?)?;
        return Ok(out.broadcast_mul(&gate_val)?);
    }
    Ok(out)
}

// ── Full forward pass (single layer) ────────────────────────────

pub fn forward_layer(
    x: &Tensor,
    model: &LoadedModel,
    layer: u32,
    kv_cache: &mut KvCache,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);
    let is_attn = is_attention_layer(layer);

    // 1. Pre-attention/SSM norm
    let norm_w = model.tensor(&format!("{}.attn_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing attn_norm for layer {}", layer))?;
    let norm_w_f = norm_w.dequantize(x.device())?;
    let normed = rms_norm(x, &norm_w_f, model.rms_norm_eps)?;

    // 2. Attention or SSM
    let block_out = if is_attn {
        self_attention(&normed, model, layer, kv_cache)?
    } else {
        ssm_block(&normed, model, layer)?
    };

    // Residual
    let x = (x + block_out)?;

    // 3. Post-attention/SSM norm
    let post_norm_w = model.tensor(&format!("{}.post_attention_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing post_attn_norm for layer {}", layer))?;
    let post_norm_f = post_norm_w.dequantize(x.device())?;
    let normed = rms_norm(&x, &post_norm_f, model.rms_norm_eps)?;

    // 4. MoE routing
    let gate_w = make_qmatmul(model, &format!("{}.ffn_gate_inp.weight", prefix))?;
    let (expert_indices, expert_weights) = moe_route(&normed, &gate_w, model.n_experts_used)?;

    // 5. Expert FFN (this crosses device boundary in split mode)
    let expert_out = expert_ffn(&normed, &expert_indices, &expert_weights, model, layer)?;

    // 6. Shared expert
    let shared_out = shared_expert_ffn(&normed, model, layer)?;

    // Combine + residual
    let ffn_out = (expert_out + shared_out)?;
    Ok((x + ffn_out)?)
}

// ── Full model forward pass ─────────────────────────────────────

pub fn forward(
    token_ids: &[u32],
    model: &LoadedModel,
    kv_caches: &mut Vec<KvCache>,
) -> Result<Tensor> {
    let device = &model.device;
    let seq_len = token_ids.len();

    // Embedding lookup
    let embd = model.tensor("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing token_embd"))?;
    let embd_f = embd.dequantize(&device)?; // [vocab, hidden]

    // Gather embeddings for input tokens
    let indices = Tensor::from_vec(
        token_ids.iter().map(|&id| id).collect::<Vec<u32>>(),
        (seq_len,),
        &device,
    )?;
    let x = embd_f.index_select(&indices, 0)?; // [seq_len, hidden]
    let x = x.unsqueeze(0)?; // [1, seq_len, hidden]

    // Forward through all layers
    let mut x = x;
    for layer in 0..model.n_layers {
        x = forward_layer(&x, model, layer, &mut kv_caches[layer as usize])?;
    }

    // Output norm
    let out_norm = model.tensor("output_norm.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing output_norm"))?;
    let out_norm_f = out_norm.dequantize(&device)?;
    let x = rms_norm(&x, &out_norm_f, model.rms_norm_eps)?;

    // Output projection (logits)
    let out_w = make_qmatmul(model, "output.weight")?;
    let logits = out_w.forward(&x)?; // [1, seq_len, vocab]

    Ok(logits)
}

// ── Split forward pass (expert FFN on remote device) ────────────

/// Forward pass with expert FFN offloaded to remote gRPC server.
/// SSM/attention/router run locally (fast GPU), experts run remotely (big memory).
pub async fn forward_layer_split(
    x: &Tensor,
    model: &LoadedModel,
    layer: u32,
    kv_cache: &mut KvCache,
    remote: &mut crate::expert_client::RemoteExpertClient,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);
    let is_attn = is_attention_layer(layer);

    // LOCAL: Pre-norm
    let norm_w = model.tensor(&format!("{}.attn_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing attn_norm for layer {}", layer))?;
    let norm_w_f = norm_w.dequantize(x.device())?;
    let normed = rms_norm(x, &norm_w_f, model.rms_norm_eps)?;

    // LOCAL: Attention or SSM
    let block_out = if is_attn {
        self_attention(&normed, model, layer, kv_cache)?
    } else {
        ssm_block(&normed, model, layer)?
    };
    let x = (x + block_out)?;

    // LOCAL: Post-norm
    let post_norm_w = model.tensor(&format!("{}.post_attention_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing post_attn_norm for layer {}", layer))?;
    let post_norm_f = post_norm_w.dequantize(x.device())?;
    let normed = rms_norm(&x, &post_norm_f, model.rms_norm_eps)?;

    // LOCAL: MoE routing (compute which experts to activate)
    let gate_w = make_qmatmul(model, &format!("{}.ffn_gate_inp.weight", prefix))?;
    let (expert_indices, expert_weights) = moe_route(&normed, &gate_w, model.n_experts_used)?;

    // REMOTE: Expert FFN + shared FFN on big memory device
    let ffn_out = remote.compute_layer(
        layer,
        &normed,
        &expert_indices,
        &expert_weights,
        model.n_experts_used,
    ).await?;

    // Move remote result to local device before residual add
    let ffn_out = ffn_out.to_device(x.device())?;

    // LOCAL: Residual
    Ok((x + ffn_out)?)
}

/// Full forward pass with split execution
pub async fn forward_split(
    token_ids: &[u32],
    model: &LoadedModel,
    kv_caches: &mut Vec<KvCache>,
    remote: &mut crate::expert_client::RemoteExpertClient,
) -> Result<Tensor> {
    let device = &model.device;
    let seq_len = token_ids.len();

    // Embedding lookup (on fast GPU — replicated tensor)
    let embd = model.tensor("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing token_embd"))?;
    let embd_f = embd.dequantize(&device)?;
    let indices = Tensor::from_vec(
        token_ids.to_vec(),
        (seq_len,),
        &device,
    )?;
    let x = embd_f.index_select(&indices, 0)?.unsqueeze(0)?;

    // Forward through all layers (split: local SSM/attn + remote experts)
    let mut x = x;
    let total_start = std::time::Instant::now();

    for layer in 0..model.n_layers {
        let layer_start = std::time::Instant::now();
        x = forward_layer_split(&x, model, layer, &mut kv_caches[layer as usize], remote).await?;

        if layer % 10 == 0 || layer == model.n_layers - 1 {
            tracing::info!(
                layer,
                elapsed_ms = layer_start.elapsed().as_millis(),
                "Layer complete"
            );
        }
    }

    tracing::info!(
        total_ms = total_start.elapsed().as_millis(),
        "All layers complete"
    );

    // Output norm + logits (on fast GPU)
    let out_norm = model.tensor("output_norm.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing output_norm"))?;
    let out_norm_f = out_norm.dequantize(&device)?;
    let x = rms_norm(&x, &out_norm_f, model.rms_norm_eps)?;
    let out_w = make_qmatmul(model, "output.weight")?;
    let logits = out_w.forward(&x)?;

    Ok(logits)
}

// ── Helpers ─────────────────────────────────────────────────────

fn make_qmatmul(model: &LoadedModel, name: &str) -> Result<QMatMul> {
    let t = model.tensor(name)
        .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
    QMatMul::from_arc(Arc::clone(t)).map_err(Into::into)
}
