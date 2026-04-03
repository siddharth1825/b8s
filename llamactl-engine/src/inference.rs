//! Inference operations for heterogeneous split execution.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_core::quantized::{QMatMul, QTensor};
use std::sync::Arc;

use crate::loader::LoadedModel;

// ── RMS Norm ────────────────────────────────────────────────────

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = (&x_f32 * &x_f32)?.mean_keepdim(candle_core::D::Minus1)?;
    let x_norm = x_f32.broadcast_div(&(variance + eps as f64)?.sqrt()?)?;
    let out = x_norm.to_dtype(dtype)?.broadcast_mul(weight)?;
    Ok(out)
}

// ── Attention (GQA with KV cache) ───────────────────────────────

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

/// Grouped-Query Attention
pub fn attention(
    x: &Tensor,
    qkv_weight: &QMatMul,
    attn_gate: Option<&QMatMul>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_cache: &mut KvCache,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;

    // QKV projection — Qwen3.5 fuses Q, K, V into one weight
    // Output dim may not match n_heads*head_dim + 2*n_kv_heads*head_dim exactly
    // because K/V may use expanded dims with separate Q/K norms
    let qkv = qkv_weight.forward(x)?;
    let qkv_dim = qkv.dim(candle_core::D::Minus1)?;

    // Split: Q gets n_heads shares, K and V each get n_kv_heads shares
    let total_heads = n_heads + 2 * n_kv_heads;
    let per_head = qkv_dim / total_heads;
    let q_dim = n_heads * per_head;
    let kv_dim = n_kv_heads * per_head;

    let q = qkv.narrow(candle_core::D::Minus1, 0, q_dim)?;
    let k = qkv.narrow(candle_core::D::Minus1, q_dim, kv_dim)?;
    let v = qkv.narrow(candle_core::D::Minus1, q_dim + kv_dim, kv_dim)?;

    // Reshape to [batch, n_heads, seq_len, per_head]
    let q = q.reshape((batch, seq_len, n_heads, per_head))?.transpose(1, 2)?;
    let k = k.reshape((batch, seq_len, n_kv_heads, per_head))?.transpose(1, 2)?;
    let v = v.reshape((batch, seq_len, n_kv_heads, per_head))?.transpose(1, 2)?;

    // GQA: repeat KV heads to match Q heads
    let repeats = n_heads / n_kv_heads;
    let k = if repeats > 1 { k.repeat(&[1, repeats, 1, 1])? } else { k };
    let v = if repeats > 1 { v.repeat(&[1, repeats, 1, 1])? } else { v };

    // KV cache
    let (k, v) = kv_cache.append(k, v)?;

    // Scaled dot-product attention
    let scale = (per_head as f64).sqrt();
    let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

    // Causal mask (manual, since candle doesn't have triu)
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
    let attn_out = attn_out.transpose(1, 2)?.reshape((batch, seq_len, q_dim))?;

    // Attention gate
    if let Some(gate) = attn_gate {
        return Ok(gate.forward(&attn_out)?);
    }
    Ok(attn_out)
}

// ── MoE Router ──────────────────────────────────────────────────

/// Returns: (expert_indices [batch*seq, k], expert_weights [batch*seq, k])
pub fn moe_route(
    x: &Tensor,
    gate_weight: &QMatMul,
    n_experts_used: u32,
) -> Result<(Tensor, Tensor)> {
    let logits = gate_weight.forward(x)?; // [batch, seq_len, n_experts]
    let k = n_experts_used as usize;

    // Softmax over experts
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

    // Top-k via argsort
    let sorted_indices = probs.arg_sort_last_dim(false)?; // descending
    let top_k_indices = sorted_indices.narrow(candle_core::D::Minus1, 0, k)?;

    // Gather top-k weights
    let top_k_weights = probs.gather(&top_k_indices, candle_core::D::Minus1)?;

    // Renormalize
    let weight_sum = top_k_weights.sum_keepdim(candle_core::D::Minus1)?;
    let top_k_weights = top_k_weights.broadcast_div(&weight_sum)?;

    Ok((top_k_indices, top_k_weights))
}

// ── Expert FFN (sparse) ─────────────────────────────────────────

/// Run sparse expert FFN on activated experts only.
/// Expert weights are 3D: [dim_out, dim_in, n_experts] (gate/up) or [dim_in, dim_out, n_experts] (down)
pub fn expert_ffn(
    x: &Tensor,
    expert_indices: &Tensor,
    expert_weights: &Tensor,
    gate_exps: &QTensor,
    up_exps: &QTensor,
    down_exps: &QTensor,
) -> Result<Tensor> {
    let (batch, seq_len, hidden) = x.dims3()?;
    let k = expert_indices.dim(2)?;

    // Dequantize to f32 for computation
    let gate_f = gate_exps.dequantize(x.device())?;
    let up_f = up_exps.dequantize(x.device())?;
    let down_f = down_exps.dequantize(x.device())?;

    let indices_vec = expert_indices.to_dtype(DType::U32)?.to_vec3::<u32>()?;
    let weights_vec = expert_weights.to_vec3::<f32>()?;

    // Process token by token (can be batched/parallelized later)
    let mut output_data = vec![0f32; batch * seq_len * hidden];

    for b in 0..batch {
        for s in 0..seq_len {
            let token = x.i((b, s))?.to_vec1::<f32>()?;
            let token_t = Tensor::from_vec(token.clone(), (1, hidden), x.device())?;
            let mut combined = vec![0f32; hidden];

            for ki in 0..k {
                let expert_idx = indices_vec[b][s][ki] as usize;
                let weight = weights_vec[b][s][ki];

                // gate_exps: [hidden, intermediate, n_experts]
                let gate_w = gate_f.i((.., .., expert_idx))?;
                let up_w = up_f.i((.., .., expert_idx))?;
                let down_w = down_f.i((.., .., expert_idx))?;

                // SwiGLU: down(silu(gate(x)) * up(x))
                let gate_out = token_t.matmul(&gate_w)?;
                let up_out = token_t.matmul(&up_w)?;
                let activated = (candle_nn::ops::silu(&gate_out)? * up_out)?;
                let expert_out = activated.matmul(&down_w.t()?)?.squeeze(0)?.to_vec1::<f32>()?;

                for i in 0..hidden {
                    combined[i] += expert_out[i] * weight;
                }
            }

            let offset = (b * seq_len + s) * hidden;
            output_data[offset..offset + hidden].copy_from_slice(&combined);
        }
    }

    let output = Tensor::from_vec(output_data, (batch, seq_len, hidden), x.device())?;
    Ok(output)
}

// ── Shared Expert FFN ───────────────────────────────────────────

pub fn shared_expert_ffn(
    x: &Tensor,
    gate_weight: &QMatMul,
    up_weight: &QMatMul,
    down_weight: &QMatMul,
    gate_inp_shexp: Option<&Tensor>,
) -> Result<Tensor> {
    let gate_out = gate_weight.forward(x)?;
    let up_out = up_weight.forward(x)?;
    let activated = (candle_nn::ops::silu(&gate_out)? * up_out)?;
    let out = down_weight.forward(&activated)?;

    if let Some(gate) = gate_inp_shexp {
        let gate_val = candle_nn::ops::sigmoid(gate)?;
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
    is_attention_layer: bool,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer);

    // 1. Attention/SSM norm
    let norm_w = model.tensor(&format!("{}.attn_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing attn_norm for layer {}", layer))?;
    let norm_w = norm_w.dequantize(x.device())?;
    let normed = rms_norm(x, &norm_w, model.rms_norm_eps)?;

    // 2. Attention or SSM
    let attn_out = if is_attention_layer {
        let qkv_t = model.tensor(&format!("{}.attn_qkv.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing attn_qkv for layer {}", layer))?;
        let qkv_w = QMatMul::from_arc(Arc::clone(qkv_t))?;

        let gate_t = model.tensor(&format!("{}.attn_gate.weight", prefix));
        let gate_w = match gate_t {
            Some(t) => Some(QMatMul::from_arc(Arc::clone(t))?),
            None => None,
        };

        attention(
            &normed, &qkv_w, gate_w.as_ref(),
            model.n_heads, model.n_kv_heads, model.head_dim,
            kv_cache,
        )?
    } else {
        // SSM placeholder — pass through for Phase 2c
        normed.clone()
    };

    // Residual
    let x = (x + attn_out)?;

    // 3. Post-attention norm
    let post_norm_w = model.tensor(&format!("{}.post_attention_norm.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing post_attn_norm for layer {}", layer))?;
    let post_norm_w = post_norm_w.dequantize(x.device())?;
    let normed = rms_norm(&x, &post_norm_w, model.rms_norm_eps)?;

    // 4. MoE routing
    let gate_t = model.tensor(&format!("{}.ffn_gate_inp.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_gate_inp for layer {}", layer))?;
    let gate_w = QMatMul::from_arc(Arc::clone(gate_t))?;
    let (expert_indices, expert_weights) = moe_route(&normed, &gate_w, model.n_experts_used)?;

    // 5. Expert FFN
    let gate_exps = model.tensor(&format!("{}.ffn_gate_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_gate_exps for layer {}", layer))?;
    let up_exps = model.tensor(&format!("{}.ffn_up_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_up_exps for layer {}", layer))?;
    let down_exps = model.tensor(&format!("{}.ffn_down_exps.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_down_exps for layer {}", layer))?;

    let expert_out = expert_ffn(&normed, &expert_indices, &expert_weights, gate_exps.as_ref(), up_exps.as_ref(), down_exps.as_ref())?;

    // 6. Shared expert
    let sh_gate_t = model.tensor(&format!("{}.ffn_gate_shexp.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_gate_shexp for layer {}", layer))?;
    let sh_up_t = model.tensor(&format!("{}.ffn_up_shexp.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_up_shexp for layer {}", layer))?;
    let sh_down_t = model.tensor(&format!("{}.ffn_down_shexp.weight", prefix))
        .ok_or_else(|| anyhow::anyhow!("Missing ffn_down_shexp for layer {}", layer))?;
    let sh_gate_w = QMatMul::from_arc(Arc::clone(sh_gate_t))?;
    let sh_up_w = QMatMul::from_arc(Arc::clone(sh_up_t))?;
    let sh_down_w = QMatMul::from_arc(Arc::clone(sh_down_t))?;

    let sh_gate_inp = model.tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix))
        .map(|t| t.dequantize(x.device()))
        .transpose()?;

    let shared_out = shared_expert_ffn(&normed, &sh_gate_w, &sh_up_w, &sh_down_w, sh_gate_inp.as_ref())?;

    // Combine + residual
    let ffn_out = (expert_out + shared_out)?;
    Ok((x + ffn_out)?)
}
