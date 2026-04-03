//! Operation categorization and DAG for heterogeneous device placement.
//!
//! Categorizes every tensor in a GGUF model by its role in the forward pass,
//! then builds a DAG of operations with device placement tags.

use crate::gguf::GgufFile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Tensor categories ───────────────────────────────────────────

/// What role a tensor plays in the forward pass
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorCategory {
    /// Attention QKV projections, output projection, attention gate
    Attention,
    /// MoE router / gate weights (selects which experts to activate)
    Router,
    /// Expert FFN weights (the bulk of a MoE model)
    ExpertFfn,
    /// Shared expert FFN weights (always-active expert)
    SharedFfn,
    /// SSM / Mamba state-space model weights
    Ssm,
    /// Layer norms (RMS norm, post-attention norm)
    Norm,
    /// Token embedding and output projection
    Embedding,
}

impl std::fmt::Display for TensorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Attention => write!(f, "attention"),
            Self::Router => write!(f, "router"),
            Self::ExpertFfn => write!(f, "expert_ffn"),
            Self::SharedFfn => write!(f, "shared_ffn"),
            Self::Ssm => write!(f, "ssm"),
            Self::Norm => write!(f, "norm"),
            Self::Embedding => write!(f, "embedding"),
        }
    }
}

/// Where a tensor should be placed for optimal performance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevicePlacement {
    /// Fast GPU with high bandwidth (e.g., RTX 3070 Ti, 530 GB/s)
    /// For latency-sensitive operations: attention, KV cache, router
    FastGpu,
    /// High-capacity device (e.g., M4 Pro 24GB, 273 GB/s)
    /// For memory-heavy operations: expert weights
    BigMemory,
    /// Both devices need a copy (e.g., norms, embeddings)
    Replicated,
}

/// A categorized tensor with its device placement
#[derive(Debug, Clone)]
pub struct CategorizedTensor {
    pub name: String,
    pub category: TensorCategory,
    pub placement: DevicePlacement,
    pub layer: Option<u32>,
    pub shape: Vec<u64>,
    pub size_bytes: u64,
}

// ── Categorizer ─────────────────────────────────────────────────

/// Categorize all tensors in a GGUF file and assign device placements
pub fn categorize_tensors(gguf: &GgufFile) -> Vec<CategorizedTensor> {
    gguf.tensors
        .iter()
        .map(|t| {
            let (category, placement) = classify_tensor(&t.name);
            let layer = extract_layer(&t.name);

            CategorizedTensor {
                name: t.name.clone(),
                category,
                placement,
                layer,
                shape: t.shape.clone(),
                size_bytes: t.size_bytes,
            }
        })
        .collect()
}

fn classify_tensor(name: &str) -> (TensorCategory, DevicePlacement) {
    // Expert FFN: ffn_{gate,up,down}_exps — the bulk of MoE models
    if name.contains("ffn_gate_exps")
        || name.contains("ffn_up_exps")
        || name.contains("ffn_down_exps")
    {
        return (TensorCategory::ExpertFfn, DevicePlacement::BigMemory);
    }

    // Shared expert FFN: always-active expert (small)
    if name.contains("ffn_gate_shexp")
        || name.contains("ffn_up_shexp")
        || name.contains("ffn_down_shexp")
    {
        return (TensorCategory::SharedFfn, DevicePlacement::BigMemory);
    }

    // Router / MoE gate: selects which experts to activate
    if name.contains("ffn_gate_inp") {
        return (TensorCategory::Router, DevicePlacement::FastGpu);
    }

    // Attention weights: Q, K, V, output, gate
    if name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
        || name.contains("attn_gate")
        || name.contains("attn_qkv")
    {
        return (TensorCategory::Attention, DevicePlacement::FastGpu);
    }

    // SSM / Mamba weights
    if name.contains("ssm_") {
        return (TensorCategory::Ssm, DevicePlacement::FastGpu);
    }

    // Norms: tiny, replicate everywhere
    if name.contains("norm") {
        return (TensorCategory::Norm, DevicePlacement::Replicated);
    }

    // Embeddings and output projection
    if name.contains("token_embd") || name == "output.weight" {
        return (TensorCategory::Embedding, DevicePlacement::Replicated);
    }

    // Fallback: put on big memory device
    (TensorCategory::ExpertFfn, DevicePlacement::BigMemory)
}

fn extract_layer(name: &str) -> Option<u32> {
    // Pattern: blk.N.something
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && parts[0] == "blk" {
        parts[1].parse().ok()
    } else {
        None
    }
}

// ── Operation DAG ───────────────────────────────────────────────

/// A node in the inference DAG — represents a single operation in the forward pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpNode {
    pub id: usize,
    pub name: String,
    pub op_type: OpType,
    pub device: DevicePlacement,
    pub layer: u32,
    /// Tensor names consumed by this operation
    pub input_tensors: Vec<String>,
    /// Size of input activation (bytes, approximate)
    pub input_activation_bytes: u64,
    /// Size of output activation (bytes, approximate)
    pub output_activation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpType {
    Embedding,
    AttentionNorm,
    Attention,
    PostAttentionNorm,
    Ssm,
    Router,
    ExpertFfn,
    SharedFfn,
    OutputNorm,
    OutputProjection,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding => write!(f, "embed"),
            Self::AttentionNorm => write!(f, "attn_norm"),
            Self::Attention => write!(f, "attention"),
            Self::PostAttentionNorm => write!(f, "post_attn_norm"),
            Self::Ssm => write!(f, "ssm"),
            Self::Router => write!(f, "router"),
            Self::ExpertFfn => write!(f, "expert_ffn"),
            Self::SharedFfn => write!(f, "shared_ffn"),
            Self::OutputNorm => write!(f, "output_norm"),
            Self::OutputProjection => write!(f, "output_proj"),
        }
    }
}

/// Edge in the DAG — data transfer between operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpEdge {
    pub from: usize,
    pub to: usize,
    /// Whether this edge crosses device boundaries (requires network transfer)
    pub cross_device: bool,
    /// Bytes transferred across this edge
    pub transfer_bytes: u64,
}

/// The complete inference DAG for one forward pass
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceDag {
    pub nodes: Vec<OpNode>,
    pub edges: Vec<OpEdge>,
    pub n_layers: u32,
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub embedding_dim: u32,
}

impl InferenceDag {
    /// Build the DAG from a GGUF file's metadata
    pub fn build(gguf: &GgufFile) -> Self {
        let n_layers = gguf.n_layers().unwrap_or(1);
        let n_experts = gguf.n_experts().unwrap_or(1);
        let n_experts_used = gguf.n_experts_used().unwrap_or(1);
        let emb_dim = gguf.embedding_dim().unwrap_or(2048);
        let activation_bytes = (emb_dim as u64) * 4; // f32 hidden state

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut id = 0;

        // Embedding
        nodes.push(OpNode {
            id,
            name: "embedding".into(),
            op_type: OpType::Embedding,
            device: DevicePlacement::Replicated,
            layer: 0,
            input_tensors: vec!["token_embd.weight".into()],
            input_activation_bytes: 4, // token ID
            output_activation_bytes: activation_bytes,
        });
        let embed_id = id;
        id += 1;

        let mut prev_id = embed_id;

        // For each layer
        for layer in 0..n_layers {
            let prefix = format!("blk.{}", layer);

            // This model alternates: attention layers every 4th, SSM for the rest
            let is_attention_layer = (layer % 4) == 0 || layer == n_layers - 1;

            // Attention/SSM norm
            let norm_id = id;
            nodes.push(OpNode {
                id,
                name: format!("{}.attn_norm", prefix),
                op_type: OpType::AttentionNorm,
                device: DevicePlacement::FastGpu,
                layer,
                input_tensors: vec![format!("{}.attn_norm.weight", prefix)],
                input_activation_bytes: activation_bytes,
                output_activation_bytes: activation_bytes,
            });
            edges.push(OpEdge {
                from: prev_id,
                to: id,
                cross_device: false,
                transfer_bytes: activation_bytes,
            });
            id += 1;

            if is_attention_layer {
                // Attention
                let attn_id = id;
                nodes.push(OpNode {
                    id,
                    name: format!("{}.attention", prefix),
                    op_type: OpType::Attention,
                    device: DevicePlacement::FastGpu,
                    layer,
                    input_tensors: vec![
                        format!("{}.attn_qkv.weight", prefix),
                        format!("{}.attn_gate.weight", prefix),
                    ],
                    input_activation_bytes: activation_bytes,
                    output_activation_bytes: activation_bytes,
                });
                edges.push(OpEdge {
                    from: norm_id,
                    to: attn_id,
                    cross_device: false,
                    transfer_bytes: activation_bytes,
                });
                id += 1;
                prev_id = attn_id;
            } else {
                // SSM (Mamba) block
                let ssm_id = id;
                nodes.push(OpNode {
                    id,
                    name: format!("{}.ssm", prefix),
                    op_type: OpType::Ssm,
                    device: DevicePlacement::FastGpu,
                    layer,
                    input_tensors: vec![
                        format!("{}.ssm_a", prefix),
                        format!("{}.ssm_conv1d.weight", prefix),
                        format!("{}.ssm_dt.bias", prefix),
                        format!("{}.ssm_out.weight", prefix),
                    ],
                    input_activation_bytes: activation_bytes,
                    output_activation_bytes: activation_bytes,
                });
                edges.push(OpEdge {
                    from: norm_id,
                    to: ssm_id,
                    cross_device: false,
                    transfer_bytes: activation_bytes,
                });
                id += 1;
                prev_id = ssm_id;
            }

            // Post-attention norm
            let post_norm_id = id;
            nodes.push(OpNode {
                id,
                name: format!("{}.post_attn_norm", prefix),
                op_type: OpType::PostAttentionNorm,
                device: DevicePlacement::FastGpu,
                layer,
                input_tensors: vec![format!("{}.post_attention_norm.weight", prefix)],
                input_activation_bytes: activation_bytes,
                output_activation_bytes: activation_bytes,
            });
            edges.push(OpEdge {
                from: prev_id,
                to: post_norm_id,
                cross_device: false,
                transfer_bytes: activation_bytes,
            });
            id += 1;

            // Router (on fast GPU — needs to compute which experts to activate)
            let router_id = id;
            nodes.push(OpNode {
                id,
                name: format!("{}.router", prefix),
                op_type: OpType::Router,
                device: DevicePlacement::FastGpu,
                layer,
                input_tensors: vec![format!("{}.ffn_gate_inp.weight", prefix)],
                input_activation_bytes: activation_bytes,
                // Router outputs: hidden state + top-k expert indices
                output_activation_bytes: activation_bytes + (n_experts_used as u64 * 4),
            });
            edges.push(OpEdge {
                from: post_norm_id,
                to: router_id,
                cross_device: false,
                transfer_bytes: activation_bytes,
            });
            id += 1;

            // Expert FFN (on big memory device) — KEY CROSS-DEVICE EDGE
            let expert_id = id;
            nodes.push(OpNode {
                id,
                name: format!("{}.expert_ffn", prefix),
                op_type: OpType::ExpertFfn,
                device: DevicePlacement::BigMemory,
                layer,
                input_tensors: vec![
                    format!("{}.ffn_gate_exps.weight", prefix),
                    format!("{}.ffn_up_exps.weight", prefix),
                    format!("{}.ffn_down_exps.weight", prefix),
                ],
                // Only hidden state + expert indices sent (not all expert weights!)
                input_activation_bytes: activation_bytes + (n_experts_used as u64 * 4),
                output_activation_bytes: activation_bytes,
            });
            edges.push(OpEdge {
                from: router_id,
                to: expert_id,
                cross_device: true, // ← This is the network hop
                transfer_bytes: activation_bytes + (n_experts_used as u64 * 4),
            });
            id += 1;

            // Shared expert FFN (on big memory, runs in parallel with experts)
            let shared_id = id;
            nodes.push(OpNode {
                id,
                name: format!("{}.shared_ffn", prefix),
                op_type: OpType::SharedFfn,
                device: DevicePlacement::BigMemory,
                layer,
                input_tensors: vec![
                    format!("{}.ffn_gate_shexp.weight", prefix),
                    format!("{}.ffn_up_shexp.weight", prefix),
                    format!("{}.ffn_down_shexp.weight", prefix),
                ],
                input_activation_bytes: activation_bytes,
                output_activation_bytes: activation_bytes,
            });
            edges.push(OpEdge {
                from: router_id,
                to: shared_id,
                cross_device: true,
                transfer_bytes: activation_bytes,
            });
            id += 1;

            // Expert + shared outputs combine, send back to fast GPU for next layer
            // (implicit: the expert_ffn node output goes back to fast GPU)
            prev_id = expert_id;
        }

        // Output norm
        let out_norm_id = id;
        nodes.push(OpNode {
            id,
            name: "output_norm".into(),
            op_type: OpType::OutputNorm,
            device: DevicePlacement::FastGpu,
            layer: n_layers,
            input_tensors: vec!["output_norm.weight".into()],
            input_activation_bytes: activation_bytes,
            output_activation_bytes: activation_bytes,
        });
        edges.push(OpEdge {
            from: prev_id,
            to: out_norm_id,
            cross_device: true, // coming back from big memory
            transfer_bytes: activation_bytes,
        });
        id += 1;

        // Output projection
        nodes.push(OpNode {
            id,
            name: "output".into(),
            op_type: OpType::OutputProjection,
            device: DevicePlacement::Replicated,
            layer: n_layers,
            input_tensors: vec!["output.weight".into()],
            input_activation_bytes: activation_bytes,
            output_activation_bytes: 248320 * 4, // vocab size * f32
        });
        edges.push(OpEdge {
            from: out_norm_id,
            to: id,
            cross_device: false,
            transfer_bytes: activation_bytes,
        });

        Self {
            nodes,
            edges,
            n_layers,
            n_experts,
            n_experts_used,
            embedding_dim: emb_dim,
        }
    }

    /// Count cross-device transfers per forward pass
    pub fn cross_device_transfers(&self) -> (usize, u64) {
        let count = self.edges.iter().filter(|e| e.cross_device).count();
        let bytes: u64 = self
            .edges
            .iter()
            .filter(|e| e.cross_device)
            .map(|e| e.transfer_bytes)
            .sum();
        (count, bytes)
    }

    /// Compute total memory needed per device
    pub fn memory_by_device(&self, tensors: &[CategorizedTensor]) -> HashMap<DevicePlacement, f64> {
        let mut mem: HashMap<DevicePlacement, f64> = HashMap::new();
        for t in tensors {
            *mem.entry(t.placement).or_default() += t.size_bytes as f64 / (1024.0 * 1024.0);
        }
        mem
    }

    pub fn summary(&self, tensors: &[CategorizedTensor]) -> String {
        let mem = self.memory_by_device(tensors);
        let (xfer_count, xfer_bytes) = self.cross_device_transfers();

        let mut s = String::new();
        s.push_str(&format!("Model: {} layers, {} experts ({} active/token)\n",
            self.n_layers, self.n_experts, self.n_experts_used));
        s.push_str(&format!("Embedding dim: {}\n\n", self.embedding_dim));

        s.push_str("Device memory requirements:\n");
        for (device, mb) in &mem {
            s.push_str(&format!("  {:?}: {:.0} MB\n", device, mb));
        }

        s.push_str(&format!("\nForward pass DAG: {} ops, {} edges\n", self.nodes.len(), self.edges.len()));
        s.push_str(&format!("Cross-device transfers per token: {} ({:.1} KB)\n",
            xfer_count, xfer_bytes as f64 / 1024.0));

        // Network bandwidth estimate
        let latency_1gbps_ms = (xfer_bytes as f64 / (125.0 * 1024.0 * 1024.0)) * 1000.0;
        s.push_str(&format!("Estimated network overhead per token @ 1 Gbps: {:.2} ms\n", latency_1gbps_ms));

        s
    }
}
