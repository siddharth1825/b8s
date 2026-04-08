mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llamactl_common::*;

#[derive(Parser)]
#[command(name = "llamactl", about = "kubectl for local LLM clusters")]
struct Cli {
    /// Server URL
    #[arg(long, env = "LLAMACTL_SERVER", default_value = "http://localhost:6969")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get cluster resources
    Get {
        #[command(subcommand)]
        resource: GetResource,
    },
    /// Deploy a model to the cluster
    Deploy {
        /// Model identifier (HuggingFace repo or local path to .gguf)
        model: String,
        /// Pin to a specific node ID
        #[arg(long)]
        node: Option<uuid::Uuid>,
        /// Inference server port on the worker (default: 8080)
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Context length
        #[arg(long, default_value = "2048")]
        context_length: u32,
        /// GPU layers to offload (default: all)
        #[arg(long, default_value = "9999")]
        gpu_layers: u32,
        /// Split model across multiple nodes
        /// Comma-separated node IDs: first is main, rest are workers
        #[arg(long, value_delimiter = ',')]
        split: Vec<uuid::Uuid>,
        /// Split backend: "rpc" (llama.cpp RPC, default) or "mlx" (MLX Distributed ring)
        #[arg(long, default_value = "rpc")]
        backend: String,
    },
    /// Stop and remove a deployment
    Stop {
        /// Deployment ID
        deployment_id: uuid::Uuid,
    },
    /// Pause a running deployment (keeps it in the DB, stops the process)
    Pause {
        /// Deployment ID
        deployment_id: uuid::Uuid,
    },
    /// Resume a paused deployment or restart a failed/stopped one
    Resume {
        /// Deployment ID
        deployment_id: uuid::Uuid,
    },
    /// Remove a stale/offline node
    RemoveNode {
        /// Node ID
        node_id: uuid::Uuid,
    },
    /// Live interactive cluster dashboard
    Top,
    /// Analyze a GGUF model for heterogeneous split inference
    Analyze {
        /// Path to GGUF model file
        model: String,
        /// Actually load all tensors and test a single forward pass
        #[arg(long)]
        load: bool,
    },
    /// Repack GGUF expert tensors into per-layer files for SSD streaming
    Repack {
        /// Path to GGUF model file
        model: String,
        /// Output directory (default: <model_dir>/packed_experts/)
        #[arg(long, short)]
        output: Option<String>,
    },
    /// Run inference on a model (auto-selects strategy based on hardware)
    Serve {
        /// Path to GGUF model file
        model: String,
        /// Port for the OpenAI-compatible API
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Context length
        #[arg(long, short, default_value = "512")]
        ctx: u32,
        /// Number of hot experts to pin in RAM (0 = none, auto = based on available RAM)
        #[arg(long, default_value = "0")]
        pin_top: usize,
        /// Force strategy: "auto", "full-ram", "flash-ssd"
        #[arg(long, default_value = "auto")]
        strategy: String,
        /// Path to llama-server binary (auto-detected if not specified)
        #[arg(long)]
        llama_server: Option<String>,
    },
    /// Benchmark inference: local-only and split modes
    Bench {
        /// Path to GGUF model file
        model: String,
        /// Expert worker gRPC address (for split mode)
        #[arg(long)]
        expert_addr: Option<String>,
        /// Number of tokens to generate for benchmark
        #[arg(long, default_value = "5")]
        n_tokens: usize,
        /// Force CPU (avoids OOM when full model doesn't fit in GPU memory)
        #[arg(long)]
        cpu: bool,
    },
}

#[derive(Subcommand)]
enum GetResource {
    /// List all registered nodes
    Nodes,
    /// List all deployments
    Deployments,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();

    match cli.command {
        Commands::Get { resource } => match resource {
            GetResource::Nodes => get_nodes(&cli.server).await?,
            GetResource::Deployments => get_deployments(&cli.server).await?,
        },
        Commands::Deploy {
            model, node, port, context_length, gpu_layers, split, backend,
        } => deploy(&cli.server, &client, model, node, port, context_length, gpu_layers, split, &backend).await?,
        Commands::Stop { deployment_id } => {
            client
                .delete(format!("{}/api/deployments/{deployment_id}", cli.server))
                .send()
                .await
                .context("Failed to connect to server")?
                .error_for_status()
                .context("Deployment not found")?;
            println!("Deployment {} deleted.", &deployment_id.to_string()[..8]);
        }
        Commands::Pause { deployment_id } => {
            let req = DeploymentControlRequest { action: DeploymentAction::Pause };
            client
                .put(format!("{}/api/deployments/{deployment_id}", cli.server))
                .json(&req)
                .send()
                .await
                .context("Failed to connect to server")?
                .error_for_status()
                .context("Failed to pause")?;
            println!("Deployment {} paused.", &deployment_id.to_string()[..8]);
        }
        Commands::Resume { deployment_id } => {
            let req = DeploymentControlRequest { action: DeploymentAction::Resume };
            client
                .put(format!("{}/api/deployments/{deployment_id}", cli.server))
                .json(&req)
                .send()
                .await
                .context("Failed to connect to server")?
                .error_for_status()
                .context("Failed to resume")?;
            println!("Deployment {} resumed.", &deployment_id.to_string()[..8]);
        }
        Commands::RemoveNode { node_id } => {
            client
                .delete(format!("{}/api/nodes/{node_id}", cli.server))
                .send()
                .await
                .context("Failed to connect to server")?
                .error_for_status()
                .context("Node not found")?;
            println!("Node {} removed.", &node_id.to_string()[..8]);
        }
        Commands::Top => tui::run(&cli.server).await?,
        Commands::Analyze { model, load } => analyze_model(&model, load)?,
        Commands::Repack { model, output } => repack_experts(&model, output.as_deref())?,
        Commands::Serve { model, port, ctx, pin_top, strategy, llama_server } =>
            serve_model(&model, port, ctx, pin_top, &strategy, llama_server.as_deref()).await?,
        Commands::Bench { model, expert_addr, n_tokens, cpu } => bench_inference(&model, expert_addr.as_deref(), n_tokens, cpu).await?,
    }

    Ok(())
}

fn analyze_model(path: &str, load_tensors: bool) -> Result<()> {
    use llamactl_engine::{categorize_tensors, GgufFile, InferenceDag, TensorCategory, DevicePlacement};
    use std::collections::HashMap;

    println!("Loading GGUF: {}\n", path);
    let gguf = GgufFile::open(path)?;

    println!("Architecture: {}", gguf.architecture().unwrap_or("unknown"));
    println!("Model:        {}", gguf.model_name().unwrap_or("unknown"));
    println!("Version:      GGUF v{}", gguf.version);
    println!("Tensors:      {}", gguf.tensors.len());
    println!("Layers:       {}", gguf.n_layers().unwrap_or(0));
    println!("Experts:      {} ({} active/token)",
        gguf.n_experts().unwrap_or(0), gguf.n_experts_used().unwrap_or(0));
    println!("Embedding:    {}", gguf.embedding_dim().unwrap_or(0));
    println!("Total size:   {:.1} MB ({:.2} GB)\n",
        gguf.total_size_mb(), gguf.total_size_mb() / 1024.0);

    // Categorize tensors
    let categorized = categorize_tensors(&gguf);

    // Summary by category
    let mut by_category: HashMap<TensorCategory, (usize, f64)> = HashMap::new();
    for t in &categorized {
        let entry = by_category.entry(t.category).or_default();
        entry.0 += 1;
        entry.1 += t.size_bytes as f64 / (1024.0 * 1024.0);
    }

    let total_mb: f64 = by_category.values().map(|v| v.1).sum();

    println!("Tensor breakdown by operation type:");
    println!("  {:<15} {:>6} {:>10} {:>6}", "CATEGORY", "COUNT", "SIZE (MB)", "%");
    let mut sorted: Vec<_> = by_category.iter().collect();
    sorted.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (cat, (count, mb)) in &sorted {
        println!("  {:<15} {:>6} {:>10.1} {:>5.1}%", cat, count, mb, mb / total_mb * 100.0);
    }

    // Summary by device
    let mut by_device: HashMap<DevicePlacement, f64> = HashMap::new();
    for t in &categorized {
        *by_device.entry(t.placement).or_default() += t.size_bytes as f64 / (1024.0 * 1024.0);
    }

    println!("\nDevice placement:");
    for (device, mb) in &by_device {
        let label = match device {
            DevicePlacement::FastGpu => "Fast GPU (attention, router, SSM)",
            DevicePlacement::BigMemory => "Big Memory (experts, shared FFN)",
            DevicePlacement::Replicated => "Replicated (norms, embeddings)",
        };
        println!("  {:?}: {:.0} MB — {}", device, mb, label);
    }

    // Build and analyze DAG
    let dag = InferenceDag::build(&gguf);
    let (xfer_count, xfer_bytes) = dag.cross_device_transfers();

    println!("\nInference DAG:");
    println!("  Operations per forward pass: {}", dag.nodes.len());
    println!("  Edges: {}", dag.edges.len());
    println!("  Cross-device transfers/token: {} ({:.1} KB)",
        xfer_count, xfer_bytes as f64 / 1024.0);

    // Network overhead estimates
    println!("\nNetwork overhead per token:");
    for (name, bw_gbps) in [("1 Gbps ethernet", 1.0), ("10 Gbps ethernet", 10.0), ("Thunderbolt 3", 40.0)] {
        let ms = (xfer_bytes as f64 / (bw_gbps * 125.0 * 1024.0 * 1024.0)) * 1000.0;
        println!("  {}: {:.2} ms", name, ms);
    }

    // Feasibility check for specific hardware
    println!("\nHardware fit check:");
    let fast_mb = by_device.get(&DevicePlacement::FastGpu).unwrap_or(&0.0)
        + by_device.get(&DevicePlacement::Replicated).unwrap_or(&0.0);
    let big_mb = by_device.get(&DevicePlacement::BigMemory).unwrap_or(&0.0)
        + by_device.get(&DevicePlacement::Replicated).unwrap_or(&0.0);

    for (name, vram_mb) in [("RTX 3070 Ti 8GB", 8192.0), ("RTX 4090 24GB", 24576.0), ("M4 Pro 24GB", 24576.0), ("M4 Max 128GB", 131072.0)] {
        let fast_fit = if fast_mb < vram_mb { "OK" } else { "TOO BIG" };
        let big_fit = if big_mb < vram_mb { "OK" } else { "TOO BIG" };
        println!("  {} as fast GPU: {:.0}/{:.0} MB [{}]", name, fast_mb, vram_mb, fast_fit);
        println!("  {} as big memory: {:.0}/{:.0} MB [{}]", name, big_mb, vram_mb, big_fit);
    }

    // ── Expert router analysis ──────────────────────────────────
    // Analyze router gate weights to understand expert selection patterns
    {
        let n_layers = gguf.n_layers().unwrap_or(0) as usize;
        let n_experts = gguf.n_experts().unwrap_or(0) as usize;
        let emb_dim = gguf.embedding_dim().unwrap_or(0) as usize;

        if n_experts > 1 && emb_dim > 0 {
            println!("\nExpert router analysis:");

            // Per-expert stats across layers
            let mut expert_total_norm = vec![0.0f64; n_experts];
            let mut expert_max_norm = vec![0.0f64; n_experts];
            let mut layers_analyzed = 0usize;

            // Per-layer diversity (how spread out are expert norms)
            let mut layer_diversity = Vec::new();

            for layer in 0..n_layers {
                let gate_name = format!("blk.{}.ffn_gate_inp.weight", layer);
                let tensor = match gguf.find_tensor(&gate_name) {
                    Some(t) => t,
                    None => continue,
                };

                // Router gate is typically F32: [n_expert, n_embd] (GGUF dims reversed)
                // For F32, we can read directly. For quantized, skip.
                if tensor.dtype != llamactl_engine::gguf::GgmlType::F32 {
                    if layer == 0 {
                        println!("  Router gate is {:?} (not F32) — skipping detailed analysis", tensor.dtype);
                    }
                    continue;
                }

                let data = gguf.tensor_data(tensor);
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                };

                // Compute L2 norm per expert row
                // GGUF shape [n_embd, n_expert] but stored row-major as [n_expert, n_embd]
                let mut expert_norms = vec![0.0f64; n_experts];
                for e in 0..n_experts {
                    let row_start = e * emb_dim;
                    let row_end = row_start + emb_dim;
                    if row_end > floats.len() { break; }
                    let norm: f64 = floats[row_start..row_end]
                        .iter()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum::<f64>()
                        .sqrt();
                    expert_norms[e] = norm;
                    expert_total_norm[e] += norm;
                    if norm > expert_max_norm[e] {
                        expert_max_norm[e] = norm;
                    }
                }

                // Layer diversity: std dev of expert norms
                let mean: f64 = expert_norms.iter().sum::<f64>() / n_experts as f64;
                let var: f64 = expert_norms.iter().map(|n| (n - mean).powi(2)).sum::<f64>() / n_experts as f64;
                layer_diversity.push((layer, mean, var.sqrt()));
                layers_analyzed += 1;
            }

            if layers_analyzed > 0 {
                // Average norm per expert across layers
                let avg_norms: Vec<f64> = expert_total_norm.iter()
                    .map(|n| n / layers_analyzed as f64)
                    .collect();

                // Find hottest and coldest experts
                let mut indexed: Vec<(usize, f64)> = avg_norms.iter()
                    .enumerate()
                    .map(|(i, &n)| (i, n))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let global_mean = avg_norms.iter().sum::<f64>() / n_experts as f64;
                let global_std = (avg_norms.iter().map(|n| (n - global_mean).powi(2)).sum::<f64>() / n_experts as f64).sqrt();

                println!("  Layers analyzed: {} (F32 router gates)", layers_analyzed);
                println!("  Expert routing weight norm (avg across layers):");
                println!("    Mean:    {:.4}", global_mean);
                println!("    Std dev: {:.4}  (higher = more expert specialization)", global_std);
                println!("    CV:      {:.1}%  (coefficient of variation)", global_std / global_mean * 100.0);

                println!("\n  All experts ranked by routing weight norm (descending):");
                println!("    {:>7} {:>8} {:>7}", "EXPERT", "NORM", "SIGMA");
                for &(idx, norm) in &indexed {
                    let sigma = (norm - global_mean) / global_std;
                    let marker = if sigma > 1.0 { " HOT" } else if sigma < -1.0 { " COLD" } else { "" };
                    println!("    {:>7} {:>8.4} {:>+6.1}σ{}", idx, norm, sigma, marker);
                }

                // Expert similarity (cosine similarity between top experts)
                // Check how many experts are "redundant"
                let high_threshold = global_mean + global_std;
                let low_threshold = global_mean - global_std;
                let hot_count = avg_norms.iter().filter(|&&n| n > high_threshold).count();
                let cold_count = avg_norms.iter().filter(|&&n| n < low_threshold).count();
                let mid_count = n_experts - hot_count - cold_count;

                println!("\n  Expert activity distribution:");
                println!("    Hot  (>+1σ): {:>4} experts ({:.1}%) — frequently selected",
                    hot_count, hot_count as f64 / n_experts as f64 * 100.0);
                println!("    Mid  (±1σ):  {:>4} experts ({:.1}%) — average selection rate",
                    mid_count, mid_count as f64 / n_experts as f64 * 100.0);
                println!("    Cold (<-1σ): {:>4} experts ({:.1}%) — rarely selected",
                    cold_count, cold_count as f64 / n_experts as f64 * 100.0);

                // Layer diversity analysis
                println!("\n  Layer-wise routing diversity (higher = more expert differentiation):");
                let early = &layer_diversity[..std::cmp::min(5, layer_diversity.len())];
                let late = &layer_diversity[layer_diversity.len().saturating_sub(5)..];
                let early_std: f64 = early.iter().map(|(_, _, s)| s).sum::<f64>() / early.len() as f64;
                let late_std: f64 = late.iter().map(|(_, _, s)| s).sum::<f64>() / late.len() as f64;
                println!("    Early layers (0-4) avg std: {:.4}", early_std);
                println!("    Late layers ({}-{}) avg std:  {:.4}",
                    layer_diversity.len() - late.len(), layer_diversity.len() - 1, late_std);
                if late_std > early_std * 1.2 {
                    println!("    → Late layers have more specialized routing (typical)");
                } else if early_std > late_std * 1.2 {
                    println!("    → Early layers have more specialized routing (unusual)");
                } else {
                    println!("    → Routing specialization is uniform across layers");
                }

                // SSD streaming estimate
                // Expert tensors are stacked: each tensor (gate_exps, up_exps, down_exps) contains
                // all n_experts stacked along one dimension. A single expert needs 1/n_experts of each.
                // There are 3 tensors per layer, so per-expert size = total_expert_MB / n_layers / n_experts
                let n_experts_used = gguf.n_experts_used().unwrap_or(8) as usize;
                let total_expert_mb = by_category.get(&TensorCategory::ExpertFfn)
                    .map(|(_count, mb)| *mb)
                    .unwrap_or(0.0);
                let single_expert_mb = total_expert_mb / n_layers as f64 / n_experts as f64;

                println!("\n  SSD streaming estimate (flash-expert mode):");
                println!("    Single expert size: {:.2} MB (all 3 projections)", single_expert_mb);
                println!("    Per-layer IO: {} experts × {:.2} MB = {:.1} MB",
                    n_experts_used, single_expert_mb, n_experts_used as f64 * single_expert_mb);
                let per_token_mb = n_layers as f64 * n_experts_used as f64 * single_expert_mb;
                println!("    Per-token IO (cold): {:.0} MB across {} layers", per_token_mb, n_layers);

                for (name, gbps) in [("Apple NVMe (M4 Pro)", 7.0), ("Windows NVMe (typical)", 3.5), ("SATA SSD", 0.5)] {
                    let ms = per_token_mb / (gbps * 1024.0) * 1000.0;
                    println!("    {} @ {} GB/s: {:.0} ms/token (cold), ~{:.1} tok/s max",
                        name, gbps, ms, 1000.0 / ms);
                }

                // Page cache estimate
                println!("\n    With OS page cache (~70% hit rate):");
                let cached_per_token_mb = per_token_mb * 0.3; // only 30% are cache misses
                for (name, gbps) in [("Apple NVMe (M4 Pro)", 7.0), ("Windows NVMe (typical)", 3.5)] {
                    let ms = cached_per_token_mb / (gbps * 1024.0) * 1000.0;
                    println!("    {} @ {} GB/s: {:.0} ms/token, ~{:.1} tok/s",
                        name, gbps, ms, 1000.0 / ms);
                }
            }
        }
    }

    if load_tensors {
        println!("\n--- Loading tensors into memory ---\n");
        let start = std::time::Instant::now();
        let model = llamactl_engine::LoadedModel::load(path)?;
        let elapsed = start.elapsed();
        println!("Loaded in {:.2}s", elapsed.as_secs_f64());
        println!("{}", model.memory_summary());
        println!("\nModel config: {} layers, {} heads ({} KV), head_dim={}, vocab={}",
            model.n_layers, model.n_heads, model.n_kv_heads, model.head_dim, model.vocab_size);

        // Test single token forward pass through layer 0
        println!("\n--- Testing forward pass (layer 0) ---\n");
        let device = candle_core::Device::Cpu;
        let hidden = model.embedding_dim as usize;

        // Create a random input (simulating embedding output)
        let x = candle_core::Tensor::randn(0f32, 1.0, (1, 1, hidden), &device)?;
        println!("Input shape: {:?}", x.shape());

        let mut kv_cache = llamactl_engine::inference::KvCache::new();
        let start = std::time::Instant::now();
        let output = llamactl_engine::inference::forward_layer(&x, &model, 0, &mut kv_cache)?;
        let elapsed = start.elapsed();

        println!("Output shape: {:?}", output.shape());
        println!("Layer 0 forward pass: {:.2}ms", elapsed.as_millis());
        let sample = output.flatten_all()?.narrow(0, 0, 5)?.to_vec1::<f32>()?;
        println!("Output sample: {:?}", sample);
        println!("\nPhase 2 forward pass working!");
    }

    Ok(())
}

async fn serve_model(
    model_path: &str,
    port: u16,
    ctx: u32,
    pin_top: usize,
    strategy: &str,
    llama_server_path: Option<&str>,
) -> Result<()> {
    use llamactl_engine::{GgufFile, categorize_tensors, TensorCategory, DevicePlacement};
    use std::collections::HashMap;
    use std::process::Command;

    println!("=== llamactl serve ===\n");

    // Step 1: Analyze the model
    let gguf = GgufFile::open(model_path)?;
    let n_layers = gguf.n_layers().unwrap_or(0) as usize;
    let n_experts = gguf.n_experts().unwrap_or(0) as usize;
    let n_experts_used = gguf.n_experts_used().unwrap_or(0) as usize;
    let model_name = gguf.model_name().unwrap_or("unknown").to_string();

    let categorized = categorize_tensors(&gguf);
    let mut by_placement: HashMap<DevicePlacement, f64> = HashMap::new();
    for t in &categorized {
        *by_placement.entry(t.placement).or_default() += t.size_bytes as f64 / (1024.0 * 1024.0);
    }

    let total_mb: f64 = categorized.iter().map(|t| t.size_bytes as f64 / (1024.0 * 1024.0)).sum();
    let active_mb = by_placement.get(&DevicePlacement::FastGpu).unwrap_or(&0.0)
        + by_placement.get(&DevicePlacement::Replicated).unwrap_or(&0.0);
    let expert_mb = by_placement.get(&DevicePlacement::BigMemory).unwrap_or(&0.0);

    println!("Model:    {} ({} layers, {} experts)", model_name, n_layers, n_experts);
    println!("Size:     {:.1} GB total, {:.0} MB active, {:.0} MB experts ({:.0}%)",
        total_mb / 1024.0, active_mb, expert_mb, expert_mb / total_mb * 100.0);

    // Step 2: Detect hardware
    let total_ram_mb = {
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("sysctl").arg("-n").arg("hw.memsize").output();
            match output {
                Ok(o) => {
                    let bytes: u64 = String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0);
                    (bytes / 1024 / 1024) as f64
                }
                Err(_) => 24576.0, // fallback 24GB
            }
        }
        #[cfg(not(target_os = "macos"))]
        { 24576.0 }
    };
    println!("RAM:      {:.0} MB", total_ram_mb);

    // Step 3: Select strategy
    let selected_strategy = if strategy != "auto" {
        strategy.to_string()
    } else if n_experts <= 1 {
        println!("Strategy: full-ram (not a MoE model)");
        "full-ram".to_string()
    } else if total_mb < total_ram_mb * 0.8 {
        println!("Strategy: full-ram (model fits in memory)");
        "full-ram".to_string()
    } else {
        println!("Strategy: flash-ssd (model {:.1} GB > {:.0}% of RAM {:.1} GB)",
            total_mb / 1024.0, 80.0, total_ram_mb / 1024.0);
        "flash-ssd".to_string()
    };

    // Step 4: Find llama-server binary
    let server_bin = if let Some(path) = llama_server_path {
        path.to_string()
    } else {
        // Search common locations
        let candidates = [
            "llama.cpp/build/bin/llama-server",
            "../llama.cpp/build/bin/llama-server",
            "/usr/local/bin/llama-server",
        ];
        let mut found = None;
        for c in &candidates {
            let p = std::path::Path::new(c);
            if p.exists() {
                found = Some(c.to_string());
                break;
            }
        }
        // Also check relative to binary location
        if found.is_none() {
            if let Ok(exe) = std::env::current_exe() {
                let dir = exe.parent().unwrap_or(std::path::Path::new("."));
                let candidate = dir.join("../../llama.cpp/build/bin/llama-server");
                if candidate.exists() {
                    found = Some(candidate.to_string_lossy().to_string());
                }
            }
        }
        match found {
            Some(f) => f,
            None => {
                anyhow::bail!("Could not find llama-server binary. Build it with:\n  cd llama.cpp && cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON && cmake --build build --target llama-server\nOr specify with --llama-server <path>");
            }
        }
    };
    println!("Backend:  {}", server_bin);

    // Step 5: Prepare based on strategy
    let mut env_vars: Vec<(String, String)> = Vec::new();
    let mut extra_args: Vec<String> = Vec::new();

    match selected_strategy.as_str() {
        "flash-ssd" => {
            // Check if experts are already repacked
            let model_dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
            let model_stem = std::path::Path::new(model_path)
                .file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            let experts_dir = model_dir.join(format!("{}.experts", model_stem));

            if !experts_dir.exists() || !experts_dir.join("manifest.json").exists() {
                println!("\nExperts not repacked yet. Running repack...");
                repack_experts(model_path, Some(experts_dir.to_str().unwrap()))?;
            }

            // Determine experts to pin
            let mut pin_list: Vec<usize> = Vec::new();
            if pin_top > 0 && n_experts > 0 {
                // Run router analysis to find hot experts
                let mut expert_norms: Vec<(usize, f64)> = Vec::new();
                let emb_dim = gguf.embedding_dim().unwrap_or(0) as usize;
                for e in 0..n_experts {
                    let mut total_norm = 0.0f64;
                    let mut layers_counted = 0;
                    for layer in 0..n_layers {
                        let gate_name = format!("blk.{}.ffn_gate_inp.weight", layer);
                        if let Some(tensor) = gguf.find_tensor(&gate_name) {
                            if tensor.dtype == llamactl_engine::gguf::GgmlType::F32 {
                                let data = gguf.tensor_data(tensor);
                                let floats: &[f32] = unsafe {
                                    std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                                };
                                let row_start = e * emb_dim;
                                let row_end = row_start + emb_dim;
                                if row_end <= floats.len() {
                                    let norm: f64 = floats[row_start..row_end]
                                        .iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                                    total_norm += norm;
                                    layers_counted += 1;
                                }
                            }
                        }
                    }
                    if layers_counted > 0 {
                        expert_norms.push((e, total_norm / layers_counted as f64));
                    }
                }
                expert_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                pin_list = expert_norms.iter().take(pin_top).map(|(idx, _)| *idx).collect();
            }

            let pin_str = pin_list.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
            let experts_path = experts_dir.to_string_lossy().to_string();

            // Estimate RAM usage
            let per_expert_mb = expert_mb / (n_layers * n_experts) as f64;
            let pinned_mb = pin_list.len() as f64 * n_layers as f64 * per_expert_mb;
            let shared_mb = expert_mb / n_experts as f64; // rough estimate for shared experts
            let total_used = active_mb + pinned_mb + shared_mb + 500.0; // 500MB for compute buffers

            println!("\nFlash-SSD configuration:");
            println!("  Experts dir:  {}", experts_path);
            println!("  Active params on Metal: {:.0} MB", active_mb);
            if !pin_list.is_empty() {
                println!("  Pinned experts ({}): {:?}", pin_list.len(), &pin_list[..std::cmp::min(10, pin_list.len())]);
                println!("  Pinned memory: {:.0} MB", pinned_mb);
            }
            println!("  Shared cache: ~{:.0} MB", shared_mb);
            println!("  Est. total RAM: {:.0} MB / {:.0} MB ({:.0}%)",
                total_used, total_ram_mb, total_used / total_ram_mb * 100.0);

            if total_used > total_ram_mb * 0.9 {
                println!("  WARNING: May exceed available RAM. Reduce --pin-top or --ctx.");
            }

            env_vars.push(("LLAMA_FLASH_EXPERT".into(), experts_path));
            if !pin_str.is_empty() {
                env_vars.push(("LLAMA_PIN_EXPERTS".into(), pin_str));
            }
            extra_args.push("--no-mmap".into());
            extra_args.push("--no-warmup".into());
        }
        "full-ram" => {
            // Nothing special — just run llama-server normally
        }
        other => {
            anyhow::bail!("Unknown strategy: {}. Use 'auto', 'full-ram', or 'flash-ssd'.", other);
        }
    }

    // Step 6: Launch llama-server
    println!("\nStarting llama-server on port {}...", port);

    let mut cmd = Command::new(&server_bin);
    cmd.arg("--model").arg(model_path)
       .arg("-ngl").arg("999")
       .arg("--port").arg(port.to_string())
       .arg("-c").arg(ctx.to_string());

    for arg in &extra_args {
        cmd.arg(arg);
    }
    for (k, v) in &env_vars {
        cmd.env(k, v);
    }

    println!("Command:  {} {}", server_bin, cmd.get_args()
        .map(|a| a.to_string_lossy().to_string())
        .collect::<Vec<_>>().join(" "));
    for (k, v) in &env_vars {
        println!("  {}={}", k, v);
    }

    let mut child = cmd.spawn().context("Failed to start llama-server")?;

    // Wait for health check
    println!("\nWaiting for server to be ready...");
    let client = reqwest::Client::new();
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let mut ready = false;
    for i in 0..300 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        if let Ok(resp) = client.get(&health_url).send().await {
            if let Ok(text) = resp.text().await {
                if text.contains("ok") {
                    println!("Server ready! ({}s)\n", i + 1);
                    ready = true;
                    break;
                }
            }
        }
        if i % 15 == 14 {
            println!("  Still loading... ({}s)", i + 1);
        }
    }

    if !ready {
        println!("Server failed to start within 300 seconds.");
        child.kill().ok();
        anyhow::bail!("Server startup timeout");
    }

    println!("OpenAI-compatible API running at http://127.0.0.1:{}/v1", port);
    println!("  Chat: curl http://127.0.0.1:{}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'", port);
    println!("\nPress Ctrl+C to stop.");

    // Wait for the child process
    let status = child.wait().context("llama-server process failed")?;
    if !status.success() {
        println!("llama-server exited with: {}", status);
    }

    Ok(())
}

fn repack_experts(model_path: &str, output_dir: Option<&str>) -> Result<()> {
    use llamactl_engine::GgufFile;
    use std::io::Write;
    use std::path::PathBuf;

    println!("=== llamactl repack — GGUF expert repacker ===\n");

    let gguf = GgufFile::open(model_path)?;

    let n_layers = gguf.n_layers().context("Model has no layer count")? as usize;
    let n_experts = gguf.n_experts().context("Model has no expert count (not a MoE model?)")? as usize;
    let n_experts_used = gguf.n_experts_used().unwrap_or(8) as usize;
    let arch = gguf.architecture().unwrap_or("unknown").to_string();
    let model_name = gguf.model_name().unwrap_or("unknown").to_string();

    println!("Model:    {} ({})", model_name, arch);
    println!("Layers:   {}", n_layers);
    println!("Experts:  {} ({} active/token)\n", n_experts, n_experts_used);

    // Determine output directory
    let out_dir = match output_dir {
        Some(d) => PathBuf::from(d),
        None => {
            let model_dir = std::path::Path::new(model_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            let model_stem = std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            model_dir.join(format!("{}.experts", model_stem))
        }
    };
    std::fs::create_dir_all(&out_dir)?;
    println!("Output:   {}\n", out_dir.display());

    // Find expert tensor patterns per layer
    // Expert tensors follow: blk.{layer}.ffn_{gate,up,down}_exps.weight
    // or fused: blk.{layer}.ffn_gate_up_exps.weight
    // Also shared: blk.{layer}.ffn_{gate,up,down}_shexp.weight
    let expert_patterns = &[
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
    ];
    let shared_patterns = &[
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp",
    ];

    // Discover which expert tensors exist in layer 0 to determine the layout
    let mut expert_tensor_names: Vec<String> = Vec::new();
    let mut shared_tensor_names: Vec<String> = Vec::new();

    for pattern in expert_patterns {
        let name = format!("blk.0.{}.weight", pattern);
        if gguf.find_tensor(&name).is_some() {
            expert_tensor_names.push(pattern.to_string());
        }
    }
    for pattern in shared_patterns {
        let name = format!("blk.0.{}.weight", pattern);
        if gguf.find_tensor(&name).is_some() {
            shared_tensor_names.push(pattern.to_string());
        }
    }

    println!("Expert tensors per layer: {:?}", expert_tensor_names);
    println!("Shared tensors per layer: {:?}", shared_tensor_names);

    // Calculate per-expert size from the stacked tensors
    // Each expert tensor has shape [dim1, dim2, n_expert] — expert i is at offset i * (total / n_expert)
    let mut per_expert_bytes: u64 = 0;
    let mut tensor_layout: Vec<(String, u64)> = Vec::new(); // (name, per_expert_size)

    for pattern in &expert_tensor_names {
        let name = format!("blk.0.{}.weight", pattern);
        let tensor = gguf.find_tensor(&name).unwrap();
        // Use actual mmap'd data length, not calculated size_bytes (which can be imprecise for block quants)
        let actual_bytes = gguf.tensor_data(tensor).len() as u64;
        let expert_slice_bytes = actual_bytes / n_experts as u64;
        per_expert_bytes += expert_slice_bytes;
        tensor_layout.push((pattern.clone(), expert_slice_bytes));
        println!("  {}: {:?} {:?} → {} bytes/expert (tensor total: {} bytes)",
            pattern, tensor.shape, tensor.dtype, expert_slice_bytes, actual_bytes);
    }

    // Shared expert sizes (not per-expert, just one per layer)
    let mut shared_bytes: u64 = 0;
    let mut shared_layout: Vec<(String, u64)> = Vec::new();
    for pattern in &shared_tensor_names {
        let name = format!("blk.0.{}.weight", pattern);
        let tensor = gguf.find_tensor(&name).unwrap();
        let actual_bytes = gguf.tensor_data(tensor).len() as u64;
        shared_bytes += actual_bytes;
        shared_layout.push((pattern.clone(), actual_bytes));
    }

    println!("\n  Per-expert size: {} bytes ({:.2} MB)", per_expert_bytes, per_expert_bytes as f64 / 1024.0 / 1024.0);
    println!("  Shared expert size: {} bytes ({:.2} MB)", shared_bytes, shared_bytes as f64 / 1024.0 / 1024.0);
    let per_layer_bytes = n_experts as u64 * per_expert_bytes + shared_bytes;
    let total_bytes = n_layers as u64 * per_layer_bytes;
    println!("  Per-layer file size: {:.2} GB", per_layer_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  Total repacked size: {:.2} GB\n", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);

    // Write manifest
    let manifest = serde_json::json!({
        "model": model_name,
        "architecture": arch,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_experts_used": n_experts_used,
        "per_expert_bytes": per_expert_bytes,
        "shared_bytes": shared_bytes,
        "expert_layout": tensor_layout.iter().map(|(name, size)| {
            serde_json::json!({"tensor": name, "bytes": size})
        }).collect::<Vec<_>>(),
        "shared_layout": shared_layout.iter().map(|(name, size)| {
            serde_json::json!({"tensor": name, "bytes": size})
        }).collect::<Vec<_>>(),
    });
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    println!("Wrote {}", manifest_path.display());

    // Repack each layer
    // NOTE: Q3_K_M uses mixed quantization — different layers may use different quant types
    // for the same tensor name. We compute per-expert sizes per layer, not just from layer 0.
    let repack_start = std::time::Instant::now();

    for layer in 0..n_layers {
        let layer_start = std::time::Instant::now();
        let layer_path = out_dir.join(format!("layer_{:03}.bin", layer));

        let mut layer_file = std::fs::File::create(&layer_path)?;

        // Compute per-expert slice sizes for THIS layer's tensors
        let mut layer_expert_layout: Vec<(&str, u64)> = Vec::new();
        for pattern in &expert_tensor_names {
            let tensor_name = format!("blk.{}.{}.weight", layer, pattern);
            let tensor = gguf.find_tensor(&tensor_name)
                .with_context(|| format!("Missing tensor: {}", tensor_name))?;
            let actual_bytes = gguf.tensor_data(tensor).len() as u64;
            let expert_slice = actual_bytes / n_experts as u64;
            layer_expert_layout.push((pattern, expert_slice));
        }

        // Write expert data: expert 0 components, expert 1 components, ..., expert N components
        for expert_idx in 0..n_experts {
            for &(pattern, expert_slice_bytes) in &layer_expert_layout {
                let tensor_name = format!("blk.{}.{}.weight", layer, pattern);
                let tensor = gguf.find_tensor(&tensor_name).unwrap();
                let all_data = gguf.tensor_data(tensor);
                let start = expert_idx as u64 * expert_slice_bytes;
                let end = start + expert_slice_bytes;
                layer_file.write_all(&all_data[start as usize..end as usize])?;
            }
        }

        // Write shared expert data at the end of the file
        for (pattern, _) in &shared_layout {
            let tensor_name = format!("blk.{}.{}.weight", layer, pattern);
            let tensor = gguf.find_tensor(&tensor_name)
                .with_context(|| format!("Missing tensor: {}", tensor_name))?;
            let data = gguf.tensor_data(tensor);
            layer_file.write_all(data)?;
        }

        // Also compute shared expert sizes for this layer
        let mut layer_shared_layout: Vec<(&str, u64)> = Vec::new();
        for (pattern, _) in &shared_layout {
            let tensor_name = format!("blk.{}.{}.weight", layer, pattern);
            let tensor = gguf.find_tensor(&tensor_name)
                .with_context(|| format!("Missing tensor: {}", tensor_name))?;
            layer_shared_layout.push((pattern, gguf.tensor_data(tensor).len() as u64));
        }

        // Write per-layer metadata
        let layer_meta = serde_json::json!({
            "layer": layer,
            "per_expert_bytes": layer_expert_layout.iter().map(|(_, s)| s).sum::<u64>(),
            "expert_tensors": layer_expert_layout.iter().map(|(name, size)| {
                serde_json::json!({"tensor": name, "bytes": size})
            }).collect::<Vec<_>>(),
            "shared_tensors": layer_shared_layout.iter().map(|(name, size)| {
                serde_json::json!({"tensor": name, "bytes": size})
            }).collect::<Vec<_>>(),
            "shared_bytes": layer_shared_layout.iter().map(|(_, s)| s).sum::<u64>(),
        });
        let meta_path = out_dir.join(format!("layer_{:03}.json", layer));
        std::fs::write(&meta_path, serde_json::to_string(&layer_meta)?)?;

        layer_file.flush()?;
        let elapsed = layer_start.elapsed();
        let file_size = std::fs::metadata(&layer_path)?.len();
        let throughput = file_size as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;

        let total_elapsed = repack_start.elapsed();
        let eta = if layer > 0 {
            total_elapsed.as_secs_f64() / (layer as f64) * ((n_layers - layer - 1) as f64)
        } else { 0.0 };

        print!("\r  Layer {:>3}/{}: {:.2} GB in {:.1}s ({:.1} GB/s) | ETA: {:.0}s    ",
            layer + 1, n_layers,
            file_size as f64 / 1024.0 / 1024.0 / 1024.0,
            elapsed.as_secs_f64(),
            throughput,
            eta);
        std::io::stdout().flush()?;
    }

    let total_elapsed = repack_start.elapsed();
    let total_written: u64 = (0..n_layers).map(|l| {
        std::fs::metadata(out_dir.join(format!("layer_{:03}.bin", l)))
            .map(|m| m.len()).unwrap_or(0)
    }).sum();

    println!("\n\nDone! Repacked {} layers in {:.1}s", n_layers, total_elapsed.as_secs_f64());
    println!("  Total: {:.2} GB at {:.1} GB/s",
        total_written as f64 / 1024.0 / 1024.0 / 1024.0,
        total_written as f64 / total_elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0);
    println!("  Output: {}", out_dir.display());
    println!("\n  To use with flash-expert mode:");
    println!("  llamactl serve --model {} --flash-expert {}", model_path, out_dir.display());

    Ok(())
}

async fn bench_inference(model_path: &str, expert_addr: Option<&str>, n_tokens: usize, force_cpu: bool) -> Result<()> {
    use llamactl_engine::{LoadedModel, inference};
    use std::time::Instant;

    println!("=== llamactl inference benchmark ===\n");
    println!("Model: {}", model_path);

    let load_start = Instant::now();
    let model = if expert_addr.is_some() && !force_cpu {
        // Split mode: only load FastGpu + Replicated onto Metal (~1.3 GB)
        // Expert weights stay on the remote worker
        println!("Split mode: loading only local tensors (FastGpu + Replicated) onto Metal GPU\n");
        LoadedModel::load_split_local(model_path)?
    } else {
        // Local-only: full model on CPU (too large for Metal)
        if !force_cpu {
            println!("Note: using CPU for local-only (full model too large for Metal)");
            println!("  For GPU benchmark, use --expert-addr to split experts to another device\n");
        }
        LoadedModel::load_with_device(model_path, &candle_core::Device::Cpu)?
    };
    let load_time = load_start.elapsed();
    println!("Device: {:?}", model.device);
    println!("Loaded in {:.2}s", load_time.as_secs_f64());
    println!("{}", model.memory_summary());
    println!("Config: {} layers, {} experts ({} active), emb={}\n",
        model.n_layers, model.n_experts, model.n_experts_used, model.embedding_dim);

    // Dummy token IDs for benchmark (just use token 1 repeated)
    let token_ids: Vec<u32> = vec![1];

    // ── Test 1: Local-only baseline (skip if split mode — no expert weights loaded)
    if expert_addr.is_none() {
    println!("--- Test 1: Local-only forward pass (all on this device) ---\n");
    {
        let mut kv_caches: Vec<inference::KvCache> = (0..model.n_layers)
            .map(|_| inference::KvCache::new())
            .collect();

        // Warm up
        println!("Warming up (1 token)...");
        let _ = inference::forward(&token_ids, &model, &mut kv_caches);

        // Reset caches
        kv_caches = (0..model.n_layers)
            .map(|_| inference::KvCache::new())
            .collect();

        // Benchmark
        println!("Running {} token(s)...", n_tokens);
        let mut total_time = std::time::Duration::ZERO;

        for i in 0..n_tokens {
            let start = Instant::now();
            let logits = inference::forward(&token_ids, &model, &mut kv_caches)?;
            let elapsed = start.elapsed();
            total_time += elapsed;

            let logits_shape = logits.shape().clone();
            println!("  Token {}: {:.1}ms (logits shape: {:?})", i + 1, elapsed.as_millis(), logits_shape);
        }

        let avg_ms = total_time.as_millis() as f64 / n_tokens as f64;
        let tok_per_sec = 1000.0 / avg_ms;
        println!("\nLocal-only: {:.1} ms/token, {:.2} tok/s\n", avg_ms, tok_per_sec);
    }
    } // end if expert_addr.is_none()

    // ── Test 2: Split forward pass (if expert worker address provided) ──
    if let Some(addr) = expert_addr {
        println!("--- Test 2: Split forward pass (experts on remote worker) ---\n");
        println!("Expert worker: {}", addr);

        let mut remote = llamactl_engine::expert_client::RemoteExpertClient::connect(addr).await
            .context("Failed to connect to expert worker")?;
        println!("Connected to expert worker\n");

        let mut kv_caches: Vec<inference::KvCache> = (0..model.n_layers)
            .map(|_| inference::KvCache::new())
            .collect();

        // Warm up
        println!("Warming up (1 token)...");
        let _ = inference::forward_split(&token_ids, &model, &mut kv_caches, &mut remote).await;
        remote.clear_timings();

        // Reset
        kv_caches = (0..model.n_layers)
            .map(|_| inference::KvCache::new())
            .collect();

        // Benchmark
        println!("Running {} token(s)...", n_tokens);
        let mut total_time = std::time::Duration::ZERO;

        for i in 0..n_tokens {
            remote.clear_timings();
            let start = Instant::now();
            let logits = inference::forward_split(&token_ids, &model, &mut kv_caches, &mut remote).await?;
            let elapsed = start.elapsed();
            total_time += elapsed;

            let logits_shape = logits.shape().clone();
            println!("  Token {}: {:.1}ms (logits: {:?})", i + 1, elapsed.as_millis(), logits_shape);
            println!("    Remote hops: {}", remote.timing_summary());
        }

        let avg_ms = total_time.as_millis() as f64 / n_tokens as f64;
        let tok_per_sec = 1000.0 / avg_ms;
        println!("\nSplit: {:.1} ms/token, {:.2} tok/s\n", avg_ms, tok_per_sec);

        // Summary
        println!("=== Comparison ===");
        println!("  This measures operation-level split overhead.");
        println!("  For fair comparison with llama.cpp RPC, both should use GPU.");
        println!("  llamactl split sends ~8KB/layer vs llama.cpp RPC ~MB/layer.");
    } else {
        println!("--- Skipping split test (no --expert-addr provided) ---");
        println!("  To test split: run expert-worker on another machine, then:");
        println!("  llamactl bench --model <path> --expert-addr http://<ip>:50100");
    }

    Ok(())
}

async fn get_nodes(server: &str) -> Result<()> {
    let nodes: Vec<NodeInfo> = reqwest::get(format!("{server}/api/nodes"))
        .await
        .context("Failed to connect to server")?
        .error_for_status()?
        .json()
        .await?;

    if nodes.is_empty() {
        println!("No nodes registered.");
        return Ok(());
    }

    println!(
        "{:<36}  {:<18}  {:<8}  {:<14}  {:<12}  {:<14}  {:<6}  {:<6}",
        "ID", "HOSTNAME", "STATUS", "BACKEND", "VRAM", "RAM", "CORES", "CPU%"
    );

    for node in &nodes {
        let vram = match (node.vram_used_mb, node.vram_mb) {
            (Some(used), Some(total)) => format!("{}/{} MB", used, total),
            (None, Some(total)) => format!("{} MB", total),
            _ => "-".to_string(),
        };

        let hostname = node.hostname
            .strip_suffix(".local").or_else(|| node.hostname.strip_suffix(".lan"))
            .unwrap_or(&node.hostname);

        println!(
            "{:<36}  {:<18}  {:<8}  {:<14}  {:<12}  {}/{} MB  {:<6}  {:.1}",
            node.id, hostname, node.status, node.backend,
            vram, node.ram_used_mb, node.total_ram_mb,
            node.cpu_cores, node.cpu_usage_pct,
        );
    }

    println!("\n{} node(s)", nodes.len());
    Ok(())
}

async fn get_deployments(server: &str) -> Result<()> {
    let deployments: Vec<DeploymentInfo> = reqwest::get(format!("{server}/api/deployments"))
        .await
        .context("Failed to connect to server")?
        .error_for_status()?
        .json()
        .await?;

    if deployments.is_empty() {
        println!("No deployments.");
        return Ok(());
    }

    println!(
        "{:<36}  {:<30}  {:<18}  {:<10}  {:<6}  {}",
        "ID", "MODEL", "NODE", "STATUS", "PORT", "ENDPOINT"
    );

    for d in &deployments {
        let model_name = d.model.rsplit('/').next().unwrap_or(&d.model);
        let model_short = if model_name.len() > 28 {
            format!("{}...", &model_name[..25])
        } else {
            model_name.to_string()
        };

        let hostname = d.node_hostname
            .strip_suffix(".local").or_else(|| d.node_hostname.strip_suffix(".lan"))
            .unwrap_or(&d.node_hostname);

        println!(
            "{:<36}  {:<30}  {:<18}  {:<10}  {:<6}  {}",
            d.id, model_short, hostname, d.status, d.port,
            d.endpoint.as_deref().unwrap_or("-"),
        );
    }

    println!("\n{} deployment(s)", deployments.len());
    Ok(())
}

async fn deploy(
    server: &str,
    client: &reqwest::Client,
    model: String,
    node: Option<uuid::Uuid>,
    port: u16,
    context_length: u32,
    gpu_layers: u32,
    split: Vec<uuid::Uuid>,
    backend: &str,
) -> Result<()> {
    let node_id = if !split.is_empty() {
        Some(split[0])
    } else {
        node
    };

    let split_backend = if split.len() > 1 {
        match backend {
            "mlx" => Some(SplitBackend::MlxRing),
            _ => Some(SplitBackend::LlamaCppRpc),
        }
    } else {
        None
    };

    let req = DeployRequest {
        model,
        node_id,
        port: Some(port),
        context_length: Some(context_length),
        gpu_layers: Some(gpu_layers),
        split_nodes: split.clone(),
        split_backend,
    };

    let resp: DeployResponse = client
        .post(format!("{server}/api/deployments"))
        .json(&req)
        .send()
        .await
        .context("Failed to connect to server")?
        .error_for_status()
        .context("Failed to create deployment")?
        .json()
        .await?;

    println!("Deployment created:");
    println!("  ID:     {}", &resp.deployment_id.to_string()[..8]);
    println!("  Node:   {} ({})", resp.node_hostname, &resp.node_id.to_string()[..8]);
    println!("  Status: {}", resp.status);
    if split.len() > 1 {
        let backend_name = match split_backend {
            Some(SplitBackend::MlxRing) => "MLX Distributed ring",
            _ => "llama.cpp RPC",
        };
        println!("  Mode:   split across {} nodes ({})", split.len(), backend_name);
        println!("          main: {}", &split[0].to_string()[..8]);
        for (i, worker) in split.iter().skip(1).enumerate() {
            println!("          worker-{}: {}", i + 1, &worker.to_string()[..8]);
        }
        if split_backend == Some(SplitBackend::MlxRing) {
            println!("\n  Requires: pip install mlx-lm, passwordless SSH between nodes");
        }
    }
    println!("\nAgents will pick this up within ~10 seconds.");
    println!("Monitor: llamactl top");

    Ok(())
}
