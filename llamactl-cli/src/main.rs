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
