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
