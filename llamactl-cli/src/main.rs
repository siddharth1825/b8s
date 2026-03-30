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
        /// Split model across multiple nodes (llama.cpp RPC tensor splitting)
        /// Comma-separated node IDs: first is main, rest are RPC workers
        #[arg(long, value_delimiter = ',')]
        split: Vec<uuid::Uuid>,
    },
    /// Stop a running deployment
    Stop {
        /// Deployment ID
        deployment_id: uuid::Uuid,
    },
    /// Live cluster dashboard
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

    match cli.command {
        Commands::Get { resource } => match resource {
            GetResource::Nodes => get_nodes(&cli.server).await?,
            GetResource::Deployments => get_deployments(&cli.server).await?,
        },
        Commands::Deploy {
            model, node, port, context_length, gpu_layers, split,
        } => deploy(&cli.server, model, node, port, context_length, gpu_layers, split).await?,
        Commands::Stop { deployment_id } => stop(&cli.server, deployment_id).await?,
        Commands::Top => top(&cli.server).await?,
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
        "{:<36}  {:<15}  {:<8}  {:<14}  {:<12}  {:<14}  {:<6}  {:<6}",
        "ID", "HOSTNAME", "STATUS", "BACKEND", "VRAM", "RAM", "CORES", "CPU%"
    );

    for node in &nodes {
        let vram = match (node.vram_used_mb, node.vram_mb) {
            (Some(used), Some(total)) => format!("{}/{} MB", used, total),
            (None, Some(total)) => format!("-/{} MB", total),
            _ => "-".to_string(),
        };

        println!(
            "{:<36}  {:<15}  {:<8}  {:<14}  {:<12}  {}/{} MB  {:<6}  {:.1}",
            node.id, node.hostname, node.status, node.backend,
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
        "{:<36}  {:<30}  {:<15}  {:<10}  {:<6}  {}",
        "ID", "MODEL", "NODE", "STATUS", "PORT", "ENDPOINT"
    );

    for d in &deployments {
        let model_short = if d.model.len() > 28 {
            format!("..{}", &d.model[d.model.len()-26..])
        } else {
            d.model.clone()
        };

        println!(
            "{:<36}  {:<30}  {:<15}  {:<10}  {:<6}  {}",
            d.id, model_short, d.node_hostname, d.status, d.port,
            d.endpoint.as_deref().unwrap_or("-"),
        );
    }

    println!("\n{} deployment(s)", deployments.len());
    Ok(())
}

async fn deploy(
    server: &str,
    model: String,
    node: Option<uuid::Uuid>,
    port: u16,
    context_length: u32,
    gpu_layers: u32,
    split: Vec<uuid::Uuid>,
) -> Result<()> {
    let node_id = if !split.is_empty() {
        Some(split[0]) // Main node is first in split list
    } else {
        node
    };

    let req = DeployRequest {
        model,
        node_id,
        port: Some(port),
        context_length: Some(context_length),
        gpu_layers: Some(gpu_layers),
        split_nodes: split.clone(),
    };

    let resp: DeployResponse = reqwest::Client::new()
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
    println!("  ID:     {}", resp.deployment_id);
    println!("  Node:   {} ({})", resp.node_hostname, resp.node_id);
    println!("  Status: {}", resp.status);
    if split.len() > 1 {
        println!("  Mode:   split across {} nodes (llama.cpp RPC)", split.len());
        println!("          main: {}", split[0]);
        for (i, worker) in split.iter().skip(1).enumerate() {
            println!("          rpc-worker-{}: {}", i + 1, worker);
        }
    }
    println!("\nAgents will pick this up within ~10 seconds.");
    println!("Check status: llamactl get deployments");

    Ok(())
}

async fn stop(server: &str, deployment_id: uuid::Uuid) -> Result<()> {
    reqwest::Client::new()
        .delete(format!("{server}/api/deployments/{deployment_id}"))
        .send()
        .await
        .context("Failed to connect to server")?
        .error_for_status()
        .context("Failed to stop deployment")?;

    println!("Deployment {} stopped.", deployment_id);
    Ok(())
}

async fn top(server: &str) -> Result<()> {
    tui::run(server).await
}
