use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llamactl_common::NodeInfo;

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
}

#[derive(Subcommand)]
enum GetResource {
    /// List all registered nodes
    Nodes,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Get { resource } => match resource {
            GetResource::Nodes => get_nodes(&cli.server).await?,
        },
    }

    Ok(())
}

async fn get_nodes(server: &str) -> Result<()> {
    let nodes: Vec<NodeInfo> = reqwest::get(format!("{server}/api/nodes"))
        .await
        .context("Failed to connect to server")?
        .error_for_status()
        .context("Server returned error")?
        .json()
        .await?;

    if nodes.is_empty() {
        println!("No nodes registered.");
        return Ok(());
    }

    // Print table header
    println!(
        "{:<36}  {:<15}  {:<8}  {:<14}  {:<10}  {:<10}  {:<6}  {:<8}",
        "ID", "HOSTNAME", "STATUS", "BACKEND", "VRAM", "RAM", "CORES", "CPU%"
    );

    for node in &nodes {
        let vram = match (node.vram_used_mb, node.vram_mb) {
            (Some(used), Some(total)) => format!("{}/{} MB", used, total),
            (None, Some(total)) => format!("-/{} MB", total),
            _ => "-".to_string(),
        };

        let ram = format!(
            "{}/{} MB",
            node.ram_used_mb,
            node.total_ram_mb
        );

        println!(
            "{:<36}  {:<15}  {:<8}  {:<14}  {:<10}  {:<10}  {:<6}  {:<8.1}",
            node.id,
            node.hostname,
            node.status,
            node.backend,
            vram,
            ram,
            node.cpu_cores,
            node.cpu_usage_pct,
        );
    }

    println!("\n{} node(s) total", nodes.len());
    Ok(())
}
