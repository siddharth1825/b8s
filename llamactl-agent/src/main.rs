mod detect;
mod runner;

use anyhow::{Context, Result};
use llamactl_common::*;
use std::time::Duration;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let server_url = std::env::var("LLAMACTL_SERVER")
        .unwrap_or_else(|_| "http://localhost:6969".to_string());

    tracing::info!("Detecting hardware...");
    let hw = detect::detect_hardware();
    tracing::info!(
        hostname = %hw.hostname,
        backend = %hw.backend,
        vram_mb = ?hw.vram_mb,
        ram_mb = hw.total_ram_mb,
        cores = hw.cpu_cores,
        "Hardware detected"
    );

    let client = reqwest::Client::new();
    let (node_id, interval) = register(&client, &server_url, &hw).await?;

    tracing::info!(
        node_id = %node_id,
        interval_secs = interval,
        "Registered with server, starting heartbeat + deployment loop"
    );

    let mut runner = runner::Runner::new(client.clone(), server_url.clone(), hw.backend);

    run_loop(&client, &server_url, node_id, interval, &mut runner).await
}

async fn register(
    client: &reqwest::Client,
    server_url: &str,
    hw: &detect::HardwareInfo,
) -> Result<(Uuid, u64)> {
    let req = RegisterRequest {
        hostname: hw.hostname.clone(),
        os: hw.os.clone(),
        arch: hw.arch.clone(),
        backend: hw.backend,
        vram_mb: hw.vram_mb,
        total_ram_mb: hw.total_ram_mb,
        cpu_cores: hw.cpu_cores,
    };

    let resp: RegisterResponse = client
        .post(format!("{server_url}/api/nodes/register"))
        .json(&req)
        .send()
        .await
        .context("Failed to connect to server")?
        .error_for_status()
        .context("Server rejected registration")?
        .json()
        .await?;

    Ok((resp.node_id, resp.heartbeat_interval_secs))
}

async fn run_loop(
    client: &reqwest::Client,
    server_url: &str,
    node_id: Uuid,
    interval_secs: u64,
    runner: &mut runner::Runner,
) -> Result<()> {
    let heartbeat_url = format!("{server_url}/api/nodes/{node_id}/heartbeat");
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;

        // Send heartbeat
        let (ram_used_mb, cpu_usage_pct) = detect::collect_metrics();
        let req = HeartbeatRequest {
            vram_used_mb: None,
            ram_used_mb,
            cpu_usage_pct,
            gpu_temp_c: None,
            tokens_per_sec: None,
        };

        match client.post(&heartbeat_url).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!("Heartbeat sent");
            }
            Ok(resp) => {
                tracing::warn!(status = %resp.status(), "Heartbeat rejected");
            }
            Err(e) => {
                tracing::error!(err = %e, "Heartbeat failed — will retry");
            }
        }

        // Poll for deployment assignments and reconcile
        runner.poll(node_id).await;
    }
}
