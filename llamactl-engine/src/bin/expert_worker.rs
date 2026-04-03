//! Expert worker — runs on the big memory device.
//! Loads expert weights from GGUF and serves gRPC requests.
//!
//! Usage: expert-worker --model /path/to/model.gguf --port 50100

use std::sync::Arc;
use anyhow::Result;
use tonic::transport::Server;
use llamactl_engine::expert_server::ExpertWorker;
use llamactl_engine::rpc_proto::expert_compute_server::ExpertComputeServer;
use llamactl_engine::LoadedModel;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: expert-worker --model <path> [--port <port>]");
    let port: u16 = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(50100);

    tracing::info!(model = %model_path, port, "Loading model for expert worker...");
    // Expert worker always uses CPU — it's the big memory device
    let model = Arc::new(LoadedModel::load_with_device(model_path, &candle_core::Device::Cpu)?);
    tracing::info!("{}", model.memory_summary());

    let addr = format!("0.0.0.0:{}", port).parse()?;
    let worker = ExpertWorker::new(model);

    tracing::info!(%addr, "Expert worker listening");
    Server::builder()
        .add_service(ExpertComputeServer::new(worker))
        .serve(addr)
        .await?;

    Ok(())
}
