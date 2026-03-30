mod db;
mod routes;

use axum::{routing::{get, post}, Router};
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://llamactl.db".to_string());
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:6969".to_string());

    let pool = db::init_pool(&db_url).await?;

    let app = Router::new()
        .route("/api/nodes/register", post(routes::register_node))
        .route("/api/nodes/{id}/heartbeat", post(routes::heartbeat))
        .route("/api/nodes", get(routes::list_nodes))
        .layer(TraceLayer::new_for_http())
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("llamactl-server listening on {bind_addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
