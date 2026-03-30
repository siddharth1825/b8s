use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use llamactl_common::*;
use sqlx::SqlitePool;
use uuid::Uuid;

const HEARTBEAT_INTERVAL_SECS: u64 = 10;
const HEARTBEAT_TIMEOUT_SECS: i64 = 30;

/// POST /api/nodes/register
pub async fn register_node(
    State(pool): State<SqlitePool>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, StatusCode> {
    let node_id = Uuid::new_v4();
    let now = Utc::now().to_rfc3339();
    let backend_str = serde_json::to_value(&req.backend)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    sqlx::query(
        "INSERT INTO nodes (id, hostname, os, arch, backend, status, vram_mb, total_ram_mb, cpu_cores, registered_at, last_heartbeat)
         VALUES (?, ?, ?, ?, ?, 'online', ?, ?, ?, ?, ?)"
    )
    .bind(node_id.to_string())
    .bind(&req.hostname)
    .bind(&req.os)
    .bind(&req.arch)
    .bind(&backend_str)
    .bind(req.vram_mb.map(|v| v as i64))
    .bind(req.total_ram_mb as i64)
    .bind(req.cpu_cores as i32)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to register node: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!(
        node_id = %node_id,
        hostname = %req.hostname,
        backend = %req.backend,
        "Node registered"
    );

    Ok(Json(RegisterResponse {
        node_id,
        heartbeat_interval_secs: HEARTBEAT_INTERVAL_SECS,
    }))
}

/// POST /api/nodes/:id/heartbeat
pub async fn heartbeat(
    State(pool): State<SqlitePool>,
    Path(node_id): Path<Uuid>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let now = Utc::now().to_rfc3339();

    let result = sqlx::query(
        "UPDATE nodes SET
            status = 'online',
            vram_used_mb = ?,
            ram_used_mb = ?,
            cpu_usage_pct = ?,
            gpu_temp_c = ?,
            tokens_per_sec = ?,
            last_heartbeat = ?
         WHERE id = ?"
    )
    .bind(req.vram_used_mb.map(|v| v as i64))
    .bind(req.ram_used_mb as i64)
    .bind(req.cpu_usage_pct)
    .bind(req.gpu_temp_c)
    .bind(req.tokens_per_sec)
    .bind(&now)
    .bind(node_id.to_string())
    .execute(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update heartbeat: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    Ok(Json(HeartbeatResponse { ok: true }))
}

/// GET /api/nodes
pub async fn list_nodes(
    State(pool): State<SqlitePool>,
) -> Result<Json<Vec<NodeInfo>>, StatusCode> {
    let rows = sqlx::query_as::<_, NodeRow>("SELECT * FROM nodes")
        .fetch_all(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to list nodes: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let now = Utc::now();
    let nodes: Vec<NodeInfo> = rows
        .into_iter()
        .map(|r| {
            let last_hb = chrono::DateTime::parse_from_rfc3339(&r.last_heartbeat)
                .unwrap()
                .with_timezone(&Utc);
            let status = if (now - last_hb).num_seconds() > HEARTBEAT_TIMEOUT_SECS {
                NodeStatus::Offline
            } else {
                serde_json::from_value(serde_json::Value::String(r.status)).unwrap_or(NodeStatus::Offline)
            };

            NodeInfo {
                id: Uuid::parse_str(&r.id).unwrap(),
                hostname: r.hostname,
                os: r.os,
                arch: r.arch,
                backend: serde_json::from_value(serde_json::Value::String(r.backend)).unwrap(),
                status,
                vram_mb: r.vram_mb.map(|v| v as u64),
                vram_used_mb: r.vram_used_mb.map(|v| v as u64),
                total_ram_mb: r.total_ram_mb as u64,
                ram_used_mb: r.ram_used_mb as u64,
                cpu_cores: r.cpu_cores as u32,
                cpu_usage_pct: r.cpu_usage_pct as f32,
                gpu_temp_c: r.gpu_temp_c.map(|v| v as f32),
                tokens_per_sec: r.tokens_per_sec.map(|v| v as f32),
                registered_at: chrono::DateTime::parse_from_rfc3339(&r.registered_at)
                    .unwrap()
                    .with_timezone(&Utc),
                last_heartbeat: last_hb,
            }
        })
        .collect();

    Ok(Json(nodes))
}

// Internal row type matching SQLite schema
#[derive(sqlx::FromRow)]
struct NodeRow {
    id: String,
    hostname: String,
    os: String,
    arch: String,
    backend: String,
    status: String,
    vram_mb: Option<i64>,
    vram_used_mb: Option<i64>,
    total_ram_mb: i64,
    ram_used_mb: i64,
    cpu_cores: i32,
    cpu_usage_pct: f64,
    gpu_temp_c: Option<f64>,
    tokens_per_sec: Option<f64>,
    registered_at: String,
    last_heartbeat: String,
}
