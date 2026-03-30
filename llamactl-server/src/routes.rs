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

// ── Deployments ─────────────────────────────────────────────────

/// POST /api/deployments
pub async fn create_deployment(
    State(pool): State<SqlitePool>,
    Json(req): Json<DeployRequest>,
) -> Result<Json<DeployResponse>, (StatusCode, String)> {
    let port = req.port.unwrap_or(8080);
    let ctx_len = req.context_length.unwrap_or(2048);
    let gpu_layers = req.gpu_layers.unwrap_or(9999);

    // Pick target node: explicit or auto-schedule (most free memory, online only)
    let node = if let Some(nid) = req.node_id {
        sqlx::query_as::<_, NodeRow>("SELECT * FROM nodes WHERE id = ? AND status = 'online'")
            .bind(nid.to_string())
            .fetch_optional(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .ok_or((StatusCode::NOT_FOUND, "Node not found or offline".into()))?
    } else {
        // Auto-schedule: pick online node with most available RAM
        let now = Utc::now();
        let rows = sqlx::query_as::<_, NodeRow>("SELECT * FROM nodes")
            .fetch_all(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        rows.into_iter()
            .filter(|r| {
                let last_hb = chrono::DateTime::parse_from_rfc3339(&r.last_heartbeat)
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or(now);
                (now - last_hb).num_seconds() <= HEARTBEAT_TIMEOUT_SECS
            })
            .max_by_key(|r| {
                // Prefer nodes with VRAM, then sort by free memory
                let free_vram = r.vram_mb.unwrap_or(0) - r.vram_used_mb.unwrap_or(0);
                let free_ram = r.total_ram_mb - r.ram_used_mb;
                (free_vram, free_ram)
            })
            .ok_or((StatusCode::BAD_REQUEST, "No online nodes available".into()))?
    };

    let deployment_id = Uuid::new_v4();
    let now = Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO deployments (id, model, node_id, status, port, context_length, gpu_layers, created_at)
         VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)"
    )
    .bind(deployment_id.to_string())
    .bind(&req.model)
    .bind(&node.id)
    .bind(port as i32)
    .bind(ctx_len as i32)
    .bind(gpu_layers as i32)
    .bind(&now)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    tracing::info!(
        deployment_id = %deployment_id,
        model = %req.model,
        node = %node.hostname,
        "Deployment created"
    );

    Ok(Json(DeployResponse {
        deployment_id,
        node_id: Uuid::parse_str(&node.id).unwrap(),
        node_hostname: node.hostname,
        status: DeploymentStatus::Pending,
    }))
}

/// GET /api/deployments
pub async fn list_deployments(
    State(pool): State<SqlitePool>,
) -> Result<Json<Vec<DeploymentInfo>>, StatusCode> {
    let rows = sqlx::query_as::<_, DeploymentRow>(
        "SELECT d.*, n.hostname as node_hostname FROM deployments d JOIN nodes n ON d.node_id = n.id"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list deployments: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let deployments = rows.into_iter().map(|r| {
        let node_id = Uuid::parse_str(&r.node_id).unwrap();
        let endpoint = if r.status == "running" {
            Some(format!("http://{}:{}", r.node_hostname, r.port))
        } else {
            r.endpoint
        };

        DeploymentInfo {
            id: Uuid::parse_str(&r.id).unwrap(),
            model: r.model,
            node_id,
            node_hostname: r.node_hostname,
            status: serde_json::from_value(serde_json::Value::String(r.status)).unwrap(),
            port: r.port as u16,
            context_length: r.context_length as u32,
            gpu_layers: r.gpu_layers as u32,
            endpoint,
            error: r.error,
            created_at: chrono::DateTime::parse_from_rfc3339(&r.created_at)
                .unwrap()
                .with_timezone(&Utc),
        }
    }).collect();

    Ok(Json(deployments))
}

/// GET /api/nodes/:id/assignments — agent polls this for work
pub async fn get_assignments(
    State(pool): State<SqlitePool>,
    Path(node_id): Path<Uuid>,
) -> Result<Json<Vec<NodeAssignment>>, StatusCode> {
    let rows = sqlx::query_as::<_, DeploymentRow>(
        "SELECT d.*, n.hostname as node_hostname FROM deployments d
         JOIN nodes n ON d.node_id = n.id
         WHERE d.node_id = ? AND d.status IN ('pending', 'running', 'downloading')"
    )
    .bind(node_id.to_string())
    .fetch_all(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get assignments: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let assignments = rows.into_iter().map(|r| NodeAssignment {
        deployment_id: Uuid::parse_str(&r.id).unwrap(),
        model: r.model,
        port: r.port as u16,
        context_length: r.context_length as u32,
        gpu_layers: r.gpu_layers as u32,
    }).collect();

    Ok(Json(assignments))
}

/// PUT /api/deployments/:id/status — agent reports status
pub async fn update_deployment_status(
    State(pool): State<SqlitePool>,
    Path(deployment_id): Path<Uuid>,
    Json(req): Json<DeploymentStatusUpdate>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let status_str = serde_json::to_value(&req.status)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    let result = sqlx::query(
        "UPDATE deployments SET status = ?, endpoint = ?, error = ? WHERE id = ?"
    )
    .bind(&status_str)
    .bind(&req.endpoint)
    .bind(&req.error)
    .bind(deployment_id.to_string())
    .execute(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update deployment status: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    tracing::info!(
        deployment_id = %deployment_id,
        status = %req.status,
        "Deployment status updated"
    );

    Ok(Json(HeartbeatResponse { ok: true }))
}

/// DELETE /api/deployments/:id — stop a deployment
pub async fn delete_deployment(
    State(pool): State<SqlitePool>,
    Path(deployment_id): Path<Uuid>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let result = sqlx::query(
        "UPDATE deployments SET status = 'stopped' WHERE id = ? AND status != 'stopped'"
    )
    .bind(deployment_id.to_string())
    .execute(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to stop deployment: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    tracing::info!(deployment_id = %deployment_id, "Deployment stopped");
    Ok(Json(HeartbeatResponse { ok: true }))
}

// ── Internal row types ──────────────────────────────────────────

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

#[derive(sqlx::FromRow)]
struct DeploymentRow {
    id: String,
    model: String,
    node_id: String,
    status: String,
    port: i32,
    context_length: i32,
    gpu_layers: i32,
    endpoint: Option<String>,
    error: Option<String>,
    created_at: String,
    node_hostname: String,
}
