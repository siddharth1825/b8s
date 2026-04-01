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
/// If a node with the same hostname already exists, reuse it (re-registration).
pub async fn register_node(
    State(pool): State<SqlitePool>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, StatusCode> {
    let now = Utc::now().to_rfc3339();
    let backend_str = serde_json::to_value(&req.backend)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Check if node with same hostname already exists
    let existing = sqlx::query_as::<_, NodeRow>(
        "SELECT * FROM nodes WHERE hostname = ?"
    )
    .bind(&req.hostname)
    .fetch_optional(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to check existing node: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let node_id = if let Some(existing) = existing {
        // Re-register: update existing node, bring it back online
        let id = existing.id.clone();
        sqlx::query(
            "UPDATE nodes SET os = ?, arch = ?, backend = ?, status = 'online',
             vram_mb = ?, total_ram_mb = ?, cpu_cores = ?, last_heartbeat = ?
             WHERE id = ?"
        )
        .bind(&req.os)
        .bind(&req.arch)
        .bind(&backend_str)
        .bind(req.vram_mb.map(|v| v as i64))
        .bind(req.total_ram_mb as i64)
        .bind(req.cpu_cores as i32)
        .bind(&now)
        .bind(&id)
        .execute(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to update node: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let uuid = Uuid::parse_str(&id).unwrap();
        tracing::info!(
            node_id = %uuid,
            hostname = %req.hostname,
            backend = %req.backend,
            "Node re-registered (existing)"
        );
        uuid
    } else {
        // New node
        let id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO nodes (id, hostname, os, arch, backend, status, vram_mb, total_ram_mb, cpu_cores, registered_at, last_heartbeat)
             VALUES (?, ?, ?, ?, ?, 'online', ?, ?, ?, ?, ?)"
        )
        .bind(id.to_string())
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
            node_id = %id,
            hostname = %req.hostname,
            backend = %req.backend,
            "Node registered (new)"
        );
        id
    };

    Ok(Json(RegisterResponse {
        node_id,
        heartbeat_interval_secs: HEARTBEAT_INTERVAL_SECS,
    }))
}

/// DELETE /api/nodes/:id — remove a stale node
pub async fn delete_node(
    State(pool): State<SqlitePool>,
    Path(node_id): Path<Uuid>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let id_str = node_id.to_string();

    // Stop any active deployments on this node
    sqlx::query("UPDATE deployments SET status = 'stopped' WHERE node_id = ? AND status IN ('pending', 'running', 'downloading')")
        .bind(&id_str)
        .execute(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to stop deployments: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Clean up cached models
    let _ = sqlx::query("DELETE FROM cached_models WHERE node_id = ?")
        .bind(&id_str)
        .execute(&pool)
        .await;

    // Clean up deployment workers
    let _ = sqlx::query("DELETE FROM deployment_workers WHERE node_id = ?")
        .bind(&id_str)
        .execute(&pool)
        .await;

    // Delete the node
    let result = sqlx::query("DELETE FROM nodes WHERE id = ?")
        .bind(&id_str)
        .execute(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to delete node: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    tracing::info!(node_id = %node_id, "Node removed");
    Ok(Json(HeartbeatResponse { ok: true }))
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

    // Update cached models
    if !req.cached_models.is_empty() {
        let node_str = node_id.to_string();
        // Replace all: delete old, insert new
        let _ = sqlx::query("DELETE FROM cached_models WHERE node_id = ?")
            .bind(&node_str)
            .execute(&pool)
            .await;
        for model in &req.cached_models {
            let _ = sqlx::query(
                "INSERT OR IGNORE INTO cached_models (node_id, model) VALUES (?, ?)"
            )
            .bind(&node_str)
            .bind(model)
            .execute(&pool)
            .await;
        }
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

    // Fetch all cached models in one query
    let cache_rows = sqlx::query_as::<_, CachedModelRow>("SELECT node_id, model FROM cached_models")
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

    let mut cache_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for cr in cache_rows {
        cache_map.entry(cr.node_id).or_default().push(cr.model);
    }

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

            let cached_models = cache_map.remove(&r.id).unwrap_or_default();

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
                cached_models,
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
        // Auto-schedule: prefer node with model cached, then most free memory
        let now = Utc::now();
        let rows = sqlx::query_as::<_, NodeRow>("SELECT * FROM nodes")
            .fetch_all(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Check which nodes have this model cached
        let cached_nodes = sqlx::query_as::<_, CachedModelRow>(
            "SELECT node_id, model FROM cached_models WHERE model = ?"
        )
        .bind(&req.model)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
        let cached_node_ids: std::collections::HashSet<String> =
            cached_nodes.into_iter().map(|r| r.node_id).collect();

        rows.into_iter()
            .filter(|r| {
                let last_hb = chrono::DateTime::parse_from_rfc3339(&r.last_heartbeat)
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or(now);
                (now - last_hb).num_seconds() <= HEARTBEAT_TIMEOUT_SECS
            })
            .max_by_key(|r| {
                // Priority: (1) model already cached, (2) free VRAM, (3) free RAM
                let has_cache = if cached_node_ids.contains(&r.id) { 1i64 } else { 0 };
                let free_vram = r.vram_mb.unwrap_or(0) - r.vram_used_mb.unwrap_or(0);
                let free_ram = r.total_ram_mb - r.ram_used_mb;
                (has_cache, free_vram, free_ram)
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

    // Handle split deployment: register workers
    if !req.split_nodes.is_empty() {
        // First node in split_nodes is main, rest are RPC workers
        let main_id = req.split_nodes[0].to_string();
        let rpc_port = 50052i32;

        // Register main node
        sqlx::query(
            "INSERT INTO deployment_workers (deployment_id, node_id, role, rpc_port) VALUES (?, ?, 'main', ?)"
        )
        .bind(deployment_id.to_string())
        .bind(&main_id)
        .bind(rpc_port)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Register RPC workers
        for (i, worker_id) in req.split_nodes.iter().skip(1).enumerate() {
            let worker_rpc_port = rpc_port + i as i32 + 1;
            sqlx::query(
                "INSERT INTO deployment_workers (deployment_id, node_id, role, rpc_port) VALUES (?, ?, 'rpc_worker', ?)"
            )
            .bind(deployment_id.to_string())
            .bind(worker_id.to_string())
            .bind(worker_rpc_port)
            .execute(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        }

        tracing::info!(
            deployment_id = %deployment_id,
            model = %req.model,
            nodes = req.split_nodes.len(),
            "Split deployment created"
        );
    } else {
        tracing::info!(
            deployment_id = %deployment_id,
            model = %req.model,
            node = %node.hostname,
            "Deployment created"
        );
    }

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

    // Fetch all deployment workers in one query
    let all_workers = sqlx::query_as::<_, DeploymentWorkerRow>(
        "SELECT dw.deployment_id, dw.node_id, dw.role, dw.rpc_port, n.hostname
         FROM deployment_workers dw JOIN nodes n ON dw.node_id = n.id"
    )
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

    let mut workers_map: std::collections::HashMap<String, Vec<WorkerInfo>> = std::collections::HashMap::new();
    for w in all_workers {
        let info = WorkerInfo {
            node_id: Uuid::parse_str(&w.node_id).unwrap(),
            hostname: w.hostname,
            role: serde_json::from_value(serde_json::Value::String(w.role)).unwrap_or(AssignmentRole::Main),
            rpc_port: w.rpc_port as u16,
            buffer_size_mb: None,
        };
        workers_map.entry(w.deployment_id).or_default().push(info);
    }

    let deployments = rows.into_iter().map(|r| {
        let node_id = Uuid::parse_str(&r.node_id).unwrap();
        let endpoint = if r.status == "running" {
            Some(format!("http://{}:{}", r.node_hostname, r.port))
        } else {
            r.endpoint
        };

        let logs: Vec<String> = serde_json::from_str(&r.logs).unwrap_or_default();
        let workers = workers_map.remove(&r.id).unwrap_or_default();

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
            progress: r.progress as f32,
            logs,
            workers,
            last_tok_per_sec: r.last_tok_per_sec.map(|v| v as f32),
        }
    }).collect();

    Ok(Json(deployments))
}

/// GET /api/nodes/:id/assignments — agent polls this for work
pub async fn get_assignments(
    State(pool): State<SqlitePool>,
    Path(node_id): Path<Uuid>,
) -> Result<Json<Vec<NodeAssignment>>, StatusCode> {
    let node_str = node_id.to_string();

    // Standard single-node deployments assigned to this node
    let rows = sqlx::query_as::<_, DeploymentRow>(
        "SELECT d.*, n.hostname as node_hostname FROM deployments d
         JOIN nodes n ON d.node_id = n.id
         WHERE d.node_id = ? AND d.status IN ('pending', 'running', 'downloading')"
    )
    .bind(&node_str)
    .fetch_all(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get assignments: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut assignments: Vec<NodeAssignment> = Vec::new();

    for r in &rows {
        // Check if this is a split deployment
        let workers = sqlx::query_as::<_, DeploymentWorkerRow>(
            "SELECT dw.deployment_id, dw.node_id, dw.role, dw.rpc_port, n.hostname
             FROM deployment_workers dw
             JOIN nodes n ON dw.node_id = n.id
             WHERE dw.deployment_id = ?"
        )
        .bind(&r.id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        if workers.is_empty() {
            // Simple single-node deployment
            assignments.push(NodeAssignment {
                deployment_id: Uuid::parse_str(&r.id).unwrap(),
                model: r.model.clone(),
                port: r.port as u16,
                context_length: r.context_length as u32,
                gpu_layers: r.gpu_layers as u32,
                role: AssignmentRole::Main,
                rpc_servers: vec![],
            });
        } else {
            // Split deployment — find this node's role
            if let Some(my_role) = workers.iter().find(|w| w.node_id == node_str) {
                let role: AssignmentRole = serde_json::from_value(
                    serde_json::Value::String(my_role.role.clone())
                ).unwrap_or(AssignmentRole::Main);

                // If main, gather RPC worker addresses
                let rpc_servers = if role == AssignmentRole::Main {
                    workers.iter()
                        .filter(|w| w.role == "rpc_worker")
                        .map(|w| format!("{}:{}", w.hostname, w.rpc_port))
                        .collect()
                } else {
                    vec![]
                };

                assignments.push(NodeAssignment {
                    deployment_id: Uuid::parse_str(&r.id).unwrap(),
                    model: r.model.clone(),
                    port: if role == AssignmentRole::Main {
                        r.port as u16
                    } else {
                        my_role.rpc_port as u16
                    },
                    context_length: r.context_length as u32,
                    gpu_layers: r.gpu_layers as u32,
                    role,
                    rpc_servers,
                });
            }
        }
    }

    // Also check for split deployments where this node is a worker but not the main node_id
    let worker_rows = sqlx::query_as::<_, DeploymentWorkerRow>(
        "SELECT dw.deployment_id, dw.node_id, dw.role, dw.rpc_port, n.hostname
         FROM deployment_workers dw
         JOIN nodes n ON dw.node_id = n.id
         WHERE dw.node_id = ? AND dw.role = 'rpc_worker'"
    )
    .bind(&node_str)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

    for wr in worker_rows {
        // Check if we already have this deployment
        let dep_id = Uuid::parse_str(&wr.deployment_id).unwrap();
        if assignments.iter().any(|a| a.deployment_id == dep_id) {
            continue;
        }

        // Fetch the deployment details
        if let Ok(Some(dep)) = sqlx::query_as::<_, DeploymentRow>(
            "SELECT d.*, n.hostname as node_hostname FROM deployments d
             JOIN nodes n ON d.node_id = n.id
             WHERE d.id = ? AND d.status IN ('pending', 'running', 'downloading')"
        )
        .bind(&wr.deployment_id)
        .fetch_optional(&pool)
        .await
        {
            assignments.push(NodeAssignment {
                deployment_id: dep_id,
                model: dep.model,
                port: wr.rpc_port as u16,
                context_length: dep.context_length as u32,
                gpu_layers: dep.gpu_layers as u32,
                role: AssignmentRole::RpcWorker,
                rpc_servers: vec![],
            });
        }
    }

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

    let logs_json = serde_json::to_string(&req.logs).unwrap_or_else(|_| "[]".into());

    let result = sqlx::query(
        "UPDATE deployments SET status = ?, endpoint = ?, error = ?, progress = ?, logs = ?, last_tok_per_sec = ? WHERE id = ?"
    )
    .bind(&status_str)
    .bind(&req.endpoint)
    .bind(&req.error)
    .bind(req.progress)
    .bind(&logs_json)
    .bind(req.last_tok_per_sec)
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

    Ok(Json(HeartbeatResponse { ok: true }))
}

/// DELETE /api/deployments/:id — fully remove a deployment from the DB
pub async fn delete_deployment(
    State(pool): State<SqlitePool>,
    Path(deployment_id): Path<Uuid>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let id_str = deployment_id.to_string();

    // First stop it if running (agent will pick up the removal)
    let _ = sqlx::query("UPDATE deployments SET status = 'stopped' WHERE id = ? AND status NOT IN ('stopped', 'failed')")
        .bind(&id_str)
        .execute(&pool)
        .await;

    // Clean up workers
    let _ = sqlx::query("DELETE FROM deployment_workers WHERE deployment_id = ?")
        .bind(&id_str)
        .execute(&pool)
        .await;

    // Delete the deployment
    let result = sqlx::query("DELETE FROM deployments WHERE id = ?")
        .bind(&id_str)
        .execute(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to delete deployment: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    tracing::info!(deployment_id = %deployment_id, "Deployment deleted");
    Ok(Json(HeartbeatResponse { ok: true }))
}

/// PUT /api/deployments/:id — pause, resume, or restart a deployment
pub async fn update_deployment_control(
    State(pool): State<SqlitePool>,
    Path(deployment_id): Path<Uuid>,
    Json(req): Json<DeploymentControlRequest>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    let id_str = deployment_id.to_string();

    let new_status = match req.action {
        DeploymentAction::Pause => "paused",
        DeploymentAction::Resume => "pending",  // agent will pick it up again
        DeploymentAction::Restart => {
            // Stop first, then set to pending
            let _ = sqlx::query("UPDATE deployments SET status = 'stopped' WHERE id = ?")
                .bind(&id_str)
                .execute(&pool)
                .await;
            "pending"
        }
    };

    let result = sqlx::query("UPDATE deployments SET status = ?, error = NULL WHERE id = ?")
        .bind(new_status)
        .bind(&id_str)
        .execute(&pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to update deployment: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    tracing::info!(deployment_id = %deployment_id, action = ?req.action, "Deployment control");
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
    progress: f64,
    logs: String,
    last_tok_per_sec: Option<f64>,
}

#[derive(sqlx::FromRow)]
struct CachedModelRow {
    node_id: String,
    model: String,
}

#[derive(sqlx::FromRow)]
struct DeploymentWorkerRow {
    deployment_id: String,
    node_id: String,
    role: String,
    rpc_port: i32,
    hostname: String,
}
