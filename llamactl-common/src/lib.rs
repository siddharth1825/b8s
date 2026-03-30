use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Backend types ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    LlamaCppRpc,
    Mlx,
    Cpu,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LlamaCppRpc => write!(f, "llama.cpp-rpc"),
            Self::Mlx => write!(f, "mlx"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

// ── Node status ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Online,
    Offline,
    Draining,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
            Self::Offline => write!(f, "offline"),
            Self::Draining => write!(f, "draining"),
        }
    }
}

// ── Registration ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub backend: BackendKind,
    pub vram_mb: Option<u64>,
    pub total_ram_mb: u64,
    pub cpu_cores: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub node_id: Uuid,
    pub heartbeat_interval_secs: u64,
}

// ── Heartbeat ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub vram_used_mb: Option<u64>,
    pub ram_used_mb: u64,
    pub cpu_usage_pct: f32,
    pub gpu_temp_c: Option<f32>,
    pub tokens_per_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub ok: bool,
}

// ── Node info (returned by GET /api/nodes) ──────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: Uuid,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub backend: BackendKind,
    pub status: NodeStatus,
    pub vram_mb: Option<u64>,
    pub vram_used_mb: Option<u64>,
    pub total_ram_mb: u64,
    pub ram_used_mb: u64,
    pub cpu_cores: u32,
    pub cpu_usage_pct: f32,
    pub gpu_temp_c: Option<f32>,
    pub tokens_per_sec: Option<f32>,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
}
