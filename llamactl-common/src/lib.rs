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
    /// Models already cached on this node (paths or HF repo IDs)
    #[serde(default)]
    pub cached_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub ok: bool,
}

// ── Deployment ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentStatus {
    Pending,
    Starting,
    Loading,
    Transferring,
    WarmingUp,
    Running,
    Paused,
    Stopped,
    Failed,
}

impl std::fmt::Display for DeploymentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Starting => write!(f, "starting"),
            Self::Loading => write!(f, "loading"),
            Self::Transferring => write!(f, "transferring"),
            Self::WarmingUp => write!(f, "warming up"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::Stopped => write!(f, "stopped"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentRole {
    /// Runs the full inference server (llama-server / mlx_lm.server)
    Main,
    /// Runs llama.cpp rpc-server to provide compute layers to the main node
    RpcWorker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployRequest {
    /// Model identifier — HuggingFace repo or local path
    /// e.g. "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" or "/path/to/model.gguf"
    pub model: String,
    /// Optional: pin to a specific node by ID (for single-node deploy)
    pub node_id: Option<Uuid>,
    /// Context length (default: 2048)
    pub context_length: Option<u32>,
    /// GPU layers to offload (default: all, i.e. 9999)
    pub gpu_layers: Option<u32>,
    /// Port for the inference server on the worker (default: 8080)
    pub port: Option<u16>,
    /// Split across multiple nodes (llama.cpp RPC tensor splitting)
    /// If set, first node is main, rest are RPC workers
    #[serde(default)]
    pub split_nodes: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployResponse {
    pub deployment_id: Uuid,
    pub node_id: Uuid,
    pub node_hostname: String,
    pub status: DeploymentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    pub id: Uuid,
    pub model: String,
    pub node_id: Uuid,
    pub node_hostname: String,
    pub status: DeploymentStatus,
    pub port: u16,
    pub context_length: u32,
    pub gpu_layers: u32,
    pub endpoint: Option<String>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Progress percentage (0-100) during loading/transferring
    #[serde(default)]
    pub progress: f32,
    /// Recent log lines from the inference process
    #[serde(default)]
    pub logs: Vec<String>,
    /// Worker nodes in a split deployment
    #[serde(default)]
    pub workers: Vec<WorkerInfo>,
    /// Inference stats from the last request
    #[serde(default)]
    pub last_tok_per_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub node_id: Uuid,
    pub hostname: String,
    pub role: AssignmentRole,
    pub rpc_port: u16,
    pub buffer_size_mb: Option<f32>,
}

/// Returned to agent when it polls for work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAssignment {
    pub deployment_id: Uuid,
    pub model: String,
    pub port: u16,
    pub context_length: u32,
    pub gpu_layers: u32,
    /// Role this node plays in the deployment
    #[serde(default = "default_role")]
    pub role: AssignmentRole,
    /// For main nodes in a split deployment: list of "host:port" for RPC workers
    #[serde(default)]
    pub rpc_servers: Vec<String>,
}

fn default_role() -> AssignmentRole {
    AssignmentRole::Main
}

/// Agent reports back deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatusUpdate {
    pub status: DeploymentStatus,
    pub endpoint: Option<String>,
    pub error: Option<String>,
    /// Progress percentage (0-100) during loading/transferring
    #[serde(default)]
    pub progress: f32,
    /// Recent log lines from the inference process
    #[serde(default)]
    pub logs: Vec<String>,
    /// Inference speed from the last request
    #[serde(default)]
    pub last_tok_per_sec: Option<f32>,
}

/// CLI/TUI sends control commands for deployments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentAction {
    Pause,
    Resume,
    Restart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentControlRequest {
    pub action: DeploymentAction,
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
    #[serde(default)]
    pub cached_models: Vec<String>,
}
