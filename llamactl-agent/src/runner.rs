use anyhow::Result;
use llamactl_common::*;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use uuid::Uuid;

const MAX_LOG_LINES: usize = 50;

/// Shared state for a running deployment — updated by the log parser task
struct DeploymentState {
    status: DeploymentStatus,
    progress: f32,
    logs: Vec<String>,
    last_tok_per_sec: Option<f32>,
    endpoint: Option<String>,
    error: Option<String>,
    dirty: bool, // whether state changed since last report
}

pub struct Runner {
    client: reqwest::Client,
    server_url: String,
    backend: BackendKind,
    running: HashMap<Uuid, Child>,
    states: HashMap<Uuid, Arc<Mutex<DeploymentState>>>,
}

impl Runner {
    pub fn new(client: reqwest::Client, server_url: String, backend: BackendKind) -> Self {
        Self {
            client,
            server_url,
            backend,
            running: HashMap::new(),
            states: HashMap::new(),
        }
    }

    pub async fn poll(&mut self, node_id: Uuid) {
        let assignments = match self.fetch_assignments(node_id).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("Failed to fetch assignments: {e}");
                return;
            }
        };

        let assigned_ids: std::collections::HashSet<Uuid> =
            assignments.iter().map(|a| a.deployment_id).collect();

        // Stop processes that are no longer assigned
        let stale: Vec<Uuid> = self.running.keys()
            .filter(|id| !assigned_ids.contains(id))
            .copied()
            .collect();
        for id in stale {
            tracing::info!(deployment_id = %id, "Stopping removed deployment");
            if let Some(mut child) = self.running.remove(&id) {
                let _ = child.kill().await;
            }
            self.states.remove(&id);
        }

        // Report status for running deployments
        for (dep_id, state) in &self.states {
            let mut s = state.lock().await;
            if s.dirty {
                self.report_status_from_state(*dep_id, &s).await;
                s.dirty = false;
            }
        }

        // Check running processes for exit
        let mut exited: Vec<(Uuid, Option<i32>)> = Vec::new();
        for (dep_id, child) in &mut self.running {
            match child.try_wait() {
                Ok(Some(exit)) => {
                    tracing::warn!(
                        deployment_id = %dep_id,
                        exit_code = ?exit.code(),
                        "Inference process exited"
                    );
                    exited.push((*dep_id, exit.code()));
                }
                Ok(None) => {}
                Err(e) => tracing::error!("Failed to check process: {e}"),
            }
        }
        for (id, exit_code) in &exited {
            if let Some(state) = self.states.get(id) {
                let mut s = state.lock().await;
                s.status = DeploymentStatus::Failed;
                s.error = Some(format!("Process exited with code {:?}", exit_code));
                self.report_status_from_state(*id, &s).await;
            }
            self.running.remove(id);
            self.states.remove(id);
        }

        // Start new assignments
        for assignment in assignments {
            if self.running.contains_key(&assignment.deployment_id) {
                continue;
            }

            tracing::info!(
                deployment_id = %assignment.deployment_id,
                model = %assignment.model,
                port = assignment.port,
                role = ?assignment.role,
                "Starting inference server"
            );

            // Create shared state
            let state = Arc::new(Mutex::new(DeploymentState {
                status: DeploymentStatus::Starting,
                progress: 0.0,
                logs: Vec::new(),
                last_tok_per_sec: None,
                endpoint: None,
                error: None,
                dirty: true,
            }));

            // Report starting status immediately
            {
                let s = state.lock().await;
                self.report_status_from_state(assignment.deployment_id, &s).await;
            }

            match self.start_inference(&assignment).await {
                Ok(child) => {
                    self.states.insert(assignment.deployment_id, state);
                    self.running.insert(assignment.deployment_id, child);
                }
                Err(e) => {
                    tracing::error!(
                        deployment_id = %assignment.deployment_id,
                        err = %e,
                        "Failed to start inference server"
                    );
                    self.report_status(
                        assignment.deployment_id,
                        DeploymentStatus::Failed,
                        None,
                        Some(e.to_string()),
                    ).await;
                }
            }
        }
    }

    async fn start_inference(&self, assignment: &NodeAssignment) -> Result<Child> {
        match assignment.role {
            AssignmentRole::RpcWorker => self.start_rpc_worker(assignment).await,
            AssignmentRole::Main => match self.backend {
                BackendKind::Mlx => self.start_mlx(assignment).await,
                BackendKind::LlamaCppRpc | BackendKind::Cpu => {
                    self.start_llama_cpp(assignment).await
                }
            },
        }
    }

    async fn start_llama_cpp(&self, a: &NodeAssignment) -> Result<Child> {
        let bin = find_binary(&["llama-server", "llama.cpp/build/bin/llama-server"])
            .ok_or_else(|| anyhow::anyhow!(
                "llama-server not found in PATH"
            ))?;

        let mut cmd = Command::new(bin);
        cmd.arg("-m").arg(&a.model)
            .arg("--port").arg(a.port.to_string())
            .arg("-c").arg(a.context_length.to_string())
            .arg("-ngl").arg(a.gpu_layers.to_string())
            .arg("--host").arg("0.0.0.0");

        if !a.rpc_servers.is_empty() {
            let rpc_list = a.rpc_servers.join(",");
            cmd.arg("--rpc").arg(&rpc_list);
            tracing::info!(rpc_servers = %rpc_list, "RPC tensor splitting enabled");
        }

        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        // Spawn log parser for stderr
        if let Some(stderr) = child.stderr.take() {
            let state = self.states.get(&a.deployment_id).cloned();
            let dep_id = a.deployment_id;
            let port = a.port;
            let rpc_count = a.rpc_servers.len();
            tokio::spawn(async move {
                parse_llama_output(stderr, state, dep_id, port, rpc_count).await;
            });
        }

        // Also pipe stdout
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(_)) = lines.next_line().await {}
            });
        }

        Ok(child)
    }

    async fn start_rpc_worker(&self, a: &NodeAssignment) -> Result<Child> {
        let bin = find_binary(&["rpc-server", "llama.cpp/build/bin/rpc-server"])
            .ok_or_else(|| anyhow::anyhow!(
                "rpc-server not found in PATH"
            ))?;

        tracing::info!(port = a.port, "Starting RPC worker");

        let mut child = Command::new(bin)
            .arg("--host").arg("0.0.0.0")
            .arg("--port").arg(a.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        // Parse RPC worker output
        if let Some(stderr) = child.stderr.take() {
            let state = self.states.get(&a.deployment_id).cloned();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // Log important lines
                    if line.contains("Accepted client") || line.contains("Starting RPC") || line.contains("error") {
                        tracing::info!(target: "rpc-worker", "{}", line);
                    }
                    if let Some(state) = &state {
                        let mut s = state.lock().await;
                        push_log(&mut s.logs, &line);
                        if line.contains("Accepted client connection") {
                            s.status = DeploymentStatus::Running;
                            s.dirty = true;
                        }
                    }
                }
            });
        }

        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(_)) = lines.next_line().await {}
            });
        }

        // Report running immediately for RPC workers
        if let Some(state) = self.states.get(&a.deployment_id) {
            let mut s = state.lock().await;
            s.status = DeploymentStatus::Running;
            s.endpoint = Some(format!("rpc://0.0.0.0:{}", a.port));
            s.dirty = true;
        }

        Ok(child)
    }

    async fn start_mlx(&self, a: &NodeAssignment) -> Result<Child> {
        let mut child = Command::new("python3")
            .arg("-m").arg("mlx_lm.server")
            .arg("--model").arg(&a.model)
            .arg("--port").arg(a.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        if let Some(stderr) = child.stderr.take() {
            let state = self.states.get(&a.deployment_id).cloned();
            let port = a.port;
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(target: "mlx", "{}", line);
                    if let Some(state) = &state {
                        let mut s = state.lock().await;
                        push_log(&mut s.logs, &line);
                        if line.contains("Starting server") || line.contains("Uvicorn running") {
                            s.status = DeploymentStatus::Running;
                            s.endpoint = Some(format!("http://0.0.0.0:{}", port));
                            s.dirty = true;
                        }
                    }
                }
            });
        }

        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(_)) = lines.next_line().await {}
            });
        }

        Ok(child)
    }

    async fn fetch_assignments(&self, node_id: Uuid) -> Result<Vec<NodeAssignment>> {
        let resp = self.client
            .get(format!("{}/api/nodes/{}/assignments", self.server_url, node_id))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp)
    }

    async fn report_status_from_state(&self, deployment_id: Uuid, s: &DeploymentState) {
        let update = DeploymentStatusUpdate {
            status: s.status,
            endpoint: s.endpoint.clone(),
            error: s.error.clone(),
            progress: s.progress,
            logs: s.logs.clone(),
            last_tok_per_sec: s.last_tok_per_sec,
        };
        let _ = self.client
            .put(format!("{}/api/deployments/{}/status", self.server_url, deployment_id))
            .json(&update)
            .send()
            .await;
    }

    async fn report_status(
        &self,
        deployment_id: Uuid,
        status: DeploymentStatus,
        endpoint: Option<String>,
        error: Option<String>,
    ) {
        let update = DeploymentStatusUpdate {
            status,
            endpoint,
            error,
            progress: 0.0,
            logs: vec![],
            last_tok_per_sec: None,
        };
        let _ = self.client
            .put(format!("{}/api/deployments/{}/status", self.server_url, deployment_id))
            .json(&update)
            .send()
            .await;
    }
}

/// Parse llama-server stderr output, updating deployment state in real time
async fn parse_llama_output(
    stderr: tokio::process::ChildStderr,
    state: Option<Arc<Mutex<DeploymentState>>>,
    dep_id: Uuid,
    port: u16,
    rpc_count: usize,
) {
    let reader = BufReader::new(stderr);
    let mut lines = reader.lines();
    let total_stages = if rpc_count > 0 { 4 } else { 3 }; // fit, load, warmup, (rpc connect)
    let mut current_stage = 0;

    while let Ok(Some(line)) = lines.next_line().await {
        // Always log to terminal
        eprintln!("[{:.8}] {}", dep_id, line);

        let Some(state) = &state else { continue };
        let mut s = state.lock().await;
        push_log(&mut s.logs, &line);

        // Parse progress stages from llama-server output
        if line.contains("fitting params to") {
            s.status = DeploymentStatus::Loading;
            s.progress = 10.0;
            s.dirty = true;
        } else if line.contains("loading model tensors") {
            s.status = DeploymentStatus::Transferring;
            s.progress = 20.0;
            s.dirty = true;
        } else if line.contains("RPC") && line.contains("model buffer size") {
            // RPC buffer transfer in progress
            current_stage += 1;
            let rpc_progress = 20.0 + (current_stage as f32 / total_stages as f32) * 40.0;
            s.progress = rpc_progress.min(60.0);
            s.dirty = true;
        } else if line.contains("MTL0") && line.contains("model buffer size") {
            s.progress = 60.0;
            s.dirty = true;
        } else if line.contains("offloaded") && line.contains("layers to GPU") {
            s.progress = 70.0;
            s.dirty = true;
        } else if line.contains("warming up") {
            s.status = DeploymentStatus::WarmingUp;
            s.progress = 80.0;
            s.dirty = true;
        } else if line.contains("model loaded") {
            s.progress = 95.0;
            s.dirty = true;
        } else if line.contains("server is listening") {
            s.status = DeploymentStatus::Running;
            s.progress = 100.0;
            s.endpoint = Some(format!("http://0.0.0.0:{}", port));
            s.dirty = true;
        } else if line.contains("predicted_per_second") || line.contains("eval time") {
            // Parse tok/s from timing line: "eval time = X ms / Y tokens (Z ms per token, W tokens per second)"
            if let Some(tps) = parse_tok_per_sec(&line) {
                s.last_tok_per_sec = Some(tps);
                s.dirty = true;
            }
        } else if line.contains("error") || line.contains("failed") || line.contains("GGML_ABORT") {
            s.error = Some(line.clone());
            s.dirty = true;
        }
    }
}

/// Parse "X tokens per second" from llama-server eval timing line
fn parse_tok_per_sec(line: &str) -> Option<f32> {
    // Pattern: "N.N tokens per second"
    let idx = line.find("tokens per second")?;
    let before = &line[..idx].trim_end();
    let num_str = before.rsplit(|c: char| !c.is_ascii_digit() && c != '.').next()?;
    num_str.parse::<f32>().ok()
}

fn push_log(logs: &mut Vec<String>, line: &str) {
    // Skip noisy metal pipeline compilation lines
    if line.contains("ggml_metal_library_compile_pipeline") {
        return;
    }
    logs.push(line.to_string());
    if logs.len() > MAX_LOG_LINES {
        logs.drain(0..logs.len() - MAX_LOG_LINES);
    }
}

fn find_binary(names: &[&str]) -> Option<String> {
    for name in names {
        if which::which(name).is_ok() {
            return Some(name.to_string());
        }
    }
    None
}
