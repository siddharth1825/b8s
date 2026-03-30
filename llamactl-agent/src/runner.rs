use anyhow::Result;
use llamactl_common::*;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::{Child, Command};
use uuid::Uuid;

pub struct Runner {
    client: reqwest::Client,
    server_url: String,
    backend: BackendKind,
    /// deployment_id -> running child process
    running: HashMap<Uuid, Child>,
}

impl Runner {
    pub fn new(client: reqwest::Client, server_url: String, backend: BackendKind) -> Self {
        Self {
            client,
            server_url,
            backend,
            running: HashMap::new(),
        }
    }

    /// Poll server for assignments and reconcile with running processes
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

        // Stop processes that are no longer assigned (stopped/deleted on server)
        let stale: Vec<Uuid> = self.running.keys()
            .filter(|id| !assigned_ids.contains(id))
            .copied()
            .collect();
        for id in stale {
            tracing::info!(deployment_id = %id, "Stopping removed deployment");
            if let Some(mut child) = self.running.remove(&id) {
                let _ = child.kill().await;
            }
        }

        // Start new assignments
        for assignment in assignments {
            if self.running.contains_key(&assignment.deployment_id) {
                // Check if process is still alive
                if let Some(child) = self.running.get_mut(&assignment.deployment_id) {
                    match child.try_wait() {
                        Ok(Some(exit)) => {
                            tracing::warn!(
                                deployment_id = %assignment.deployment_id,
                                exit_code = ?exit.code(),
                                "Inference process exited unexpectedly"
                            );
                            self.running.remove(&assignment.deployment_id);
                            self.report_status(
                                assignment.deployment_id,
                                DeploymentStatus::Failed,
                                None,
                                Some(format!("Process exited with code {:?}", exit.code())),
                            ).await;
                        }
                        Ok(None) => {} // still running, all good
                        Err(e) => {
                            tracing::error!("Failed to check process status: {e}");
                        }
                    }
                    continue;
                }
            }

            // Not running yet — start it
            tracing::info!(
                deployment_id = %assignment.deployment_id,
                model = %assignment.model,
                port = assignment.port,
                "Starting inference server"
            );

            match self.start_inference(&assignment).await {
                Ok(child) => {
                    self.running.insert(assignment.deployment_id, child);
                    let endpoint = format!("http://0.0.0.0:{}", assignment.port);
                    self.report_status(
                        assignment.deployment_id,
                        DeploymentStatus::Running,
                        Some(endpoint),
                        None,
                    ).await;
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
                "llama-server not found in PATH. Install llama.cpp and ensure llama-server is on your PATH."
            ))?;

        let mut cmd = Command::new(bin);
        cmd.arg("-m").arg(&a.model)
            .arg("--port").arg(a.port.to_string())
            .arg("-c").arg(a.context_length.to_string())
            .arg("-ngl").arg(a.gpu_layers.to_string())
            .arg("--host").arg("0.0.0.0");

        // If this is a split deployment, pass RPC servers for tensor splitting
        if !a.rpc_servers.is_empty() {
            let rpc_list = a.rpc_servers.join(",");
            cmd.arg("--rpc").arg(&rpc_list);
            tracing::info!(
                rpc_servers = %rpc_list,
                "Starting llama-server with RPC tensor splitting"
            );
        }

        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        Ok(child)
    }

    /// Start llama.cpp rpc-server — provides compute to a remote main node
    async fn start_rpc_worker(&self, a: &NodeAssignment) -> Result<Child> {
        let bin = find_binary(&["rpc-server", "llama.cpp/build/bin/rpc-server"])
            .ok_or_else(|| anyhow::anyhow!(
                "rpc-server not found in PATH. Build llama.cpp with RPC support (cmake -DGGML_RPC=ON)."
            ))?;

        tracing::info!(
            port = a.port,
            "Starting llama.cpp RPC worker"
        );

        let child = Command::new(bin)
            .arg("--host").arg("0.0.0.0")
            .arg("--port").arg(a.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        Ok(child)
    }

    async fn start_mlx(&self, a: &NodeAssignment) -> Result<Child> {
        let child = Command::new("python3")
            .arg("-m").arg("mlx_lm.server")
            .arg("--model").arg(&a.model)
            .arg("--port").arg(a.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

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

    async fn report_status(
        &self,
        deployment_id: Uuid,
        status: DeploymentStatus,
        endpoint: Option<String>,
        error: Option<String>,
    ) {
        let update = DeploymentStatusUpdate { status, endpoint, error };
        if let Err(e) = self.client
            .put(format!("{}/api/deployments/{}/status", self.server_url, deployment_id))
            .json(&update)
            .send()
            .await
        {
            tracing::error!(deployment_id = %deployment_id, err = %e, "Failed to report status");
        }
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
