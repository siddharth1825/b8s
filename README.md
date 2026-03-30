# llamactl

kubectl for local LLM clusters. A Rust CLI/TUI tool for managing distributed AI inference across multiple machines using llama.cpp RPC and MLX.

## Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ llamactl-cli │──────►│llamactl-server│◄──────│llamactl-agent│
│   (CLI/TUI)  │ HTTP  │  (axum+SQLite)│ HTTP  │  (per worker)│
└──────────────┘       └──────────────┘       └──────────────┘
                              ▲
                              │ HTTP
                       ┌──────────────┐
                       │llamactl-agent│
                       │ (more workers)│
                       └──────────────┘
```

- **llamactl-server** — Central API server. Node registry, heartbeat tracking, metrics aggregation. Axum + SQLite.
- **llamactl-agent** — Runs on each worker machine. Auto-detects hardware (Metal/CUDA/CPU), self-registers with the server, streams metrics via heartbeat.
- **llamactl** — CLI to interact with the cluster (`get nodes`, `deploy`, `top`, `logs`).

## Supported backends

| Backend | Platform | Detection |
|---------|----------|-----------|
| MLX | Apple Silicon | `aarch64` + `macOS` |
| llama.cpp RPC | Any (CUDA) | `nvidia-smi` on PATH |
| CPU | Any | Fallback |

## Quick start

### Prerequisites

- [Rust](https://rustup.rs/) (1.75+)

### Build

```bash
cargo build --release
```

### Run the server

```bash
./target/release/llamactl-server
# Listens on 0.0.0.0:6969 by default
```

### Run an agent (on each worker machine)

```bash
# On the same machine as the server
./target/release/llamactl-agent

# On a remote machine — point to the server's LAN IP
LLAMACTL_SERVER=http://192.168.1.100:6969 ./target/release/llamactl-agent
```

On Windows:
```powershell
$env:LLAMACTL_SERVER = "http://192.168.1.100:6969"
.\target\release\llamactl-agent.exe
```

### Deploy a model

```bash
# Auto-schedules to the best available node
./target/release/llamactl deploy mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

# Pin to a specific node
./target/release/llamactl deploy /path/to/model.gguf --node <node-id> --port 8080

# With custom settings
./target/release/llamactl deploy bartowski/Llama-3.1-8B-GGUF --context-length 4096 --gpu-layers 35
```

The agent on the target node will automatically start the appropriate inference server:
- **Apple Silicon nodes**: `mlx_lm.server` (requires `pip install mlx-lm`)
- **CUDA/CPU nodes**: `llama-server` (requires [llama.cpp](https://github.com/ggerganov/llama.cpp) built and on PATH)

### Split a model across multiple nodes

The killer feature — run a model too large for one machine by splitting layers across nodes via llama.cpp RPC:

```bash
# Get your node IDs
./target/release/llamactl get nodes

# Split across two nodes: first is main (runs llama-server), rest are RPC workers
./target/release/llamactl deploy model.gguf --split <mac-node-id>,<windows-node-id>
```

This starts `rpc-server` on the worker nodes and `llama-server --rpc <worker1>:50053,...` on the main node. The model's layers are distributed across all GPUs transparently.

Requirements: `llama-server` and `rpc-server` on PATH (build llama.cpp with `-DGGML_RPC=ON`).

### Unified inference endpoint

The server proxies OpenAI-compatible requests to the right node automatically:

```bash
# Point any client at the server — no need to know which node has the model
curl http://localhost:6969/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "messages": [{"role": "user", "content": "hello"}]}'

# Works with Open WebUI, LiteLLM, or any OpenAI-compatible client
# Just set the base URL to http://<server-ip>:6969/v1

# List available models
curl http://localhost:6969/v1/models
```

### Query the cluster

```bash
./target/release/llamactl get nodes
```

```
ID                                    HOSTNAME         STATUS    BACKEND         VRAM        RAM         CORES   CPU%
a1b2c3d4-...                          macbook-pro      online    mlx             -/32768 MB  14523/32768 10      23.5
e5f6a7b8-...                          win-desktop      online    llama.cpp-rpc   -/8192 MB   6234/16384  8       45.2
```

```bash
./target/release/llamactl get deployments
```

```
ID                                    MODEL                           NODE             STATUS      PORT    ENDPOINT
53374d1d-...                          ..Llama-3.1-8B-Instruct-4bit    macbook-pro      running     8080    http://macbook-pro:8080
```

### Live dashboard

```bash
./target/release/llamactl top
```

Full ratatui TUI with color-coded status, CPU bars, cached model counts. Press `q` to exit.

### Stop a deployment

```bash
./target/release/llamactl stop <deployment-id>
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIND_ADDR` | `0.0.0.0:6969` | Server listen address |
| `DATABASE_URL` | `sqlite://llamactl.db` | SQLite database path |
| `LLAMACTL_SERVER` | `http://localhost:6969` | Server URL (used by agent and CLI) |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

## Project structure

```
llamactl/
├── llamactl-common/     # Shared types (registration, heartbeat, node info)
├── llamactl-server/     # Central API server
│   ├── migrations/      # SQLite schema
│   └── src/
│       ├── main.rs      # Axum server setup
│       ├── db.rs        # Database init
│       ├── routes.rs    # API endpoints
│       └── proxy.rs     # OpenAI-compatible inference proxy
├── llamactl-agent/      # Worker agent
│   └── src/
│       ├── main.rs      # Registration + heartbeat loop
│       ├── detect.rs    # Hardware/backend auto-detection
│       └── runner.rs    # Inference process lifecycle management
└── llamactl-cli/        # CLI
    └── src/
        ├── main.rs      # deploy, get nodes/deployments, stop, top
        └── tui.rs        # Ratatui live dashboard
```

## Roadmap

- [x] Node registration + heartbeat
- [x] `llamactl deploy <model>` — schedule model deployments
- [x] `llamactl top` — live dashboard
- [x] `llamactl stop` — stop deployments
- [x] Auto-scheduling (picks node with most free memory)
- [x] llama.cpp + MLX process management
- [x] Unified OpenAI-compatible proxy (`/v1/chat/completions`)
- [x] Ratatui-based live TUI for `top`
- [x] Model registry — tracks cached models per node, prefers cache-hit scheduling
- [x] Multi-node split inference (llama.cpp RPC tensor splitting)
- [ ] `llamactl logs <deployment>` — stream logs from workers
- [ ] WebSocket live streaming for metrics
- [ ] GPU temperature + VRAM usage from inference runtimes
- [ ] Model download progress tracking
- [ ] gRPC transport (tonic) for agent ↔ server

## License

MIT
