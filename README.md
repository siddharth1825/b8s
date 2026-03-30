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

### Query the cluster

```bash
./target/release/llamactl get nodes
```

```
ID                                    HOSTNAME         STATUS    BACKEND         VRAM        RAM         CORES   CPU%
a1b2c3d4-...                          macbook-pro      online    mlx             -/32768 MB  14523/32768 10      23.5
e5f6a7b8-...                          win-desktop      online    llama.cpp-rpc   -/8192 MB   6234/16384  8       45.2
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
│       └── routes.rs    # API endpoints
├── llamactl-agent/      # Worker agent
│   └── src/
│       ├── main.rs      # Registration + heartbeat loop
│       └── detect.rs    # Hardware/backend auto-detection
└── llamactl-cli/        # CLI
    └── src/main.rs      # clap CLI with `get nodes`
```

## Roadmap

- [ ] `llamactl deploy <model>` — schedule model deployments across nodes
- [ ] `llamactl top` — live TUI dashboard (ratatui)
- [ ] `llamactl logs <deployment>` — stream logs from workers
- [ ] WebSocket live streaming for metrics
- [ ] GPU temperature + VRAM usage from inference runtimes
- [ ] llama.cpp RPC server process management
- [ ] MLX distributed inference orchestration
- [ ] gRPC transport (tonic) for agent ↔ server

## License

MIT
