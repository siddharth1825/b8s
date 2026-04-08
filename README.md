# llamactl

Run LLMs larger than your RAM. A unified CLI for distributed and memory-constrained inference on heterogeneous hardware.

## What it does

llamactl lets you run MoE (Mixture of Experts) models that don't fit in memory by streaming expert weights from SSD on demand. Only the active parameters (~3-6% of the model) stay in GPU memory. The rest streams from your NVMe drive at inference time.

**Example:** Run a 55GB model on a 24GB laptop:

```bash
llamactl serve --model Qwen3.5-122B-A10B-Q3_K_M.gguf --pin-top 10
```

llamactl analyzes the model, repacks expert weights for fast SSD access, pins the 10 hottest experts in RAM, and launches an OpenAI-compatible server. Total RAM usage: ~5GB.

## Results

| Model | Size on disk | Hardware | RAM used | Speed |
|-------|-------------|----------|----------|-------|
| Qwen3.5-122B-A10B | 55 GB (Q3_K_M) | M4 Pro 24GB | ~5.3 GB | 1.78 tok/s |
| Qwen3.5-35B-A3B | 16 GB (Q3_K_M) | M4 Pro 24GB | ~3.7 GB | 45 tok/s (fits in RAM) |

## How it works

MoE models have two types of weights:

- **Active parameters** (~3-8%): attention, SSM, router, norms, embeddings. Small, needed every token.
- **Expert parameters** (~92-97%): hundreds of expert FFN networks. Only K (typically 4-8) are active per token.

llamactl exploits this structure:

1. **Analyze** the model to identify expert vs active tensors
2. **Repack** expert weights into per-layer binary files optimized for `pread()`
3. **Serve** with only active params on GPU, streaming experts from SSD per token
4. **Pin** frequently-used ("hot") experts in RAM for instant access

```
GPU Metal (3.7 GB)                    SSD (51.7 GB)
+-----------------------+             +------------------+
| attention weights     |  pread()    | layer_000.bin    |
| SSM/Mamba weights     | <--------  | layer_001.bin    |
| router/gate weights   |  8 experts | ...              |
| norms + embeddings    |  per layer  | layer_047.bin    |
+-----------------------+             +------------------+
       |                                    |
       v                                    v
  Metal GPU compute              K=8 active experts
  (attention + routing)          streamed on demand
```

## Installation

```bash
# Clone with submodule
git clone --recursive https://github.com/siddharth1825/b8s.git
cd b8s

# Build the Rust CLI
cargo build --release --bin llamactl

# Build the llama.cpp backend (Metal for macOS)
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)
cd ..
```

For NVIDIA GPUs (CUDA):
```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
```

## Commands

### `llamactl serve` - Run inference

The main command. Analyzes the model, selects strategy, and launches an OpenAI-compatible API.

```bash
# Auto-detect strategy (full-ram if model fits, flash-ssd otherwise)
llamactl serve --model model.gguf

# Force SSD streaming with 20 hot experts pinned
llamactl serve --model Qwen3.5-122B.gguf --strategy flash-ssd --pin-top 20

# Custom port and context length
llamactl serve --model model.gguf --port 9090 -c 2048
```

The API is OpenAI-compatible:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hello!"}]}'
```

### `llamactl analyze` - Inspect a model

Shows tensor breakdown, expert statistics, SSD streaming estimates, and hardware fit analysis.

```bash
llamactl analyze model.gguf
```

Output includes:
- Tensor breakdown by category (expert FFN, attention, SSM, etc.)
- Device placement recommendations (FastGPU vs BigMemory)
- Per-expert routing weight analysis (hot/cold classification)
- SSD streaming speed estimates for different drives
- Hardware fit checks for common GPUs

### `llamactl repack` - Prepare experts for SSD streaming

Converts GGUF expert tensors into per-layer binary files optimized for `pread()`.

```bash
llamactl repack model.gguf
# Creates model.experts/ directory with layer_000.bin ... layer_047.bin + manifest.json
```

- Handles mixed quantization (Q3K/Q4K/Q5K per layer in Q3_K_M models)
- Writes per-layer JSON metadata for exact tensor sizes
- Works with any MoE model and any GGUF quant type

### Cluster commands

For multi-node deployments across a LAN:

```bash
# Start the coordinator
llamactl-server

# On each node, start the agent
llamactl-agent --server http://coordinator:6969

# Deploy and manage from any machine
llamactl get nodes
llamactl deploy model.gguf --split <node1>,<node2>
llamactl top  # interactive TUI dashboard
```

## Architecture

```
llamactl (Rust CLI)
  |-- serve       Launch llama-server with optimal strategy
  |-- analyze     Parse GGUF, categorize tensors, profile experts
  |-- repack      Convert experts to per-layer pread-optimized files
  |-- top         Interactive cluster dashboard (ratatui TUI)
  |-- deploy      Multi-node deployment via server/agent
  
llama.cpp fork (C++/Metal)
  |-- flash-expert     SSD streaming: pread + Metal GPU dequant+matvec
  |-- remote-expert    Network split: gRPC expert offloading
  |-- expert-worker    Standalone expert compute server
  |-- Metal shaders    Q3K/Q4K/Q5K dequant+matvec (scalar + SIMD)
```

### Crates

| Crate | Purpose |
|-------|---------|
| `llamactl-cli` | User-facing CLI: serve, analyze, repack, top, deploy |
| `llamactl-engine` | GGUF parser, tensor categorization, inference DAG, candle inference |
| `llamactl-common` | Shared types for server/agent/CLI communication |
| `llamactl-server` | Coordinator: node registry, deployment scheduler, inference proxy |
| `llamactl-agent` | Per-node agent: hardware detection, deployment runner |

### Flash-Expert (llama.cpp fork)

The key innovation. Patches on top of llama.cpp:

- **Hook in `build_moe_ffn`**: Replaces the expert FFN subgraph with a custom op
- **Tensor skipping**: `--no-mmap` mode skips expert tensor allocation (saves 50+ GB)
- **SSD streaming**: `pread()` only the K active experts per layer from repacked files
- **Metal GPU shaders**: Dequant+matvec for Q3K/Q4K/Q5K directly on GPU
- **Expert pinning**: User-configured hot experts kept in RAM
- **Shared expert caching**: Loaded once at startup, served from memory

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) for Metal GPU
- Or Linux/Windows with NVIDIA GPU for CUDA
- Rust 1.70+ for the CLI
- CMake 3.14+ for llama.cpp
- Fast NVMe SSD recommended for flash-expert mode

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - the inference backend
- [flash-moe](https://github.com/danveloper/flash-moe) - inspiration for SSD expert streaming
- [candle](https://github.com/huggingface/candle) - Rust ML framework used in early prototyping
