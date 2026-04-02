use llamactl_common::BackendKind;
use sysinfo::System;

pub struct HardwareInfo {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub backend: BackendKind,
    pub vram_mb: Option<u64>,
    pub total_ram_mb: u64,
    pub cpu_cores: u32,
}

pub fn detect_hardware() -> HardwareInfo {
    let sys = System::new_all();

    // Use LAN IP as hostname for RPC connectivity (mDNS often broken)
    // Override with LLAMACTL_HOSTNAME if needed
    let hostname = std::env::var("LLAMACTL_HOSTNAME").unwrap_or_else(|_| {
        detect_lan_ip().unwrap_or_else(|| {
            System::host_name().unwrap_or_else(|| "unknown".into())
        })
    });

    let os = format!(
        "{} {}",
        System::name().unwrap_or_else(|| "unknown".into()),
        System::os_version().unwrap_or_else(|| "".into())
    );
    let arch = std::env::consts::ARCH.to_string();
    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let cpu_cores = sys.cpus().len() as u32;

    let (backend, vram_mb) = detect_backend();

    HardwareInfo {
        hostname,
        os,
        arch,
        backend,
        vram_mb,
        total_ram_mb,
        cpu_cores,
    }
}

/// Detect the primary LAN IP (public wrapper for runner)
pub fn detect_lan_ip_pub() -> Option<String> {
    detect_lan_ip()
}

/// Detect the primary LAN IP by opening a UDP socket to a public address.
/// No traffic is actually sent — this just lets the OS pick the right interface.
fn detect_lan_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

fn detect_backend() -> (BackendKind, Option<u64>) {
    // Allow manual override: LLAMACTL_BACKEND=llama_cpp_rpc|mlx|cpu
    if let Ok(override_val) = std::env::var("LLAMACTL_BACKEND") {
        let backend = match override_val.to_lowercase().as_str() {
            "llama_cpp_rpc" | "llamacpp" | "llama" => BackendKind::LlamaCppRpc,
            "mlx" => BackendKind::Mlx,
            "cpu" => BackendKind::Cpu,
            other => {
                tracing::warn!(
                    value = other,
                    "Unknown LLAMACTL_BACKEND value, falling back to auto-detect"
                );
                return auto_detect_backend();
            }
        };
        let vram = detect_vram(&backend);
        return (backend, vram);
    }

    auto_detect_backend()
}

fn auto_detect_backend() -> (BackendKind, Option<u64>) {
    // Check for Apple Silicon (Metal)
    if cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64" {
        let sys = System::new_all();
        let total_mb = sys.total_memory() / (1024 * 1024);
        return (BackendKind::Mlx, Some(total_mb));
    }

    // Check for NVIDIA GPU via nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(vram) = s.trim().parse::<u64>() {
                    return (BackendKind::LlamaCppRpc, Some(vram));
                }
            }
        }
    }

    (BackendKind::Cpu, None)
}

fn detect_vram(backend: &BackendKind) -> Option<u64> {
    match backend {
        BackendKind::Mlx => {
            let sys = System::new_all();
            Some(sys.total_memory() / (1024 * 1024))
        }
        BackendKind::LlamaCppRpc => {
            // Try nvidia-smi, fall back to unified memory on Mac
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=memory.total")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if output.status.success() {
                    if let Ok(s) = String::from_utf8(output.stdout) {
                        if let Ok(vram) = s.trim().parse::<u64>() {
                            return Some(vram);
                        }
                    }
                }
            }
            // Apple Silicon with llama.cpp — unified memory is VRAM
            if cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64" {
                let sys = System::new_all();
                return Some(sys.total_memory() / (1024 * 1024));
            }
            None
        }
        BackendKind::Cpu => None,
    }
}

pub fn collect_metrics() -> (u64, f32) {
    let sys = System::new_all();
    let ram_used_mb = sys.used_memory() / (1024 * 1024);
    let cpu_usage = sys.global_cpu_usage();
    (ram_used_mb, cpu_usage)
}

/// Scan for cached models in standard locations:
/// - HuggingFace hub cache (~/.cache/huggingface/hub/)
/// - MLX cache (~/.cache/huggingface/hub/ with mlx in the name)
pub fn scan_cached_models() -> Vec<String> {
    let mut models = Vec::new();

    // HuggingFace cache: ~/.cache/huggingface/hub/models--<org>--<name>/
    if let Some(home) = dirs_hint() {
        let hf_hub = home.join(".cache/huggingface/hub");
        if hf_hub.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&hf_hub) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with("models--") {
                        // Convert "models--org--name" -> "org/name"
                        let model_id = name
                            .strip_prefix("models--")
                            .unwrap()
                            .replacen("--", "/", 1);
                        models.push(model_id);
                    }
                }
            }
        }
    }

    models.sort();
    models.dedup();
    models
}

fn dirs_hint() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
}
