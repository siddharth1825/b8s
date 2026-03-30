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

    let hostname = System::host_name().unwrap_or_else(|| "unknown".into());
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

fn detect_backend() -> (BackendKind, Option<u64>) {
    // Check for Apple Silicon (Metal)
    if cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64" {
        // On Apple Silicon, unified memory acts as VRAM
        // Report total system memory as available VRAM since it's unified
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

pub fn collect_metrics() -> (u64, f32) {
    let sys = System::new_all();
    let ram_used_mb = sys.used_memory() / (1024 * 1024);
    let cpu_usage = sys.global_cpu_usage();
    (ram_used_mb, cpu_usage)
}
