CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    hostname        TEXT NOT NULL,
    os              TEXT NOT NULL,
    arch            TEXT NOT NULL,
    backend         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'online',
    vram_mb         INTEGER,
    vram_used_mb    INTEGER,
    total_ram_mb    INTEGER NOT NULL,
    ram_used_mb     INTEGER NOT NULL DEFAULT 0,
    cpu_cores       INTEGER NOT NULL,
    cpu_usage_pct   REAL NOT NULL DEFAULT 0.0,
    gpu_temp_c      REAL,
    tokens_per_sec  REAL,
    registered_at   TEXT NOT NULL,
    last_heartbeat  TEXT NOT NULL
);
