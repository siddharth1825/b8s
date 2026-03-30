CREATE TABLE IF NOT EXISTS deployments (
    id              TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    node_id         TEXT NOT NULL REFERENCES nodes(id),
    status          TEXT NOT NULL DEFAULT 'pending',
    port            INTEGER NOT NULL DEFAULT 8080,
    context_length  INTEGER NOT NULL DEFAULT 2048,
    gpu_layers      INTEGER NOT NULL DEFAULT 9999,
    endpoint        TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL
);
