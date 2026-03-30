-- Tracks which nodes participate in a split deployment and their role
CREATE TABLE IF NOT EXISTS deployment_workers (
    deployment_id   TEXT NOT NULL REFERENCES deployments(id),
    node_id         TEXT NOT NULL REFERENCES nodes(id),
    role            TEXT NOT NULL DEFAULT 'main',  -- 'main' or 'rpc_worker'
    rpc_port        INTEGER NOT NULL DEFAULT 50052,
    PRIMARY KEY (deployment_id, node_id)
);
