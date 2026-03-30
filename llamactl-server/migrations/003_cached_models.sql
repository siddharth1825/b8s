CREATE TABLE IF NOT EXISTS cached_models (
    node_id     TEXT NOT NULL REFERENCES nodes(id),
    model       TEXT NOT NULL,
    PRIMARY KEY (node_id, model)
);
