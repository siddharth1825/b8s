use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use sqlx::SqlitePool;

/// Proxy all /v1/* requests to the appropriate inference endpoint.
///
/// Routing logic:
/// 1. If `model` field in JSON body matches a deployment's model, route there.
/// 2. If only one running deployment exists, route there.
/// 3. Otherwise return 400 with available models.
pub async fn proxy_inference(
    State(pool): State<SqlitePool>,
    req: Request,
) -> Result<Response, (StatusCode, String)> {
    let (parts, body) = req.into_parts();
    let path = parts.uri.path().to_string();

    // Read body bytes so we can inspect the model field and forward
    let body_bytes = axum::body::to_bytes(body, 10 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read body: {e}")))?;

    // Try to extract model from request body
    let requested_model = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str().map(String::from)));

    // Find running deployments
    let deployments = sqlx::query_as::<_, DeploymentEndpoint>(
        "SELECT d.id, d.model, d.port, n.hostname
         FROM deployments d JOIN nodes n ON d.node_id = n.id
         WHERE d.status = 'running'"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if deployments.is_empty() {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "No running deployments".into()));
    }

    // Pick the right deployment
    let target = if let Some(ref model) = requested_model {
        // Match by model name (exact or suffix match for flexibility)
        deployments
            .iter()
            .find(|d| d.model == *model || d.model.ends_with(model) || model.ends_with(&d.model))
    } else {
        None
    };

    let target = match target {
        Some(t) => t,
        None if deployments.len() == 1 => &deployments[0],
        None => {
            let models: Vec<&str> = deployments.iter().map(|d| d.model.as_str()).collect();
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Multiple models deployed. Specify \"model\" in your request. Available: {:?}",
                    models
                ),
            ));
        }
    };

    let upstream_url = format!("http://{}:{}{}", target.hostname, target.port, path);

    tracing::debug!(
        model = %target.model,
        upstream = %upstream_url,
        "Proxying inference request"
    );

    // Forward the request
    let client = reqwest::Client::new();
    let mut upstream_req = client.post(&upstream_url);

    // Forward relevant headers
    if let Some(ct) = parts.headers.get(header::CONTENT_TYPE) {
        upstream_req = upstream_req.header(header::CONTENT_TYPE, ct);
    }
    if let Some(auth) = parts.headers.get(header::AUTHORIZATION) {
        upstream_req = upstream_req.header(header::AUTHORIZATION, auth);
    }

    let upstream_resp = upstream_req
        .body(body_bytes.to_vec())
        .send()
        .await
        .map_err(|e| {
            tracing::error!(err = %e, upstream = %upstream_url, "Upstream request failed");
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {e}"))
        })?;

    // Stream the response back (important for SSE streaming completions)
    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut response_builder = Response::builder().status(status);

    // Forward response headers
    for (key, value) in upstream_resp.headers() {
        response_builder = response_builder.header(key, value);
    }

    let stream = upstream_resp.bytes_stream().map(|result| {
        result.map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })
    });

    let body = Body::from_stream(stream);

    response_builder.body(body).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to build response: {e}"))
    })
}

/// GET /v1/models — return list of deployed models in OpenAI-compatible format
pub async fn list_models(
    State(pool): State<SqlitePool>,
) -> Result<impl IntoResponse, StatusCode> {
    let deployments = sqlx::query_as::<_, DeploymentEndpoint>(
        "SELECT d.id, d.model, d.port, n.hostname
         FROM deployments d JOIN nodes n ON d.node_id = n.id
         WHERE d.status = 'running'"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list models: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let models: Vec<serde_json::Value> = deployments
        .iter()
        .map(|d| {
            serde_json::json!({
                "id": d.model,
                "object": "model",
                "owned_by": "llamactl",
            })
        })
        .collect();

    Ok(axum::Json(serde_json::json!({
        "object": "list",
        "data": models,
    })))
}

#[derive(sqlx::FromRow)]
struct DeploymentEndpoint {
    #[allow(dead_code)]
    id: String,
    model: String,
    port: i32,
    hostname: String,
}
