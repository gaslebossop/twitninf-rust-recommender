use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::warn;

use crate::handlers::AppState;

#[derive(Deserialize)]
pub struct InvalidateRequest {
    pub user_id: String,
}

/// Vérifie la clé interne service-to-service (X-Service-Key).
fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    // Comparaison à temps constant
    if provided.len() != secret.len() {
        return false;
    }
    provided
        .as_bytes()
        .iter()
        .zip(secret.as_bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

pub async fn invalidate_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<InvalidateRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Invalidate: unauthorized — missing or invalid X-Service-Key");
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })),
        );
    }

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }

    state.cache.invalidate_recommendations(&req.user_id).await;
    (StatusCode::OK, Json(json!({ "success": true })))
}
