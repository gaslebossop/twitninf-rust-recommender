use axum::{extract::State, http::{HeaderMap, StatusCode}, Json};
use serde_json::{json, Value};
use tracing::{error, warn};

use crate::handlers::AppState;
use crate::models::RecommendRequest;

/// Vérifie la clé interne service-to-service (comparaison à temps constant).
fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.len() != secret.len() { return false; }
    provided.as_bytes().iter().zip(secret.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub async fn recommend_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<RecommendRequest>,
) -> (StatusCode, Json<Value>) {
    // `/track` et `/invalidate` exigeaient déjà la clé interne ; `/recommendations`
    // était la seule route métier ouverte. Le service n'écoute que sur 127.0.0.1,
    // mais n'importe quel process local pouvait donc lire le fil de n'importe
    // quel utilisateur en connaissant son UUID.
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Recommendations: unauthorized — missing or invalid X-Service-Key");
        return (StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })));
    }

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "invalid user_id: must be a UUID" })));
    }

    match state.recommender.recommend(&req).await {
        Ok(resp) => (StatusCode::OK, Json(json!(resp))),
        Err(e) => {
            // Log complet en interne, message générique vers le client
            error!("Recommendation error for user {}: {:?}", req.user_id, e);
            (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "success": false, "error": "Internal server error" })))
        }
    }
}
