//! `POST /velocity-throttle` — pose le frein d'une heure (voir `crate::velocity`).
//!
//! Appelé par l'API Node juste après qu'une action déclenchante a réussi
//! (suppression d'un tweet, changement d'avatar/bio, rafale de publication) —
//! jamais par une décision de modération, qui passe par `/admin/strike`.

use axum::{extract::State, http::{HeaderMap, StatusCode}, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::handlers::AppState;

fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.len() != secret.len() { return false; }
    provided.as_bytes().iter().zip(secret.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

#[derive(Debug, Deserialize)]
pub struct VelocityThrottleRequest {
    pub user_id: String,
    /// Ce qui a déclenché le frein — jamais interprété ici, juste journalisé :
    /// c'est ce qui permet de distinguer un pic légitime d'un abus au moment
    /// de relire les logs.
    #[serde(default)]
    pub reason: Option<String>,
}

pub async fn velocity_throttle_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<VelocityThrottleRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("VelocityThrottle: unauthorized — missing or invalid X-Service-Key");
        return (StatusCode::UNAUTHORIZED, Json(json!({ "success": false, "error": "Unauthorized" })));
    }

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }

    state.cache.set_velocity_throttle(&req.user_id).await;
    info!(user_id = %req.user_id, reason = req.reason.as_deref().unwrap_or("unspecified"),
          "Frein de vélocité posé (1h, ×0.5)");

    (StatusCode::OK, Json(json!({ "success": true })))
}

/// `POST /velocity/post-burst` — compte une publication, pose le frein si la
/// rafale franchit le seuil (voir `crate::velocity::record_post_and_maybe_throttle`).
///
/// Appelé par l'API Node à CHAQUE création de tweet, sans exception : un post
/// isolé ne déclenche rien, c'est le rythme qui compte — la décision se prend
/// ici, pas côté Node, pour ne pas dupliquer la fenêtre glissante dans deux
/// systèmes qui pourraient diverger.
pub async fn velocity_post_burst_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<VelocityThrottleRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("VelocityPostBurst: unauthorized — missing or invalid X-Service-Key");
        return (StatusCode::UNAUTHORIZED, Json(json!({ "success": false, "error": "Unauthorized" })));
    }

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }

    let count = state.cache.record_post_and_maybe_throttle(&req.user_id).await;
    let throttled = count >= crate::velocity::BURST_THRESHOLD;
    if throttled {
        info!(user_id = %req.user_id, count, "Rafale de publication détectée — frein posé (1h, ×0.5)");
    }

    (StatusCode::OK, Json(json!({ "success": true, "count": count, "throttled": throttled })))
}
