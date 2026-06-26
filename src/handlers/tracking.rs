use axum::{extract::State, http::{HeaderMap, StatusCode}, Json};
use serde_json::{json, Value};
use tracing::{error, info, warn};

use crate::handlers::AppState;
use crate::models::{TrackInteractionRequest, TrackResponse};

const MAX_DWELL_MS: u32 = 60_000; // 1 minute max — rejeter les valeurs absurdes

fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.len() != secret.len() { return false; }
    provided.as_bytes().iter().zip(secret.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub async fn track_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<TrackInteractionRequest>,
) -> (StatusCode, Json<Value>) {
    // Auth service-to-service obligatoire
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Track: unauthorized — missing or invalid X-Service-Key");
        return (StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })));
    }

    // Validation UUID user_id et tweet_id
    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }
    if uuid::Uuid::parse_str(&req.tweet_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })));
    }

    let weight = req.interaction_type.weight();

    // Plafonner dwell_ms pour éviter l'amplification du score
    let dwell_bonus = req.dwell_ms.map(|ms| {
        let capped = ms.min(MAX_DWELL_MS);
        if capped > 10_000 { 0.5 }
        else if capped > 3_000 { 0.2 }
        else { 0.0 }
    }).unwrap_or(0.0);

    let total_weight = weight + dwell_bonus;

    match state.cache.update_tweet_score(&req.tweet_id, total_weight).await {
        Ok(new_score) => {
            if total_weight > 0.0 {
                state.cache.mark_tweet_seen(&req.user_id, &req.tweet_id).await;
            }
            state.cache.invalidate_recommendations(&req.user_id).await;

            info!(
                user_id = %req.user_id,
                tweet_id = %req.tweet_id,
                interaction = ?req.interaction_type,
                weight = total_weight,
                new_score,
                "Interaction tracked"
            );

            (StatusCode::OK, Json(json!(TrackResponse {
                success: true,
                tweet_id: req.tweet_id,
                user_id: req.user_id,
                new_score,
                weight_applied: total_weight,
            })))
        }
        Err(e) => {
            error!("Track error: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "success": false, "error": "Internal server error" })))
        }
    }
}
