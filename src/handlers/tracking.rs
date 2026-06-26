use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use tracing::{error, info};

use crate::handlers::AppState;
use crate::models::{TrackInteractionRequest, TrackResponse};

pub async fn track_handler(
    State(state): State<AppState>,
    Json(req): Json<TrackInteractionRequest>,
) -> (StatusCode, Json<Value>) {
    let weight = req.interaction_type.weight();

    let dwell_bonus = req.dwell_ms.map(|ms| {
        if ms > 10_000 { 0.5 }
        else if ms > 3_000 { 0.2 }
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
                new_score = new_score,
                "Interaction tracked"
            );

            (
                StatusCode::OK,
                Json(json!(TrackResponse {
                    success: true,
                    tweet_id: req.tweet_id,
                    user_id: req.user_id,
                    new_score,
                    weight_applied: total_weight,
                })),
            )
        }
        Err(e) => {
            error!("Track error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "success": false, "error": e.to_string() })),
            )
        }
    }
}
