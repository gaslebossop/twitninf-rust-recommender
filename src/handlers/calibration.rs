//! `POST /calibration/round` et `POST /calibration/finish` — recalibration
//! explicite de l'algorithme, voir `crate::calibration` pour le mécanisme.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde_json::{json, Value};
use tracing::warn;

use crate::calibration::{self, CalibrationFinishRequest, CalibrationRoundRequest, ROUNDS};
use crate::handlers::AppState;

fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
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

fn valid_round(round: u8) -> bool {
    (1..=ROUNDS).contains(&round)
}

pub async fn calibration_round_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<CalibrationRoundRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Calibration: unauthorized — missing or invalid X-Service-Key");
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
    if !valid_round(req.round) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "success": false, "error": format!("round must be between 1 and {ROUNDS}") }),
            ),
        );
    }

    match calibration::round_candidates(
        &state.pg,
        &req.user_id,
        req.round,
        &req.liked_tweet_ids,
        &req.skipped_tweet_ids,
    )
    .await
    {
        Ok(tweet_ids) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "data": { "round": req.round, "tweet_ids": tweet_ids }
            })),
        ),
        Err(e) => {
            warn!(user_id = %req.user_id, round = req.round, error = %e, "Calibration: sélection du tour échouée");
            (
                StatusCode::OK,
                Json(json!({ "success": true, "data": { "round": req.round, "tweet_ids": [] } })),
            )
        }
    }
}

pub async fn calibration_finish_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<CalibrationFinishRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Calibration: unauthorized — missing or invalid X-Service-Key");
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

    match calibration::finish(&state.pg, &state.cache, &req.user_id, &req.liked_tweet_ids).await {
        Ok(applied) => (
            StatusCode::OK,
            Json(json!({ "success": true, "data": { "applied": applied } })),
        ),
        Err(e) => {
            warn!(user_id = %req.user_id, error = %e, "Calibration: finalisation échouée");
            (
                StatusCode::OK,
                Json(json!({ "success": false, "error": "calibration finish failed" })),
            )
        }
    }
}
