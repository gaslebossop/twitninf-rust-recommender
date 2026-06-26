use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use tracing::error;

use crate::handlers::AppState;
use crate::models::RecommendRequest;

pub async fn recommend_handler(
    State(state): State<AppState>,
    Json(req): Json<RecommendRequest>,
) -> (StatusCode, Json<Value>) {
    match state.recommender.recommend(&req).await {
        Ok(resp) => (StatusCode::OK, Json(json!(resp))),
        Err(e) => {
            error!("Recommendation error for user {}: {}", req.user_id, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "success": false,
                    "error": e.to_string()
                })),
            )
        }
    }
}
