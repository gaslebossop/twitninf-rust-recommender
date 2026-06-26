use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use tracing::error;

use crate::handlers::AppState;
use crate::models::RecommendRequest;

pub async fn recommend_handler(
    State(state): State<AppState>,
    Json(req): Json<RecommendRequest>,
) -> (StatusCode, Json<Value>) {
    // Validation d'entrée : user_id doit être un UUID. On rejette en 400
    // (mauvaise requête) plutôt que 500, et avant tout accès base/cache.
    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "success": false,
                "error": "invalid user_id: must be a UUID"
            })),
        );
    }

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
