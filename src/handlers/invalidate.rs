use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::handlers::AppState;

#[derive(Deserialize)]
pub struct InvalidateRequest {
    pub user_id: String,
}

pub async fn invalidate_handler(
    State(state): State<AppState>,
    Json(req): Json<InvalidateRequest>,
) -> (StatusCode, Json<Value>) {
    state.cache.invalidate_recommendations(&req.user_id).await;
    (StatusCode::OK, Json(json!({ "success": true })))
}
