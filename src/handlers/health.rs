use axum::{extract::State, http::StatusCode, Json};
use std::time::SystemTime;
use tracing::warn;

use crate::handlers::AppState;
use crate::models::HealthResponse;

pub async fn health_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let db_status = match state.pg.get().await {
        Ok(client) => {
            if client.query_one("SELECT 1", &[]).await.is_ok() {
                "ok".to_string()
            } else {
                "error".to_string()
            }
        }
        Err(e) => {
            warn!("DB health check failed: {}", e);
            "error".to_string()
        }
    };

    let redis_status = if state.cache.health_check().await {
        "ok".to_string()
    } else {
        "error".to_string()
    };

    let uptime = SystemTime::now()
        .duration_since(state.start_time)
        .unwrap_or_default()
        .as_secs();

    let overall = if db_status == "ok" && redis_status == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        overall,
        Json(HealthResponse {
            status: if overall == StatusCode::OK {
                "healthy".into()
            } else {
                "degraded".into()
            },
            version: env!("CARGO_PKG_VERSION"),
            db: db_status,
            redis: redis_status,
            uptime_secs: uptime,
            algorithm: "NeuralRank Fusion v2.4 - 10 dimensions + ML multi-objectif + bandit + filtre poubelle",
        }),
    )
}
