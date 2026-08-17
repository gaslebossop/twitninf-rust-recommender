//! `POST /embed-tweet` — calcule et stocke l'embedding d'un tweet.
//!
//! Appelé par l'API Node juste après la création réussie d'un tweet
//! (fire-and-forget) — voir `crate::embeddings` pour le pourquoi et le comment.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::warn;

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

#[derive(Debug, Deserialize)]
pub struct EmbedTweetRequest {
    pub tweet_id: String,
    pub content: String,
}

pub async fn embed_tweet_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<EmbedTweetRequest>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("EmbedTweet: unauthorized — missing or invalid X-Service-Key");
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })),
        );
    }

    if uuid::Uuid::parse_str(&req.tweet_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })),
        );
    }

    // Absent plutôt qu'en erreur : un modèle pas encore chargé (ou qui a
    // échoué à charger) ne doit pas faire échouer la publication d'un tweet,
    // qui n'a structurellement rien à voir avec les embeddings.
    let Some(embedder) = state.embeddings.get() else {
        return (
            StatusCode::OK,
            Json(json!({ "success": true, "skipped": "embeddings_disabled" })),
        );
    };

    match crate::embeddings::embed_and_store(&state.pg, embedder, &req.tweet_id, &req.content).await
    {
        Ok(_) => (StatusCode::OK, Json(json!({ "success": true }))),
        Err(e) => {
            warn!(tweet_id = %req.tweet_id, error = %e, "Embedding du tweet échoué");
            (
                StatusCode::OK,
                Json(json!({ "success": false, "error": "embedding failed" })),
            )
        }
    }
}
