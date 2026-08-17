//! `GET /account-status?user_id=…` — ce qu'un compte peut lire sur lui-même.
//!
//! C'est la pièce qui manquait le plus, et pas seulement par correction : une
//! suppression de portée que personne ne peut voir ni dater n'apprend rien à
//! personne. Le compte ne corrige pas ce qu'il ignore, l'équipe de modération ne
//! reçoit que des plaintes vagues, et la restriction devient permanente de fait.
//!
//! TikTok a fait exactement ce chemin en 2023 : son système d'application, jugé
//! opaque par les créateurs, a été doublé d'une page « état du compte » où l'on
//! lit ses avertissements, leur motif et leur date d'expiration. Le point n'est
//! pas d'être gentil — c'est que la sanction ne produit le comportement visé que
//! si elle est lisible.
//!
//! Réservé à l'appel service-à-service : l'API Node est responsable de ne servir
//! à un utilisateur que son propre état.

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::warn;

use crate::handlers::AppState;

#[derive(Debug, Deserialize)]
pub struct AccountStatusQuery {
    pub user_id: String,
}

fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.len() != secret.len() { return false; }
    provided.as_bytes().iter().zip(secret.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub async fn account_status_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Query(q): Query<AccountStatusQuery>,
) -> (StatusCode, Json<Value>) {
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("AccountStatus: unauthorized — missing or invalid X-Service-Key");
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })),
        );
    }

    if uuid::Uuid::parse_str(&q.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }

    let status = state.cache.shadowban_account_status(&q.user_id).await;

    // Le frein de vélocité (`crate::velocity`) vit dans une clé Redis à part,
    // hors du registre d'avertissements — sans ce merge, un compte tout juste
    // freiné se lirait « Clean » ici alors que son score est bien divisé par
    // deux en ce moment même. `velocity_throttled` s'ajoute donc au même JSON,
    // pas dans un champ séparé de la réponse : la page « état du compte »
    // n'a qu'un seul objet à lire.
    let velocity_throttled = state
        .cache
        .load_velocity_throttles(std::slice::from_ref(&q.user_id))
        .await
        .contains_key(&q.user_id);

    let mut data = serde_json::to_value(&status).unwrap_or_else(|_| json!({}));
    if let Value::Object(ref mut map) = data {
        map.insert("velocity_throttled".to_string(), json!(velocity_throttled));
    }

    (StatusCode::OK, Json(json!({ "success": true, "data": data })))
}
