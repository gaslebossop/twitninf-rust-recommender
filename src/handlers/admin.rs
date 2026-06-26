/// Handlers admin — contrôle de l'algo en temps réel.
///
/// Toutes les routes sont protégées par le header `X-Admin-Key`.
/// La clé est configurée via ADMIN_SECRET (obligatoire au démarrage).
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::admin::{
    AdminActionResponse, AlgoStatsResponse, AlgoWeights, AlgoWeightsResponse, BanRequest,
    FiltersResponse, SetShadowbanRequest, SetWeightsRequest, UnbanRequest,
};
use crate::handlers::AppState;

const MAX_REASON_LEN: usize = 500;
const MAX_WEIGHT_VALUE: f64 = 100.0;

// ─── Auth helper (comparaison à temps constant) ────────────────────────────────
//
// On XOR byte-à-byte et on OR les résultats, ce qui évite le short-circuit
// qui permet une attaque timing pour deviner la clé caractère par caractère.

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // On compare quand même pour masquer la longueur, mais on retourne false
        let dummy = b"0000000000000000";
        let _ = a.iter().zip(dummy.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y));
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn check_admin_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Admin-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    constant_time_eq(provided.as_bytes(), secret.as_bytes())
}

macro_rules! require_admin {
    ($headers:expr, $state:expr) => {
        if !check_admin_key(&$headers, &$state.admin_secret) {
            warn!("Admin access denied — invalid or missing X-Admin-Key");
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "success": false, "error": "Unauthorized" })),
            );
        }
    };
}

fn validate_reason(reason: &Option<String>) -> Result<(), &'static str> {
    if let Some(r) = reason {
        if r.len() > MAX_REASON_LEN {
            return Err("reason must be 500 characters or fewer");
        }
    }
    Ok(())
}

// ─── GET /admin/filters ───────────────────────────────────────────────────────

pub async fn admin_filters_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let shadowbanned = state.cache.admin_get_shadowbanned_users().await;
    let hard_banned  = state.cache.admin_get_banned_users().await;

    let total_shadowbanned = shadowbanned.len();
    let total_hard_banned  = hard_banned.len();

    info!(total_shadowbanned, total_hard_banned, "Admin: filters list requested");

    (
        StatusCode::OK,
        Json(json!(FiltersResponse {
            shadowbanned,
            hard_banned,
            total_shadowbanned,
            total_hard_banned,
        })),
    )
}

// ─── POST /admin/shadowban ────────────────────────────────────────────────────

pub async fn admin_set_shadowban_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<SetShadowbanRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }
    if let Err(e) = validate_reason(&req.reason) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e })));
    }

    let level_label = req.level.label();
    state.cache.admin_set_shadowban(&req.user_id, req.level, req.reason.as_deref()).await;

    info!(user_id = %req.user_id, level = level_label, "Admin: shadowban level set");

    (StatusCode::OK, Json(json!(AdminActionResponse {
        success: true,
        message: format!("User {} shadowban set to {}", req.user_id, level_label),
    })))
}

// ─── POST /admin/ban ──────────────────────────────────────────────────────────

pub async fn admin_ban_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<BanRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }
    if let Err(e) = validate_reason(&req.reason) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e })));
    }

    state.cache.admin_set_hard_ban(&req.user_id, req.reason.as_deref()).await;

    info!(user_id = %req.user_id, "Admin: hard ban applied");

    (StatusCode::OK, Json(json!(AdminActionResponse {
        success: true,
        message: format!("User {} hard-banned", req.user_id),
    })))
}

// ─── POST /admin/unban ────────────────────────────────────────────────────────

pub async fn admin_unban_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<UnbanRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }

    state.cache.admin_remove_hard_ban(&req.user_id).await;
    info!(user_id = %req.user_id, "Admin: hard ban removed");

    (StatusCode::OK, Json(json!(AdminActionResponse {
        success: true,
        message: format!("User {} unbanned", req.user_id),
    })))
}

// ─── GET /admin/algo/weights ──────────────────────────────────────────────────

pub async fn admin_get_weights_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let admin_override = state.cache.admin_load_weights().await;
    let (ctr_samples, global_ctr) = state.recommender.ctr_stats();
    let auto_tuned = state.auto_tuner.is_auto_tuned() && admin_override.is_none();
    let weights = state.auto_tuner.active_weights(admin_override.as_ref());

    (StatusCode::OK, Json(json!(AlgoWeightsResponse {
        weights, auto_tuned, ctr_samples, global_ctr,
    })))
}

// ─── POST /admin/algo/weights ─────────────────────────────────────────────────

pub async fn admin_set_weights_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<SetWeightsRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let current = state.cache.admin_load_weights().await.unwrap_or_default();

    let new_weights = AlgoWeights {
        d1_engagement_velocity:  req.d1.unwrap_or(current.d1_engagement_velocity),
        d2_content_intelligence: req.d2.unwrap_or(current.d2_content_intelligence),
        d3_social_graph:         req.d3.unwrap_or(current.d3_social_graph),
        d4_temporal:             req.d4.unwrap_or(current.d4_temporal),
        d5_behavioral:           req.d5.unwrap_or(current.d5_behavioral),
        d6_diversity:            req.d6.unwrap_or(current.d6_diversity),
        d7_viral:                req.d7.unwrap_or(current.d7_viral),
        d8_personalization:      req.d8.unwrap_or(current.d8_personalization),
    };

    let arr = new_weights.as_array();

    // Rejeter les NaN/infinis et les valeurs hors plage
    if arr.iter().any(|w| !w.is_finite()) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "Weights must be finite numbers (no NaN/Inf)" })));
    }
    if arr.iter().any(|&w| w < 0.0) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "All weights must be >= 0.0" })));
    }
    if arr.iter().any(|&w| w > MAX_WEIGHT_VALUE) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": format!("Each weight must be <= {MAX_WEIGHT_VALUE}") })));
    }
    let sum: f64 = arr.iter().sum();
    if sum <= 0.0 {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "Sum of weights must be > 0" })));
    }

    state.cache.admin_save_weights(&new_weights).await;

    info!(
        d1 = new_weights.d1_engagement_velocity,
        d2 = new_weights.d2_content_intelligence,
        "Admin: algo weights overridden manually"
    );

    (StatusCode::OK, Json(json!({
        "success": true,
        "message": "Weights updated — auto-tuner override active",
        "weights": new_weights,
    })))
}

// ─── POST /admin/algo/weights/reset ──────────────────────────────────────────

pub async fn admin_reset_weights_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    state.cache.admin_clear_weights().await;
    info!("Admin: algo weights reset — auto-tuner re-enabled");

    (StatusCode::OK, Json(json!({
        "success": true,
        "message": "Manual override cleared — auto-tuner re-enabled",
    })))
}

// ─── GET /admin/algo/stats ────────────────────────────────────────────────────

pub async fn admin_algo_stats_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let (ctr_samples, global_ctr) = state.recommender.ctr_stats();
    let admin_override = state.cache.admin_load_weights().await;
    let auto_tuned  = state.auto_tuner.is_auto_tuned() && admin_override.is_none();
    let weights     = state.auto_tuner.active_weights(admin_override.as_ref());
    let ml_active   = ctr_samples >= 200;

    (StatusCode::OK, Json(json!(AlgoStatsResponse {
        ctr_samples, global_ctr, weights, auto_tuned, ml_active,
        algorithm_version: "2.1.0 — 8D + ML CTR + bandit + garbage filter + admin node",
    })))
}
