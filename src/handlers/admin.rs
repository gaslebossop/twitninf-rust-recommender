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
    AdminActionResponse, AlgoStatsResponse, AlgoWeights, AlgoWeightsResponse, BackfillCtrRequest,
    BanRequest, FiltersResponse, IssueStrikeRequest, RevokeStrikeRequest, SetShadowbanRequest,
    SetWeightsRequest, UnbanRequest,
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
        let _ = a
            .iter()
            .zip(dummy.iter())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y));
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
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
    let hard_banned = state.cache.admin_get_banned_users().await;

    let total_shadowbanned = shadowbanned.len();
    let total_hard_banned = hard_banned.len();

    info!(
        total_shadowbanned,
        total_hard_banned, "Admin: filters list requested"
    );

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
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }
    if let Err(e) = validate_reason(&req.reason) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e })),
        );
    }

    let level_label = req.level.label();
    state
        .cache
        .admin_set_shadowban(
            &req.user_id,
            req.level,
            req.reason.as_deref(),
            req.expires_in_days,
        )
        .await;

    info!(user_id = %req.user_id, level = level_label,
          expires_in_days = req.expires_in_days, "Admin: shadowban level set");

    let terme = match req.expires_in_days {
        Some(d) if d > 0 => format!(" for {d} day(s)"),
        _ => " with no end date".to_string(),
    };
    (
        StatusCode::OK,
        Json(json!(AdminActionResponse {
            success: true,
            message: format!(
                "User {} shadowban set to {}{}",
                req.user_id, level_label, terme
            ),
        })),
    )
}

// ─── POST /admin/strike ───────────────────────────────────────────────────────

/// Émet un avertissement daté — le chemin normal, à préférer à `/admin/shadowban`.
///
/// La différence tient entièrement à l'expiration : un avertissement disparaît
/// seul au bout de 90 jours et le compte remonte de lui-même, alors qu'un niveau
/// posé à la main reste jusqu'à ce que quelqu'un pense à le retirer. Le niveau
/// n'est pas choisi ici : il se déduit du nombre d'avertissements actifs dans le
/// domaine concerné, avec des seuils propres à chaque domaine.
pub async fn admin_issue_strike_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<IssueStrikeRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }
    if req
        .tweet_id
        .as_deref()
        .is_some_and(|t| uuid::Uuid::parse_str(t).is_err())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })),
        );
    }
    if let Err(e) = validate_reason(&req.reason) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e })),
        );
    }

    let ledger = state
        .cache
        .shadowban_add_strike(
            &req.user_id,
            req.policy,
            req.tweet_id.as_deref(),
            req.reason.as_deref(),
        )
        .await;
    let status = ledger.status(chrono::Utc::now());

    info!(user_id = %req.user_id, policy = req.policy.label(),
          level = status.level_label, active = status.active_strikes,
          "Admin: avertissement émis");

    (
        StatusCode::OK,
        Json(json!({ "success": true, "data": status })),
    )
}

// ─── POST /admin/strike/revoke ────────────────────────────────────────────────

/// Recours accepté : retire les avertissements liés à un tweet, ou tous.
///
/// Retirer l'avertissement fait partie du recours, pas seulement rétablir le
/// contenu : sans cela, gagner son recours ne répare que la moitié du dommage —
/// le post revient mais le compte reste au même palier.
pub async fn admin_revoke_strike_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<RevokeStrikeRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }

    match req.tweet_id.as_deref() {
        Some(tweet_id) => {
            if uuid::Uuid::parse_str(tweet_id).is_err() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })),
                );
            }
            let removed = state
                .cache
                .shadowban_revoke_strikes_for_tweet(&req.user_id, tweet_id)
                .await;
            info!(user_id = %req.user_id, tweet_id, removed, "Admin: recours accepté");
            let status = state.cache.shadowban_account_status(&req.user_id).await;
            (
                StatusCode::OK,
                Json(json!({ "success": true, "removed": removed, "data": status })),
            )
        }
        None => {
            state.cache.shadowban_clear_strikes(&req.user_id).await;
            info!(user_id = %req.user_id, "Admin: registre d'avertissements vidé");
            let status = state.cache.shadowban_account_status(&req.user_id).await;
            (
                StatusCode::OK,
                Json(json!({ "success": true, "data": status })),
            )
        }
    }
}

// ─── POST /admin/ban ──────────────────────────────────────────────────────────

pub async fn admin_ban_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<BanRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }
    if let Err(e) = validate_reason(&req.reason) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e })),
        );
    }

    state
        .cache
        .admin_set_hard_ban(&req.user_id, req.reason.as_deref())
        .await;

    info!(user_id = %req.user_id, "Admin: hard ban applied");

    (
        StatusCode::OK,
        Json(json!(AdminActionResponse {
            success: true,
            message: format!("User {} hard-banned", req.user_id),
        })),
    )
}

// ─── POST /admin/unban ────────────────────────────────────────────────────────

pub async fn admin_unban_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<UnbanRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }

    state.cache.admin_remove_hard_ban(&req.user_id).await;
    info!(user_id = %req.user_id, "Admin: hard ban removed");

    (
        StatusCode::OK,
        Json(json!(AdminActionResponse {
            success: true,
            message: format!("User {} unbanned", req.user_id),
        })),
    )
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

    (
        StatusCode::OK,
        Json(json!(AlgoWeightsResponse {
            weights,
            auto_tuned,
            ctr_samples,
            global_ctr,
        })),
    )
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
        d1_engagement_velocity: req.d1.unwrap_or(current.d1_engagement_velocity),
        d2_content_intelligence: req.d2.unwrap_or(current.d2_content_intelligence),
        d3_social_graph: req.d3.unwrap_or(current.d3_social_graph),
        d4_temporal: req.d4.unwrap_or(current.d4_temporal),
        d5_behavioral: req.d5.unwrap_or(current.d5_behavioral),
        d6_diversity: req.d6.unwrap_or(current.d6_diversity),
        d7_viral: req.d7.unwrap_or(current.d7_viral),
        d8_personalization: req.d8.unwrap_or(current.d8_personalization),
        d9_llm_understanding: req.d9.unwrap_or(current.d9_llm_understanding),
    };

    let arr = new_weights.as_array();

    // Rejeter les NaN/infinis et les valeurs hors plage
    if arr.iter().any(|w| !w.is_finite()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "success": false, "error": "Weights must be finite numbers (no NaN/Inf)" }),
            ),
        );
    }
    if arr.iter().any(|&w| w < 0.0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "All weights must be >= 0.0" })),
        );
    }
    if arr.iter().any(|&w| w > MAX_WEIGHT_VALUE) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "success": false, "error": format!("Each weight must be <= {MAX_WEIGHT_VALUE}") }),
            ),
        );
    }
    let sum: f64 = arr.iter().sum();
    if sum <= 0.0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "Sum of weights must be > 0" })),
        );
    }

    state.cache.admin_save_weights(&new_weights).await;

    info!(
        d1 = new_weights.d1_engagement_velocity,
        d2 = new_weights.d2_content_intelligence,
        "Admin: algo weights overridden manually"
    );

    (
        StatusCode::OK,
        Json(json!({
            "success": true,
            "message": "Weights updated — auto-tuner override active",
            "weights": new_weights,
        })),
    )
}

// ─── POST /admin/algo/weights/reset ──────────────────────────────────────────

pub async fn admin_reset_weights_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    state.cache.admin_clear_weights().await;
    info!("Admin: algo weights reset — auto-tuner re-enabled");

    (
        StatusCode::OK,
        Json(json!({
            "success": true,
            "message": "Manual override cleared — auto-tuner re-enabled",
        })),
    )
}

// ─── GET /admin/algo/stats ────────────────────────────────────────────────────

pub async fn admin_algo_stats_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let (ctr_samples, global_ctr) = state.recommender.ctr_stats();
    let admin_override = state.cache.admin_load_weights().await;
    let auto_tuned = state.auto_tuner.is_auto_tuned() && admin_override.is_none();
    let weights = state.auto_tuner.active_weights(admin_override.as_ref());
    let ml_active = ctr_samples >= 200;
    let (dwell_samples, dwell_mean_weight) = state.recommender.dwell_stats();
    let dwell_active = dwell_samples >= 200;
    let ((amplify_samples, amplify_rate), (reject_samples, reject_rate)) =
        state.recommender.objective_stats();
    let min_objective = crate::ml::objectives::MIN_SAMPLES;

    (
        StatusCode::OK,
        Json(json!(AlgoStatsResponse {
            ctr_samples,
            global_ctr,
            weights,
            auto_tuned,
            ml_active,
            dwell_samples,
            dwell_mean_weight,
            dwell_active,
            amplify_samples,
            amplify_rate,
            amplify_active: amplify_samples >= min_objective,
            reject_samples,
            reject_rate,
            reject_active: reject_samples >= min_objective,
            algorithm_version: "2.3.0 — 9D + multi-objectif (CTR, dwell, amplification, rejet) + bandit + admin node",
        })),
    )
}

// ─── POST /admin/algo/backfill-ctr ─────────────────────────────────────────────
//
// Reconstruit le modèle CTR depuis les interactions réelles des N derniers
// jours (défaut 14) — voir `services::ctr_backfill` pour les approximations
// assumées. `apply: false` par défaut : ne modifie rien, ne fait que rapporter
// ce que donnerait la reconstruction. Appeler UNE FOIS en dry-run, vérifier
// `resulting_global_ctr`/`resulting_weights`, puis seulement avec
// `apply: true` si le résultat semble sain — celui-là sauvegarde l'ancien
// modèle avant de l'écraser, mais un redémarrage du service reste nécessaire
// pour que le nouveau modèle soit effectivement chargé et servi.
pub async fn admin_backfill_ctr_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<BackfillCtrRequest>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let since_days = req.since_days.unwrap_or(14).clamp(1, 90);
    let apply = req.apply.unwrap_or(false);

    info!(since_days, apply, "Backfill CTR demandé");

    match state.recommender.backfill_ctr_model(since_days, apply).await {
        Ok(report) => (StatusCode::OK, Json(json!({ "success": true, "report": report }))),
        Err(e) => {
            warn!(error = %e, "Backfill CTR échoué");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "success": false, "error": e.to_string() })),
            )
        }
    }
}

// ─── GET /admin/logs ──────────────────────────────────────────────────────────

pub async fn admin_logs_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    // Essaie les deux noms de service (nouveau puis ancien)
    let service_names = ["rust-recommender", "twitninf-rust-recommender"];
    let mut raw_logs: Vec<serde_json::Value> = vec![];

    for service in &service_names {
        let output = std::process::Command::new("journalctl")
            .args(&["-u", service, "-n", "100", "--no-pager", "-o", "json"])
            .output();

        if let Ok(out) = output {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let parsed: Vec<serde_json::Value> = stdout
                    .lines()
                    .filter_map(|line| serde_json::from_str(line).ok())
                    .collect();
                if !parsed.is_empty() {
                    raw_logs = parsed;
                    break;
                }
            }
        }
    }

    // Transformer en format lisible: extraire le message depuis la structure journald/tracing
    let logs: Vec<serde_json::Value> = raw_logs
        .into_iter()
        .rev() // plus récents en premier
        .filter_map(|entry| {
            let raw_msg = entry
                .get("MESSAGE")
                .and_then(|m| m.as_str())
                .unwrap_or("")
                .to_string();
            if raw_msg.is_empty() {
                return None;
            }

            // Priorité journald → niveau
            let priority = entry
                .get("PRIORITY")
                .and_then(|p| p.as_str())
                .unwrap_or("6");
            let default_level = match priority {
                "0" | "1" | "2" | "3" => "ERROR",
                "4" | "5" => "WARN",
                "7" => "DEBUG",
                _ => "INFO",
            };

            // Timestamp depuis __REALTIME_TIMESTAMP (microsecondes)
            let timestamp = entry
                .get("__REALTIME_TIMESTAMP")
                .and_then(|t| t.as_str())
                .and_then(|t| t.parse::<u64>().ok())
                .map(|us| {
                    let secs = us / 1_000_000;
                    let ms = (us % 1_000_000) / 1000;
                    format!("{}.{:03}", secs, ms)
                })
                .unwrap_or_else(|| "-".to_string());

            // Tenter de parser le JSON tracing dans MESSAGE
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw_msg) {
                let level = parsed
                    .get("level")
                    .and_then(|l| l.as_str())
                    .unwrap_or(default_level);
                let ts = parsed
                    .get("timestamp")
                    .and_then(|t| t.as_str())
                    .unwrap_or(&timestamp);
                let fields = parsed.get("fields").cloned().unwrap_or(json!({}));
                let main_msg = fields.get("message").and_then(|m| m.as_str()).unwrap_or("");

                // Extraire les champs supplémentaires (tout sauf "message")
                let extras: Vec<String> = fields
                    .as_object()
                    .map(|obj| {
                        obj.iter()
                            .filter(|(k, _)| *k != "message")
                            .map(|(k, v)| format!("{}={}", k, v.as_str().unwrap_or(&v.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();

                let full_msg = if extras.is_empty() {
                    main_msg.to_string()
                } else {
                    format!("{} {}", main_msg, extras.join(" "))
                };

                if full_msg.trim().is_empty() {
                    return None;
                }

                Some(json!({
                    "timestamp": ts,
                    "level": level,
                    "message": full_msg,
                    "target": parsed.get("target").and_then(|t| t.as_str()).unwrap_or(""),
                }))
            } else {
                // Message texte plain
                if raw_msg.contains("Address already in use") {
                    return None;
                } // filtrer le bruit du crash loop
                Some(json!({
                    "timestamp": timestamp,
                    "level": default_level,
                    "message": raw_msg,
                    "target": "",
                }))
            }
        })
        .take(50)
        .collect();

    let count = logs.len();
    if count == 0 {
        return (
            StatusCode::OK,
            Json(json!({
                "logs": [{"timestamp": "-", "level": "WARN", "message": "No logs available", "target": ""}],
                "count": 0,
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "logs": logs,
            "count": logs.len(),
        })),
    )
}

// ─── GET /admin/data ──────────────────────────────────────────────────────────

pub async fn admin_data_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    require_admin!(headers, state);

    let (ctr_samples, global_ctr) = state.recommender.ctr_stats();
    let admin_override = state.cache.admin_load_weights().await;
    let auto_tuned = state.auto_tuner.is_auto_tuned() && admin_override.is_none();
    let weights = state.auto_tuner.active_weights(admin_override.as_ref());
    let ml_active = ctr_samples >= 200;

    // Récupère les listes de bans
    let shadowbanned = state.cache.admin_get_shadowbanned_users().await;
    let hard_banned = state.cache.admin_get_banned_users().await;

    (
        StatusCode::OK,
        Json(json!({
            "algorithm": {
                "ctr_samples": ctr_samples,
                "global_ctr": global_ctr,
                "ml_active": ml_active,
                "auto_tuned": auto_tuned,
                "weights": weights,
            },
            "filters": {
                "shadowbanned_count": shadowbanned.len(),
                "hard_banned_count": hard_banned.len(),
                "shadowbanned": shadowbanned,
                "hard_banned": hard_banned,
            },
            "uptime": state.start_time.elapsed()
                .map(|d| format!("{:.0}s", d.as_secs_f64()))
                .unwrap_or_else(|_| "unknown".to_string()),
        })),
    )
}

// ─── GET /admin/panel ─────────────────────────────────────────────────────────

pub use crate::admin::ui::admin_ui_handler;
