use axum::{extract::State, http::{HeaderMap, StatusCode}, Json};
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::algorithm::dwell::{dwell_weight, DwellContext};
use crate::handlers::AppState;
use crate::experiments;
use crate::models::{TrackInteractionRequest, TrackResponse};

const MAX_DWELL_MS: u32 = 60_000; // 1 minute max — rejeter les valeurs absurdes

fn check_service_key(headers: &HeaderMap, secret: &str) -> bool {
    let provided = headers
        .get("X-Service-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.len() != secret.len() { return false; }
    provided.as_bytes().iter().zip(secret.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub async fn track_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<TrackInteractionRequest>,
) -> (StatusCode, Json<Value>) {
    // Auth service-to-service obligatoire
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Track: unauthorized — missing or invalid X-Service-Key");
        return (StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })));
    }

    // Validation UUID user_id et tweet_id
    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })));
    }
    if uuid::Uuid::parse_str(&req.tweet_id).is_err() {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })));
    }
    if req.experiment_id.as_deref().is_some_and(|id| uuid::Uuid::parse_str(id).is_err()) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "experiment_id must be a valid UUID" })));
    }
    if req.variant_id.as_deref().is_some_and(|id| uuid::Uuid::parse_str(id).is_err()) {
        return (StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "variant_id must be a valid UUID" })));
    }

    let weight = req.interaction_type.weight();

    // Le temps passé est rapporté au temps que CE contenu demandait, pas jugé
    // à l'aveugle sur sa valeur brute : sans ça le classement apprend que le
    // public préfère les contenus longs, ce qui est une propriété du
    // chronomètre et pas du public. Voir `algorithm::dwell`.
    //
    // Le plafond reste en place en amont : il ne corrige pas un biais, il
    // écarte les valeurs absurdes (téléphone laissé allumé sur un tweet).
    let dwell_context = req.dwell_media.map(|media| DwellContext {
        media,
        content_chars: req.content_chars.unwrap_or(0),
        video_duration_ms: req.video_duration_ms,
    });
    let dwell_bonus = req.dwell_ms
        .map(|ms| dwell_weight(ms.min(MAX_DWELL_MS), dwell_context.as_ref()))
        .unwrap_or(0.0);

    let total_weight = weight + dwell_bonus;

    // Le suivi A/B est persistant en PostgreSQL et ne dépend pas de Redis.
    // Une panne du cache ne doit donc pas faire perdre l'échantillon.
    match experiments::record_interaction(
        &state.pg,
        &req.user_id,
        &req.tweet_id,
        req.variant_id.as_deref(),
        total_weight,
        req.interaction_type == crate::models::InteractionType::View,
    ).await {
        Ok(Some(winner)) => {
            info!(
                experiment_id = %winner.experiment_id,
                variant_id = %winner.variant_id,
                "A/B winner promoted"
            );
        }
        Ok(None) => {}
        Err(error) => {
            warn!(
                tweet_id = %req.tweet_id,
                error = ?error,
                "A/B tracking failed without blocking NeuralRank tracking"
            );
        }
    }

    match state.cache.update_tweet_score(&req.tweet_id, total_weight).await {
        Ok(new_score) => {
            if total_weight > 0.0 {
                state.cache.mark_tweet_seen(&req.user_id, &req.tweet_id).await;
            }

            // ── Une VUE n'invalide pas le classement ────────────────────────
            // Toute interaction jetait le classement caché du lecteur. Tant que
            // seules les actions délibérées (like, retweet, commentaire…)
            // étaient suivies, ça restait rare. Une vue, elle, arrive à CHAQUE
            // tweet regardé : invalider dessus rendrait le cache inutile
            // précisément pour les lecteurs les plus actifs, et surtout ferait
            // recalculer le classement ENTRE deux pages d'une même session de
            // lecture — la pagination avancerait alors dans un classement qui
            // n'est plus celui d'où venait la page précédente, en sautant des
            // tweets et en en resservant d'autres.
            //
            // La vue n'est pas perdue pour autant : `mark_tweet_seen` ci-dessus
            // et l'impression mémorisée la font peser sur le prochain recalcul
            // naturel (expiration du TTL ou `force_refresh`).
            if req.interaction_type != crate::models::InteractionType::View {
                state.cache.invalidate_recommendations(&req.user_id).await;
            }

            // ── « Ça ne m'intéresse pas » : la réponse doit se voir ──────────
            // Deux effets, parce qu'un seul ne suffit pas :
            //   * le tweet est marqué vu, donc il ne peut plus revenir ;
            //   * l'AUTEUR est mis en sourdine pour ce lecteur (voir
            //     `author_damping`) — c'est le seul niveau où le refus produit
            //     un changement perceptible dans le fil, mesures de Mozilla sur
            //     YouTube à l'appui.
            // Sans `author_id` (client ancien) seul le premier effet s'applique,
            // ce qui redonne exactement le bouton décoratif qu'on veut éviter :
            // à surveiller si le geste paraît sans effet.
            if req.interaction_type == crate::models::InteractionType::NotInterested {
                state.cache.mark_tweet_seen(&req.user_id, &req.tweet_id).await;
                match req.author_id.as_deref() {
                    Some(author_id) if uuid::Uuid::parse_str(author_id).is_ok() => {
                        state.cache.damp_author(&req.user_id, author_id).await;
                        info!(user_id = %req.user_id, author_id, "Author damped by explicit disinterest");
                    }
                    _ => warn!(user_id = %req.user_id, tweet_id = %req.tweet_id,
                               "NotInterested sans author_id : le refus ne portera que sur ce tweet"),
                }
            }

            // Alimente le modèle CTR avec le vecteur de features réellement
            // utilisé pour classer ce tweet dans le feed de ce lecteur. On ne
            // l'invente plus ici : on le relit depuis l'impression mémorisée.
            // Pas d'impression retrouvée (feed servi avant ce correctif, TTL
            // dépassé, ou interaction hors feed) → on n'entraîne pas, plutôt
            // que d'entraîner sur des valeurs fabriquées.
            match req.interaction_type.ctr_label() {
                Some(clicked) => {
                    match state.cache.take_impression(&req.user_id, &req.tweet_id).await {
                        Some(features) => {
                            state.recommender.record_ctr_event(&features, clicked);
                        }
                        None => {
                            debug!(user_id = %req.user_id, tweet_id = %req.tweet_id,
                                   "CTR: impression absente, entraînement ignoré");
                        }
                    }
                }
                None => {
                    // Vue : l'impression a déjà été mémorisée au moment où le
                    // feed a été servi. Rien à faire, la fenêtre d'attribution
                    // court.
                }
            }

            info!(
                user_id = %req.user_id,
                tweet_id = %req.tweet_id,
                interaction = ?req.interaction_type,
                weight = total_weight,
                new_score,
                "Interaction tracked"
            );

            (StatusCode::OK, Json(json!(TrackResponse {
                success: true,
                tweet_id: req.tweet_id,
                user_id: req.user_id,
                new_score,
                weight_applied: total_weight,
            })))
        }
        Err(e) => {
            error!("Track error: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "success": false, "error": "Internal server error" })))
        }
    }
}
