use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::algorithm::dwell::{dwell_weight, DwellContext};
use crate::experiments;
use crate::handlers::AppState;
use crate::models::{TrackInteractionRequest, TrackResponse};

const MAX_DWELL_MS: u32 = 60_000; // 1 minute max — rejeter les valeurs absurdes

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

pub async fn track_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<TrackInteractionRequest>,
) -> (StatusCode, Json<Value>) {
    // Auth service-to-service obligatoire
    if !check_service_key(&headers, &state.internal_secret) {
        warn!("Track: unauthorized — missing or invalid X-Service-Key");
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "success": false, "error": "Unauthorized" })),
        );
    }

    // Validation UUID user_id et tweet_id
    if uuid::Uuid::parse_str(&req.user_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "user_id must be a valid UUID" })),
        );
    }
    if uuid::Uuid::parse_str(&req.tweet_id).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "tweet_id must be a valid UUID" })),
        );
    }
    if req
        .experiment_id
        .as_deref()
        .is_some_and(|id| uuid::Uuid::parse_str(id).is_err())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "experiment_id must be a valid UUID" })),
        );
    }
    if req
        .variant_id
        .as_deref()
        .is_some_and(|id| uuid::Uuid::parse_str(id).is_err())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": "variant_id must be a valid UUID" })),
        );
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
    let dwell_bonus = req
        .dwell_ms
        .map(|ms| dwell_weight(ms.min(MAX_DWELL_MS), dwell_context.as_ref()))
        .unwrap_or(0.0);

    // Entraîne le modèle qui PRÉDIT le dwell (ml::dwell_predictor), sur le
    // poids déjà normalisé ci-dessus — jamais la durée brute, qui porterait
    // le biais de longueur documenté dans `algorithm::dwell`. `peek_impression`
    // et non `take_impression` : une vue ne doit pas retirer l'impression de
    // la file d'attribution CTR, sous peine de priver le balayage
    // (`ctr_sweeper`) des négatifs qu'il en tire quand rien d'autre ne suit.
    if req.dwell_ms.is_some() && dwell_context.is_some() {
        if let Some(features) = state.cache.peek_impression(&req.user_id, &req.tweet_id).await {
            state.recommender.record_dwell_event(&features, dwell_bonus);
        }
    }

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
    )
    .await
    {
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

    if total_weight > 0.0 {
        state
            .cache
            .mark_tweet_seen(&req.user_id, &req.tweet_id)
            .await;
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

    // ── Refus explicites : la réponse doit se VOIR dans le fil ──────
    //
    // Trois gestes disent « je n'en veux pas », de plus en plus fort :
    // « ça ne m'intéresse pas », un signalement, un blocage. Chacun
    // marque le tweet vu — il ne peut donc plus revenir — et met
    // l'AUTEUR en sourdine pour ce lecteur, avec un poids qui dépend du
    // geste (voir `InteractionType::refusal_strikes`).
    //
    // Ce qui a changé : seul « ça ne m'intéresse pas » agissait, alors
    // que c'est le PLUS FAIBLE des trois. Un signalement et un blocage
    // ne marquaient même pas le tweet vu — leur poids est négatif (-12
    // et -20), donc ils tombaient sous le `total_weight > 0.0` du
    // marquage plus haut, et aucune branche explicite ne les
    // rattrapait. Un tweet signalé pouvait revenir dans le fil de la
    // personne qui venait de le signaler.
    //
    // Le blocage passe par ailleurs par `user_follows.status = 'blocked'`
    // côté API, ce qui l'exclut du vivier via `UserProfile::blocked_ids`.
    // Mais cette écriture appartient à une autre couche, sur un autre
    // chemin : le classement ne doit pas dépendre d'elle pour honorer un
    // geste que le lecteur vient de faire ici.
    //
    // Porter le refus sur l'AUTEUR et pas seulement sur le tweet est le
    // point décisif — mesuré par Mozilla sur YouTube : le bouton « pas
    // intéressé » n'évite que ~11 % des recommandations non voulues,
    // quand « ne plus recommander cette chaîne » en évite 43 %. Écarter
    // un tweet parmi les mille du même compte ne change rien de
    // perceptible, et l'utilisateur en conclut — à raison — que le
    // bouton est décoratif.
    if let Some(strikes) = req.interaction_type.refusal_strikes() {
        state
            .cache
            .mark_tweet_seen(&req.user_id, &req.tweet_id)
            .await;
        match req.author_id.as_deref() {
            Some(author_id) if uuid::Uuid::parse_str(author_id).is_ok() => {
                state
                    .cache
                    .damp_author_by(&req.user_id, author_id, strikes)
                    .await;
                info!(user_id = %req.user_id, author_id, strikes,
                      interaction = ?req.interaction_type,
                      "Auteur mis en sourdine par un refus explicite");
            }
            // Sans `author_id` (client ancien) seul le marquage s'applique,
            // ce qui redonne exactement le bouton décoratif qu'on veut
            // éviter : à surveiller si le geste paraît sans effet.
            _ => warn!(user_id = %req.user_id, tweet_id = %req.tweet_id,
                       interaction = ?req.interaction_type,
                       "Refus sans author_id : il ne portera que sur ce tweet"),
        }
    }

    // ── « Skip » marque vu, sans viser l'auteur ──────────────────────
    // Le menu du fil promet « il n'apparaîtra plus » sur ce geste, tenu
    // côté client par un retrait local de la ligne — mais sans ceci le
    // moteur ne l'a jamais marqué vu (Skip pèse -0.5, sous le seuil de
    // `total_weight > 0.0` plus haut) et le tweet revient à la session
    // suivante. Volontairement SANS mise en sourdine de l'auteur,
    // contrairement aux trois refus ci-dessus : ignorer UN tweet ne dit
    // rien de son auteur.
    if req.interaction_type == crate::models::InteractionType::Skip {
        state
            .cache
            .mark_tweet_seen(&req.user_id, &req.tweet_id)
            .await;
    }

    // Alimente le modèle CTR avec le vecteur de features réellement
    // utilisé pour classer ce tweet dans le feed de ce lecteur. On ne
    // l'invente plus ici : on le relit depuis l'impression mémorisée.
    // Pas d'impression retrouvée (feed servi avant ce correctif, TTL
    // dépassé, ou interaction hors feed) → on n'entraîne pas, plutôt
    // que d'entraîner sur des valeurs fabriquées.
    match req.interaction_type.ctr_label() {
        Some(clicked) => {
            match state
                .cache
                .take_impression(&req.user_id, &req.tweet_id)
                .await
            {
                Some(features) => {
                    state.recommender.record_ctr_event(&features, clicked);
                    // Mêmes features, autres lectures : les têtes
                    // multi-objectifs apprennent de CE geste précis, pas du
                    // booléen « engagé / pas engagé » qui les confondrait
                    // toutes. Voir `ml::objectives`.
                    state
                        .recommender
                        .record_objective_event(&features, req.interaction_type);
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

    // ── Boost temps réel (30 min), à l'intérieur de la session ───────
    // Même `ctr_label()` que l'entraînement CTR ci-dessus : un signal
    // qui vaut pour apprendre un poids global vaut aussi pour ajuster
    // ce lecteur-là, tout de suite, sur sa page suivante — voir
    // `services::feedback_loop`.
    if let Some(clicked) = req.interaction_type.ctr_label() {
        if let Some(author_id) = req.author_id.as_deref() {
            if uuid::Uuid::parse_str(author_id).is_ok() {
                state
                    .cache
                    .record_author_feedback(&req.user_id, author_id, clicked)
                    .await;
                // Bras du bandit d'exploration — voir `bandit::contextual`.
                // Même signal, même auteur : ça vaut la peine de dupliquer
                // l'écriture plutôt que de faire dépendre le bandit d'une clé
                // pensée pour un usage différent (30 min, par lecteur) qui
                // pourrait changer de forme sans prévenir.
                state.cache.record_arm_reward(author_id, clicked).await;
            }
        }
    }

    // ── Boucle auteur alimentée par le dwell (VUE, pas de clic) ────────
    // `ctr_label()` renvoie `None` sur une `View` : le bloc juste au-dessus
    // ne se déclenche donc JAMAIS pour un tweet longuement lu mais jamais
    // cliqué — tout le profil de goût restait bâti sur les seuls likes. On
    // rejoue ici le même geste que `record_author_feedback` ci-dessus, mais
    // décidé sur `dwell_bonus` plutôt que sur `ctr_label()`, et seulement
    // quand le signal est FRANC dans un sens ou dans l'autre — la zone
    // ambiguë (lu partiellement, ni franchement consommé ni franchement
    // survolé) ne doit rien émettre : un boost/damping mal fondé abîmerait
    // la page suivante pour rien. Volontairement SANS bandit ni
    // entraînement CTR ici : effet immédiat sur le fil, sans toucher au
    // modèle — voir §3.7 de la passation.
    const FIRM_POSITIVE_DWELL_BONUS: f64 = 0.3; // ~consommation complète ou plus
    const FIRM_NEGATIVE_DWELL_BONUS: f64 = 0.0; // dwell_weight est déjà dans sa branche « survol »
    if req.interaction_type == crate::models::InteractionType::View && dwell_context.is_some() {
        if let Some(author_id) = req.author_id.as_deref() {
            if uuid::Uuid::parse_str(author_id).is_ok() {
                if dwell_bonus > FIRM_POSITIVE_DWELL_BONUS {
                    state
                        .cache
                        .record_author_feedback(&req.user_id, author_id, true)
                        .await;
                } else if dwell_bonus < FIRM_NEGATIVE_DWELL_BONUS {
                    state
                        .cache
                        .record_author_feedback(&req.user_id, author_id, false)
                        .await;
                }
            }
        }
    }

    // ── Filtrage collaboratif léger ────────────────────────────────────
    // Uniquement sur un LIKE, pas sur tout signal positif : c'est le geste
    // le plus intentionnel, celui qui ressemble le moins à un survol. Voir
    // `crate::cooccurrence`.
    if req.interaction_type == crate::models::InteractionType::Like {
        if let Some(author_id) = req.author_id.as_deref() {
            if uuid::Uuid::parse_str(author_id).is_ok() {
                state
                    .cache
                    .record_like_cooccurrence(&req.user_id, author_id)
                    .await;
            }
        }
    }

    info!(
        user_id = %req.user_id,
        tweet_id = %req.tweet_id,
        interaction = ?req.interaction_type,
        weight = total_weight,
        "Interaction tracked"
    );

    (
        StatusCode::OK,
        Json(json!(TrackResponse {
            success: true,
            tweet_id: req.tweet_id,
            user_id: req.user_id,
            weight_applied: total_weight,
        })),
    )
}
