/// Rattrapage hors-ligne du modèle CTR.
///
/// Reconstruit un modèle NEUF à partir des interactions réelles des derniers
/// jours, plutôt que de repartir d'un modèle vide (perte du signal accumulé)
/// ou de garder le modèle actuel (qui a pu apprendre sur une période jugée
/// non représentative — voir la discussion sur D6 dans la passation).
///
/// ── Deux approximations assumées, aucune des deux évitable ──────────────────
///
/// 1. Aucune trace durable des features vues au moment de chaque impression
///    passée n'existe : elles vivent en Redis, éphémères (voir
///    `CacheManager::record_impression`, fenêtre d'attribution de 30 min).
///    Chaque paire (lecteur, tweet) est donc recalculée avec l'état ACTUEL du
///    tweet et du profil — pas celui du moment de l'interaction. Un tweet qui
///    a beaucoup grossi depuis, ou un profil qui a beaucoup changé, biaise la
///    reconstruction dans le sens de « aujourd'hui », pas « alors ».
/// 2. Pas de journal « montré, jamais engagé » côté base — les négatifs
///    historiques n'existent pas plus que les features. Pour chaque lecteur,
///    on échantillonne des tweets de la même fenêtre qu'il n'a PAS engagés :
///    pratique standard de ré-entraînement hors ligne (« negative sampling »)
///    quand le vrai journal d'impressions n'existe pas. Un tweet non engagé
///    n'a pas forcément été VU — c'est plus bruité qu'un vrai négatif observé,
///    dans le même sens que ce que le classement lui-même produirait (un tweet
///    jamais montré à ce lecteur n'aurait de toute façon pas eu de chance
///    d'être engagé).
use std::collections::{HashMap, HashSet};

use anyhow::Result;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tracing::{info, warn};

use crate::admin::AlgoWeights;
use crate::algorithm::scoring::{ctr_feature_vector, score_tweet_with_weights};
use crate::ml::ctr_predictor::{CtrModel, N_FEATURES};
use crate::models::RawTweet;
use crate::services::recommender::RecommenderService;

/// Négatifs échantillonnés par lecteur, en multiple de son nombre de positifs.
const NEGATIVES_PER_USER_RATIO: usize = 2;
/// Un lecteur sans engagement dans la fenêtre n'apporte aucun signal fiable —
/// et gonflerait le pool de négatifs pour rien.
const MIN_POSITIVES_PER_USER: usize = 1;
/// Borne du pool de négatifs potentiels : au-delà, la requête n'apporte plus
/// de diversité supplémentaire au tirage, seulement de la latence.
const NEGATIVE_POOL_LIMIT: i64 = 5000;

#[derive(Debug, serde::Serialize)]
pub struct BackfillReport {
    pub since_days: i32,
    pub distinct_users: usize,
    pub positives_found: usize,
    pub negatives_sampled: usize,
    /// Vues réellement journalisées sur la fenêtre (`user_behavior_data`) —
    /// le vrai dénominateur de `resulting_global_ctr`, pas la taille du
    /// tirage équilibré utilisé pour entraîner les poids.
    pub real_views: i64,
    pub samples_trained: usize,
    pub resulting_global_ctr: f64,
    pub resulting_weights: [f64; N_FEATURES],
    pub applied: bool,
    pub backup_path: Option<String>,
}

impl RecommenderService {
    /// `apply = false` : reconstruit et rapporte, n'écrit rien — c'est le mode
    /// à utiliser en premier pour juger le résultat avant d'y toucher.
    /// `apply = true` : sauvegarde `data/ctr_model.json` existant puis
    /// l'écrase. Le service en mémoire garde l'ancien modèle jusqu'au
    /// redémarrage — appliquer ne bascule rien tout seul.
    pub async fn backfill_ctr_model(&self, since_days: i32, apply: bool) -> Result<BackfillReport> {
        let client = self.pg().get().await?;
        // `i32`, pas `i64` : `make_interval(days => ...)` attend un `int4`
        // Postgres — un `int8` fait échouer la sérialisation du paramètre
        // avant même que la requête ne parte.

        // Positifs : like + retweet, regroupés par (lecteur, tweet) — un CTR
        // compte un engagement, pas chaque type d'engagement séparément.
        let rows = client
            .query(
                "SELECT user_id::text, tweet_id::text FROM ( \
                    SELECT user_id, tweet_id FROM tweet_likes WHERE created_at > NOW() - make_interval(days => $1) \
                    UNION \
                    SELECT user_id, tweet_id FROM tweet_retweets WHERE created_at > NOW() - make_interval(days => $1) \
                 ) e",
                &[&since_days],
            )
            .await?;

        let mut positives: HashMap<String, HashSet<String>> = HashMap::new();
        for row in &rows {
            let user_id: String = row.get(0);
            let tweet_id: String = row.get(1);
            positives.entry(user_id).or_default().insert(tweet_id);
        }
        let positives_found: usize = positives.values().map(|s| s.len()).sum();

        // Pool de négatifs potentiels : tout tweet public de la même fenêtre,
        // même filtre que la sélection de candidats en direct (voir
        // `CANDIDATES_CTE`) — pas de compte suspendu, pas de contenu de test.
        let pool_rows = client
            .query(
                "SELECT t.id::text FROM tweets t JOIN users u ON u.id = t.user_id \
                 WHERE t.deleted_at IS NULL AND t.moderation_status = 'approved' AND t.is_private = false \
                   AND COALESCE(t.is_data_test, false) = false \
                   AND u.is_active = true AND COALESCE(u.is_suspended, false) = false \
                   AND t.created_at > NOW() - make_interval(days => $1) \
                 LIMIT $2",
                &[&since_days, &NEGATIVE_POOL_LIMIT],
            )
            .await?;
        let pool: Vec<String> = pool_rows.iter().map(|r| r.get(0)).collect();

        let mut model = CtrModel::default();
        // `StdRng`, pas `thread_rng()` : ce dernier n'est pas `Send`
        // (`Rc` interne), et cette boucle traverse plusieurs `.await` par
        // itération — un handler Axum doit rester `Send` de bout en bout.
        let mut rng = StdRng::from_entropy();
        let mut negatives_sampled = 0usize;
        let mut samples_trained = 0usize;
        let mut users_used = 0usize;

        for (user_id, liked) in &positives {
            if liked.len() < MIN_POSITIVES_PER_USER || pool.is_empty() {
                continue;
            }

            let profile = match self.build_user_profile(user_id).await {
                Ok(p) => p,
                Err(e) => {
                    warn!(user_id, error = %e, "Backfill CTR: profil illisible, lecteur ignoré");
                    continue;
                }
            };

            let target_negatives = (liked.len() * NEGATIVES_PER_USER_RATIO).max(1);
            let negatives: Vec<String> = pool
                .choose_multiple(&mut rng, (target_negatives + liked.len()).min(pool.len()))
                .filter(|id| !liked.contains(*id))
                .take(target_negatives)
                .cloned()
                .collect();
            negatives_sampled += negatives.len();

            let mut ids: Vec<String> = liked.iter().cloned().collect();
            ids.extend(negatives.iter().cloned());

            let tweets = match self.hydrate_tweets_for_user(user_id, &ids).await {
                Ok(t) => t,
                Err(e) => {
                    warn!(user_id, error = %e, "Backfill CTR: tweets illisibles, lecteur ignoré");
                    continue;
                }
            };
            if tweets.is_empty() {
                continue;
            }
            let by_id: HashMap<&str, &RawTweet> =
                tweets.iter().map(|t| (t.id.as_str(), t)).collect();

            let mut trained_for_user = 0usize;
            for tweet_id in liked.iter().chain(negatives.iter()) {
                let Some(tweet) = by_id.get(tweet_id.as_str()) else {
                    continue;
                };
                let clicked = liked.contains(tweet_id);
                let scored =
                    score_tweet_with_weights(
                        tweet,
                        &profile,
                        0,
                        // Reconstruction hors ligne d'une interaction passee :
                        // il n'y a pas de fil autour de ce tweet, D6 le voit
                        // donc comme une ouverture de page.
                        crate::algorithm::scoring::FeedShape::empty(),
                        &AlgoWeights::default(),
                    );
                let features = ctr_feature_vector(tweet, &profile, &scored);
                model.update(&features, clicked);
                samples_trained += 1;
                trained_for_user += 1;
            }
            if trained_for_user > 0 {
                users_used += 1;
            }
        }

        // `model.global_ctr()` sortirait ici le ratio du plan d'échantillonnage
        // (`NEGATIVES_PER_USER_RATIO`), pas un taux réel — exactement le piège
        // que l'auto-tuner documente déjà pour l'étiquetage dégénéré, en pire :
        // ici c'est un artefact CONNU du tirage, pas une anomalie de données.
        // On recalcule le vrai ratio depuis les vues effectivement journalisées
        // sur la même fenêtre, et ce sont CES compteurs qu'on écrit dans le
        // modèle — les poids appris, eux, restent ceux du tirage équilibré
        // (nécessaire pour que le gradient ait un signal négatif exploitable).
        let real_views: i64 = client
            .query_one(
                "SELECT COUNT(DISTINCT (user_id, target_id)) FROM user_behavior_data \
                 WHERE action_type = 'tweet_view' AND target_type = 'tweet' \
                   AND COALESCE(is_data_test, false) = false \
                   AND timestamp > NOW() - make_interval(days => $1)",
                &[&since_days],
            )
            .await
            .map(|r| r.get(0))
            .unwrap_or(0);
        // Le suivi de vue a des trous connus par plateforme (voir la
        // passation volumétrie — le web n'émet quasi aucun événement) : borné
        // pour ne jamais tomber sous le nombre de positifs déjà comptés, sinon
        // le ratio dépasserait 1.0, ce qui n'a pas de sens pour un taux de clic.
        model.total_views = real_views.max(positives_found as i64) as u64;
        model.total_clicks = positives_found as u64;

        let resulting_global_ctr = model.global_ctr();
        let resulting_weights = model.weights;
        let mut backup_path = None;

        if apply {
            const LIVE_PATH: &str = "data/ctr_model.json";
            if tokio::fs::metadata(LIVE_PATH).await.is_ok() {
                let backup = format!(
                    "data/ctr_model.backup-{}.json",
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                );
                tokio::fs::copy(LIVE_PATH, &backup).await?;
                info!(backup, "Modèle CTR actuel sauvegardé avant remplacement");
                backup_path = Some(backup);
            }
            let json = serde_json::to_string_pretty(&model)?;
            tokio::fs::create_dir_all("data").await.ok();
            tokio::fs::write(LIVE_PATH, json).await?;
            info!(
                samples_trained,
                resulting_global_ctr,
                "Backfill CTR appliqué sur disque — redémarrage du service requis pour le charger"
            );
        } else {
            info!(
                samples_trained,
                resulting_global_ctr, "Backfill CTR — dry-run, rien d'écrit sur disque"
            );
        }

        Ok(BackfillReport {
            since_days,
            distinct_users: users_used,
            positives_found,
            negatives_sampled,
            real_views,
            samples_trained,
            resulting_global_ctr,
            resulting_weights,
            applied: apply,
            backup_path,
        })
    }
}
