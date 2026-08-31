use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use rand::Rng;
use tokio::join;
use tracing::{debug, info, trace, warn};

use crate::algorithm::scoring::{
    compute_feed_metrics, impression_fatigue, score_tweet_ml_with_weights, BlendProfile,
    ScoringContext,
    theme_diversity_multiplier, FeedShape,
};
use crate::bandit::bandit_select;
use crate::constants::{
    COLD_START_FOLLOW_BOOST_MAX, COLD_START_INTERACTION_FLOOR, EXCLUDE_SEEN_MIN_REMAINING,
    FOLLOW_FEED_BOOST, FOLLOW_MUTUAL_BOOST, TRENDING_HOOK_POOL, TRENDING_HOOK_SIZE,
    TRENDING_HOOK_TEMPERATURE, TRENDING_MEDIA_BOOST, TRENDING_MIN_POOL,
    TRENDING_SHUFFLE_TEMPERATURE, TRENDING_TASTE_BOOST_MAX, TRENDING_WIDEN_FACTOR,
};
use crate::experiments;
use crate::ml::auto_tuner::AutoTuner;
use crate::ml::ctr_predictor::CtrPredictor;
use crate::ml::dwell_predictor::DwellPredictor;
use crate::ml::objectives::ObjectivePredictor;
use crate::models::*;
use crate::services::cache_manager::CacheManager;
use crate::shadowban::{
    content_eligibility, AutoStrikeCandidate, ContentEligibility, GarbageContentDetector,
    ShadowbanEnforcer,
};

/// Convertit un vecteur de features sérialisé en tableau de taille fixe.
/// Rejette toute taille inattendue : un vecteur tronqué décalerait chaque
/// feature d'un cran et corromprait le modèle silencieusement.
fn to_feature_array(features: &[f64]) -> Option<[f64; crate::ml::ctr_predictor::N_FEATURES]> {
    if features.len() != crate::ml::ctr_predictor::N_FEATURES {
        return None;
    }
    let mut out = [0.0f64; crate::ml::ctr_predictor::N_FEATURES];
    out.copy_from_slice(features);
    Some(out)
}

pub struct RecommenderService {
    pg: PgPool,
    cache: CacheManager,
    ctr_predictor: CtrPredictor,
    dwell_predictor: DwellPredictor,
    /// Têtes multi-objectifs (amplification, rejet) — voir `ml::objectives`.
    objectives: ObjectivePredictor,
    auto_tuner: std::sync::Arc<AutoTuner>,
    /// Espace collaboratif — voir `crate::collab`.
    ///
    /// Reconstruit périodiquement en tâche de fond, jamais modifié sur place :
    /// le classement le lit sous verrou partagé pendant qu'une reconstruction
    /// prépare le suivant à côté, puis l'échange d'un coup.
    collab: std::sync::Arc<std::sync::RwLock<crate::collab::CollabSpace>>,
    /// Client du modele neuronal externe — voir `crate::neural`.
    ///
    /// Inerte par defaut : il faut ET la cle de service dans l'environnement,
    /// ET `admin:taste:enabled` a `1` dans Redis. Un deploiement du moteur ne
    /// peut donc pas allumer le modele tout seul.
    pub neural: std::sync::Arc<crate::neural::NeuralClient>,
}

impl RecommenderService {
    /// Acces au cache pour les routes admin (interrupteur du modele neuronal).
    pub fn cache_manager(&self) -> &CacheManager {
        &self.cache
    }
}

impl RecommenderService {
    pub fn new(pg: PgPool, cache: CacheManager) -> Self {
        Self {
            pg,
            cache,
            ctr_predictor: CtrPredictor::new(),
            dwell_predictor: DwellPredictor::new(),
            objectives: ObjectivePredictor::new(),
            collab: Default::default(),
            neural: std::sync::Arc::new(crate::neural::NeuralClient::from_env()),
            auto_tuner: std::sync::Arc::new(AutoTuner::new()),
        }
    }

    pub fn new_with_tuner(
        pg: PgPool,
        cache: CacheManager,
        auto_tuner: std::sync::Arc<AutoTuner>,
    ) -> Self {
        Self {
            pg,
            cache,
            ctr_predictor: CtrPredictor::new(),
            dwell_predictor: DwellPredictor::new(),
            objectives: ObjectivePredictor::new(),
            collab: Default::default(),
            neural: std::sync::Arc::new(crate::neural::NeuralClient::from_env()),
            auto_tuner,
        }
    }

    /// Variante de production : recharge les modèles persistés au lieu de
    /// repartir des poids par défaut. Sans ça, chaque redémarrage jetait
    /// l'intégralité de l'apprentissage accumulé.
    pub async fn new_with_tuner_and_ml(
        pg: PgPool,
        cache: CacheManager,
        auto_tuner: std::sync::Arc<AutoTuner>,
    ) -> Self {
        let ctr_predictor = CtrPredictor::load_or_default().await;
        let dwell_predictor = DwellPredictor::load_or_default().await;
        let objectives = ObjectivePredictor::load_or_default().await;
        Self {
            pg,
            cache,
            ctr_predictor,
            dwell_predictor,
            objectives,
            collab: Default::default(),
            neural: std::sync::Arc::new(crate::neural::NeuralClient::from_env()),
            auto_tuner,
        }
    }

    pub async fn new_with_ml(pg: PgPool, cache: CacheManager) -> Self {
        let ctr_predictor = CtrPredictor::load_or_default().await;
        let dwell_predictor = DwellPredictor::load_or_default().await;
        let objectives = ObjectivePredictor::load_or_default().await;
        Self {
            pg,
            cache,
            ctr_predictor,
            dwell_predictor,
            objectives,
            collab: Default::default(),
            neural: std::sync::Arc::new(crate::neural::NeuralClient::from_env()),
            auto_tuner: std::sync::Arc::new(AutoTuner::new()),
        }
    }

    /// Enregistre un engagement/rejet et met à jour le modèle ML en temps réel.
    /// `features` provient de l'impression mémorisée à l'affichage.
    pub fn record_ctr_event(&self, features: &[f64], clicked: bool) {
        let Some(vec) = to_feature_array(features) else {
            warn!(
                len = features.len(),
                "CTR: vecteur de features de taille invalide, ignoré"
            );
            return;
        };
        self.ctr_predictor.record_interaction(vec, clicked);
        let (samples, global_ctr) = self.ctr_predictor.stats();
        if samples % 100 == 0 {
            info!(
                samples,
                global_ctr, "CTR model checkpoint — 100 new samples"
            );
        }
    }

    /// Expose les stats CTR pour l'admin node
    pub fn ctr_stats(&self) -> (u64, f64) {
        self.ctr_predictor.stats()
    }

    /// Persiste le modèle CTR sur disque.
    pub async fn persist_ctr_model(&self) {
        self.ctr_predictor.save().await;
    }

    /// Reconstruit l'espace collaboratif depuis la co-occurrence.
    ///
    /// Appelé par la boucle de fond (`ml::ctr_sweeper`). La reconstruction se
    /// fait ENTIÈREMENT hors du verrou : lire Redis et factoriser prend le
    /// temps que ça prend, et bloquer le classement pendant ce temps mettrait
    /// tous les fils en attente. Seul l'échange final est sous verrou.
    pub async fn refresh_collab_space(&self) {
        let space = crate::collab::CollabSpace::build(&self.cache).await;
        let authors = space.len();
        let usable = space.is_usable();
        *self.collab.write().unwrap() = space;
        info!(
            authors,
            usable, "Espace collaboratif echange"
        );
    }

    /// (auteurs places, exploitable ?) — pour `/admin/algo/stats`.
    pub fn collab_stats(&self) -> (usize, bool) {
        let s = self.collab.read().unwrap();
        (s.len(), s.is_usable())
    }

    pub fn ctr_predictor(&self) -> &CtrPredictor {
        &self.ctr_predictor
    }

    /// Enregistre un poids de dwell RÉELLEMENT observé (`algorithm::dwell::dwell_weight`,
    /// jamais la durée brute). `features` provient de la même impression mémorisée
    /// que le CTR, lue sans la consommer — voir `CacheManager::peek_impression`.
    pub fn record_dwell_event(&self, features: &[f64], observed_weight: f64) {
        let Some(vec) = to_feature_array(features) else {
            warn!(
                len = features.len(),
                "Dwell: vecteur de features de taille invalide, ignoré"
            );
            return;
        };
        self.dwell_predictor.record_interaction(vec, observed_weight);
        let (samples, mean_weight) = self.dwell_predictor.stats();
        if samples % 100 == 0 {
            info!(samples, mean_weight, "Dwell model checkpoint — 100 new samples");
        }
    }

    pub fn dwell_stats(&self) -> (u64, f64) {
        self.dwell_predictor.stats()
    }

    /// Persiste le modèle de dwell sur disque.
    pub async fn persist_dwell_model(&self) {
        self.dwell_predictor.save().await;
    }

    pub fn dwell_predictor(&self) -> &DwellPredictor {
        &self.dwell_predictor
    }

    // ─── Têtes multi-objectifs ───────────────────────────────────────────────

    /// Entraîne les têtes concernées par cette interaction, à partir du même
    /// vecteur d'impression que le CTR — voir `ml::objectives`. Retourne
    /// `true` si au moins une tête a appris quelque chose.
    pub fn record_objective_event(&self, features: &[f64], interaction: InteractionType) -> bool {
        let Some(vec) = to_feature_array(features) else {
            warn!(
                len = features.len(),
                "Objectifs : vecteur de features de taille invalide, ignoré"
            );
            return false;
        };
        self.objectives.record_interaction(&vec, interaction)
    }

    /// Impression expirée sans la moindre réaction : négatif pour les deux
    /// têtes — voir `ml::ctr_sweeper`.
    pub fn record_objective_ignored(&self, features: &[f64]) {
        let Some(vec) = to_feature_array(features) else {
            return;
        };
        self.objectives.record_ignored(&vec);
    }

    /// ((échantillons, taux) amplification, ((échantillons, taux) rejet)
    pub fn objective_stats(&self) -> ((u64, f64), (u64, f64)) {
        self.objectives.stats()
    }

    pub fn objective_samples(&self) -> u64 {
        self.objectives.total_samples()
    }

    pub async fn persist_objective_models(&self) {
        self.objectives.save().await;
    }

    pub fn objective_predictor(&self) -> &ObjectivePredictor {
        &self.objectives
    }

    /// Accès direct au pool — réservé au rattrapage hors-ligne
    /// (`services::ctr_backfill`), qui interroge `tweet_likes`/`tweet_retweets`
    /// pour lister les interactions passées avant de les reconstruire en
    /// features via `hydrate_tweets_for_user` ci-dessous.
    pub(crate) fn pg(&self) -> &PgPool {
        &self.pg
    }

    /// Hydrate un lot de tweets PRÉCIS (pas une sélection de candidats) pour
    /// UN lecteur donné — voir `BY_IDS_SQL`. Les doublons de `tweet_ids` ne
    /// posent pas de problème : `WHERE id = ANY($2)` les déduplique de fait.
    pub(crate) async fn hydrate_tweets_for_user(
        &self,
        user_id: &str,
        tweet_ids: &[String],
    ) -> Result<Vec<RawTweet>> {
        if tweet_ids.is_empty() {
            return Ok(Vec::new());
        }
        let uid = uuid::Uuid::parse_str(user_id)?;
        let ids: Vec<uuid::Uuid> = tweet_ids
            .iter()
            .filter_map(|s| uuid::Uuid::parse_str(s).ok())
            .collect();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let client = self.pg.get().await?;
        let rows = client.query(BY_IDS_SQL.as_str(), &[&uid, &ids]).await?;
        Ok(map_rows(rows))
    }

    /// Mémorise le vecteur de features de chaque tweet servi, pour pouvoir
    /// entraîner le modèle sur ce qu'il a réellement produit.
    async fn record_impressions(
        &self,
        user_id: &str,
        page_ids: &[String],
        scored: &[ScoredTweet],
        offset: usize,
    ) {
        let by_id: HashMap<&str, &ScoredTweet> =
            scored.iter().map(|s| (s.tweet_id.as_str(), s)).collect();

        let mut stored = 0usize;
        for (index, tweet_id) in page_ids.iter().enumerate() {
            let Some(s) = by_id.get(tweet_id.as_str()) else {
                continue;
            };
            let Some(features) = s.ctr_features.as_ref() else {
                continue;
            };

            // ── Rang réellement servi ────────────────────────────────────────
            // Les features ont été construites au SCORING, avant la
            // pagination : elles portent donc la position neutre de tête de
            // page, celle qui sert à classer. C'est ici, et seulement ici,
            // qu'on sait à quelle place ce tweet est effectivement parti.
            //
            // C'est ce rang-là qu'il faut mémoriser pour l'entraînement : un
            // tweet en position 40 cliqué vaut bien plus qu'un tweet en
            // position 1 cliqué, et sans cette distinction le modèle attribue
            // au CONTENU ce qui n'est qu'un effet de rang — puis remonte
            // encore ce qui était déjà en tête. Voir
            // `ctr_predictor::POSITION_FEATURE`.
            //
            // `offset + index` et pas `index` : une deuxième page servie
            // commence au rang 50, pas au rang 0.
            let mut features = features.clone();
            if let Some(slot) = features.get_mut(crate::ml::ctr_predictor::POSITION_FEATURE) {
                *slot = crate::ml::ctr_predictor::position_discount(offset + index);
            }

            self.cache
                .record_impression(user_id, tweet_id, &features)
                .await;
            stored += 1;
        }
        if stored > 0 {
            debug!(user_id, stored, offset, "Impressions CTR mémorisées");
        }
    }

    pub async fn recommend(&self, req: &RecommendRequest) -> Result<RecommendResponse> {
        let start = Instant::now();

        // ── Sécurité : valider que user_id est un UUID strict ────────────────────
        // Toutes les requêtes interpolent user_id dans le SQL ; un identifiant
        // non-UUID pourrait casser le littéral de chaîne (injection). On rejette
        // tôt et proprement avant le moindre accès base.
        if uuid::Uuid::parse_str(&req.user_id).is_err() {
            warn!(user_id = %req.user_id, "Rejected: user_id is not a valid UUID (injection guard)");
            anyhow::bail!("invalid user_id: must be a UUID");
        }

        let mode = req.mode.clone().unwrap_or_default();
        let mode_str = mode_label(&mode);
        let limit = req.limit.unwrap_or(50).clamp(1, 200) as usize;
        let offset = req.offset.unwrap_or(0).max(0) as usize;
        let force_refresh = req.force_refresh.unwrap_or(false);

        debug!(user_id = %req.user_id, mode = mode_str, limit, offset, force_refresh, "━━━ RECOMMEND REQUEST ━━━");

        if !force_refresh {
            if let Some(cached) = self.cache.get_recommendations(&req.user_id, mode_str).await {
                let cached_total = cached.len();
                let page: Vec<FeedEntry> = cached.into_iter().skip(offset).take(limit).collect();
                let count = page.len();
                debug!(
                    cache_hit = true,
                    cached_total,
                    page_size = count,
                    "Cache hit!"
                );
                // ⚠ Une page peut couper un fil en deux : le parent finit la
                // page N, sa réponse ouvre la page N+1. `thread_links` ne
                // rattache alors rien pour cette réponse, et les clients
                // l'écartent comme orpheline — un tweet perdu par page, contre
                // une pagination qui reste alignée sur des bornes fixes.
                let threads = thread_links(&page);
                let scores = page_scores(&page);
                let page_ids: Vec<String> = page.into_iter().map(|entry| entry.id).collect();
                let mut response = self.build_empty_response(
                    &req.user_id,
                    page_ids,
                    count,
                    mode_str,
                    start.elapsed().as_millis() as u64,
                    true,
                );
                response.threads = threads;
                response.scores = scores;
                // Les publicités sont choisies MÊME sur un service depuis le
                // cache : le classement des tweets peut être resservi tel
                // quel, pas le choix publicitaire, qui dépend du plafond de
                // fréquence et du budget restant — tous deux mouvants à la
                // minute. Un profil est rechargé pour ça (cache 300 s, donc
                // sans coût dans le cas courant).
                //
                // Uniquement en `for_you` : c'est le seul fil dont le client
                // sait afficher une publicité (étiquette « Sponsorisé »,
                // carte de compte promu). La grille Explorer (`trending`) n'a
                // jamais eu ce rendu — elle affichait la publicité comme un
                // tweet ordinaire, EN PLUS de la version organique du même
                // tweet quand elle apparaissait aussi, deux entrées de même
                // id dans une grille qui ne les distingue pas : React y
                // voyait une clé dupliquée. Sélectionner quand même consommait
                // en pure perte le plafond de fréquence du lecteur (compteur
                // Redis incrémenté dans `select_for_feed`) pour une
                // publicité jamais montrée nulle part.
                if mode_str == "for_you" {
                    if let Ok(profile) = self.build_user_profile(&req.user_id).await {
                        response.ads = crate::ads::select_for_feed(
                            &self.pg,
                            &self.cache,
                            &req.user_id,
                            &profile,
                            response.tweet_ids.len(),
                        )
                        .await;
                    }
                }
                if req.enable_experiments.unwrap_or(false) {
                    response.experiments = experiments::assign_variants(
                        &self.pg,
                        &req.user_id,
                        &response.tweet_ids,
                    )
                    .await
                    .unwrap_or_else(|error| {
                        warn!(error = ?error, "A/B assignment failed on cached recommendations");
                        Vec::new()
                    });
                }
                return Ok(response);
            }
        }

        // ── Vérification hard-ban sur le demandeur lui-même ──────────────────────
        // Un compte hard-banni ne reçoit plus de recommandations personnalisées
        if self.cache.admin_is_hard_banned(&req.user_id).await {
            warn!(user_id = %req.user_id, "Hard-banned user requested recommendations — returning empty");
            return Ok(self.build_empty_response(
                &req.user_id,
                vec![],
                0,
                mode_str,
                start.elapsed().as_millis() as u64,
                false,
            ));
        }

        debug!("Building user profile...");
        let profile = self.build_user_profile(&req.user_id).await?;
        trace!(following_count = profile.following_ids.len(), top_authors = profile.top_authors.len(),
               user_type = ?profile.user_type, "User profile built");

        // ── Charger la liste des hard-bannis pour filtrage SQL ───────────────────
        // Unie ici aux comptes que CE lecteur a bloqués (ou qui l'ont bloqué,
        // voir `UserProfile::blocked_ids`) : un compte bloqué doit disparaître
        // du vivier exactement comme un compte hard-banni, mais seulement pour
        // ce lecteur-là — d'où l'union par requête plutôt qu'un ajout au cache
        // admin, qui est partagé entre tous les utilisateurs.
        let mut banned_set = self.cache.admin_get_hard_banned_set().await;
        debug!(banned_count = banned_set.len(), "Hard-banned set loaded");
        banned_set.extend(profile.blocked_ids.iter().cloned());

        debug!("Collecting candidates from {} sources...", 8);
        let (mut sources, source_stats) = self
            .collect_candidates(&req.user_id, &profile, &mode, &banned_set)
            .await?;

        // Remonter les parents AVANT la déduplication et le plancher de qualité :
        // un parent est un tweet comme un autre et doit subir exactement les
        // mêmes contrôles. L'entrer plus tard reviendrait à le faire passer par
        // une porte que les candidats normaux n'ont pas.
        if let Err(error) = self
            .hydrate_thread_parents(&req.user_id, &mut sources, &banned_set)
            .await
        {
            // Échec non bloquant : sans les parents, `shape_feed` retombe sur
            // son comportement d'avant — il écarte les réponses. Le fil est plus
            // pauvre, il n'est pas cassé.
            warn!(error = ?error, "Remontée des parents de fil impossible, les réponses seront écartées");
        }

        self.hydrate_semantic_candidates(&req.user_id, &profile, &mut sources, &banned_set)
            .await;
        self.hydrate_cooccurrence_candidates(&profile, &mut sources, &banned_set)
            .await;

        let total_candidates = sources.len();
        debug!(
            total_candidates,
            trending = source_stats.trending,
            social_graph = source_stats.social_graph,
            viral = source_stats.viral,
            discovery = source_stats.discovery,
            temporal = source_stats.temporal,
            influencer = source_stats.influencer,
            personalized = source_stats.personalized,
            quality = source_stats.quality,
            "Candidates collected from 8 sources"
        );

        if sources.is_empty() {
            warn!("No candidates found for user");
            return Ok(self.build_empty_response(
                &req.user_id,
                vec![],
                0,
                mode_str,
                start.elapsed().as_millis() as u64,
                false,
            ));
        }

        debug!("Deduplicating {} candidates...", total_candidates);
        let mut deduped = deduplicate(sources);
        let deduped_count = deduped.len();

        // ── Charger les niveaux de shadowban depuis Redis ────────────────────────
        // On collecte les author_ids uniques puis on fait un batch lookup Redis.
        let author_ids: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            deduped
                .iter()
                .filter_map(|t| {
                    if seen.insert(t.user_id.clone()) {
                        Some(t.user_id.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };
        // Décision manuelle ET niveau dérivé des avertissements, en deux MGET
        // pour tout le pool (voir `shadowban/store.rs`). L'ancienne lecture
        // faisait un aller-retour Redis par auteur, en série.
        let shadowban_levels = self.cache.shadowban_load_levels(&author_ids).await;
        if !shadowban_levels.is_empty() {
            debug!(
                count = shadowban_levels.len(),
                "Shadowban levels loaded from Redis"
            );
            for tweet in deduped.iter_mut() {
                if let Some(&level) = shadowban_levels.get(&tweet.user_id) {
                    tweet.author_shadowban_level = level;
                }
            }
        }
        // Frein de vélocité (1h, ×0.5) — voir `crate::velocity`. Même patron de
        // lecture par lot que le registre d'avertissements, mais un mécanisme
        // entièrement séparé : pas de niveau, pas de surface fermée, juste un
        // multiplicateur appliqué plus bas, après le scoring.
        let velocity_throttles = self.cache.load_velocity_throttles(&author_ids).await;
        if !velocity_throttles.is_empty() {
            debug!(
                count = velocity_throttles.len(),
                "Velocity throttles loaded from Redis"
            );
        }
        // Boost temps réel (30 min) — voir `services::feedback_loop`. Batché sur
        // le même `author_ids` que le shadowban et le frein de vélocité : aucun
        // aller-retour Redis supplémentaire par tweet.
        let realtime_author_boosts = self
            .cache
            .load_realtime_author_boosts(&req.user_id, &author_ids)
            .await;
        debug!(
            deduped_count,
            removed = total_candidates - deduped_count,
            "Deduplication complete"
        );

        // ── Plancher de qualité ─────────────────────────────────────────────────
        // Écarté AVANT le scoring, et pas rétrogradé : un tweet sans apport
        // remonterait sinon dès que le vivier est maigre. Les tweets non annotés
        // et les étiquettes hésitantes passent (voir `below_quality_floor`).
        let before_quality = deduped.len();
        deduped.retain(|t| !crate::algorithm::d9_llm_understanding::below_quality_floor(t));
        let dropped_low_quality = before_quality - deduped.len();
        if dropped_low_quality > 0 {
            debug!(
                dropped = dropped_low_quality,
                remaining = deduped.len(),
                floor = crate::algorithm::d9_llm_understanding::MIN_QUALITY,
                "Low-quality tweets excluded from recommendations"
            );
        }

        // ── Déjà vu : ne pas resservir la journée d'hier ────────────────────────
        // `exclude_seen` était déclaré dans `RecommendRequest` et lu NULLE PART —
        // de la plomberie morte, comme `force_refresh` avant lui. Or c'est le
        // levier qui décide si une page de découverte vaut la peine d'être
        // rouverte demain : sans lui, revenir donne la même page, et la seule
        // fraîcheur possible vient de ce qui a été publié entre-temps.
        //
        // La liste vient du set Redis `twitninf:seen:<user>` (TTL 24 h, rafraîchi
        // à chaque marquage) : c'est donc « ce que j'ai vu aujourd'hui », pas un
        // historique définitif — un bon tweet redevient éligible le lendemain.
        //
        // Filtrage abandonné s'il ne laisse pas de quoi remplir la page : voir
        // `EXCLUDE_SEEN_MIN_REMAINING`.
        if req.exclude_seen.unwrap_or(false) && !profile.seen_tweet_ids.is_empty() {
            let seen: HashSet<&str> = profile.seen_tweet_ids.iter().map(|s| s.as_str()).collect();
            let remaining = deduped
                .iter()
                .filter(|t| !seen.contains(t.id.as_str()))
                .count();
            if remaining >= EXCLUDE_SEEN_MIN_REMAINING {
                let before = deduped.len();
                deduped.retain(|t| !seen.contains(t.id.as_str()));
                debug!(
                    dropped = before - deduped.len(),
                    remaining = deduped.len(),
                    "Already-seen tweets excluded"
                );
            } else {
                debug!(
                    remaining,
                    floor = EXCLUDE_SEEN_MIN_REMAINING,
                    "Already-seen filter skipped: would leave too few candidates"
                );
            }
        }

        // ── Charger les poids actifs (admin override > auto-tuner > defaults) ────
        let admin_weights = self.cache.admin_load_weights().await;
        self.auto_tuner
            .maybe_update(&self.ctr_predictor, admin_weights.as_ref());
        let active_weights = self.auto_tuner.active_weights(admin_weights.as_ref());

        // Bras du bandit (Phase 3) = auteurs de ce pool. Même `author_ids` que
        // le shadowban/frein de vélocité ci-dessus, un seul aller-retour de
        // plus (deux `MGET`, voir `bandit::contextual::store::load_arm_stats`).
        let arm_stats = self.cache.load_arm_stats(&author_ids).await;

        // Renfort d'affinité, mode Trending (= onglet Explorer) UNIQUEMENT.
        // Calculé ici et non dans `score_all`, qui est synchrone : même patron
        // que `velocity_throttles` et `realtime_author_boosts` juste au-dessus.
        let taste_boosts = self.taste_boosts(&mode, &profile, &deduped).await;

        // ── Modele neuronal externe (`taste-model`) ──────────────────────────
        //
        // L'interrupteur est relu ICI, sur l'aller-retour Redis qui existe deja
        // pour les poids : eteindre le modele prend effet a la requete suivante,
        // sans recompiler ni redemarrer le moteur. C'est ce qui rend le
        // branchement sur le fil de tout le monde reversible en une commande.
        self.neural
            .set_enabled(self.cache.admin_flag_enabled("admin:taste:enabled").await);
        // Un seul appel pour tout le vivier, avant la boucle SYNCHRONE de
        // scoring — meme patron que `velocity_throttles` et `taste_boosts`.
        // En cas de panne, de lenteur ou d'extinction, la carte est vide et le
        // classement est exactement celui d'avant.
        let neural_scores = if self.neural.is_active() {
            let ids: Vec<String> = deduped.iter().map(|t| t.id.clone()).collect();
            self.neural.scores(&req.user_id, &ids).await
        } else {
            std::collections::HashMap::new()
        };

        debug!("Scoring {} tweets with 8 dimensions...", deduped_count);
        let mut auto_strike_candidates = Vec::new();
        let scored = self.score_all(
            &deduped,
            &profile,
            &mode,
            &active_weights,
            &velocity_throttles,
            &realtime_author_boosts,
            &arm_stats,
            &taste_boosts,
            &neural_scores,
            &mut auto_strike_candidates,
        );
        debug!(scored_count = scored.len(), "Scoring complete");
        if !auto_strike_candidates.is_empty() {
            self.cache
                .shadowban_process_auto_strikes(auto_strike_candidates)
                .await;
        }

        // Show top 5 scores
        let top_5: Vec<_> = scored
            .iter()
            .take(5)
            .map(|s| (&s.tweet_id, s.score))
            .collect();
        trace!("Top 5 scores: {:?}", top_5);

        // ── La carte ne contient QUE les tweets ADMIS ────────────────────────
        //
        // Construite sur `deduped`, elle contenait aussi ceux que
        // `enforcer.admit` venait d'écarter. `shape_feed` remonte la chaîne des
        // réponses en lisant cette carte : une réponse à un compte `Ghosted`
        // ramenait donc son parent dans le fil, alors que ce parent avait été
        // refusé quelques lignes plus haut.
        //
        // Constaté en production le 2026-08-22 : `policiercongo`, au niveau
        // `Ghosted`, sortait encore 3 tweets dans le fil d'un lecteur qui ne le
        // suit pas — les 3 étaient des parents tirés par leurs réponses.
        //
        // Le SQL des candidats ferme déjà cette porte pour les comptes bannis
        // ou suspendus (« répondre à un compte bloqué aurait suffi à le faire
        // réapparaître par la porte de derrière »), mais le shadowban est
        // décidé côté Rust, APRÈS la requête. Il fallait la fermer ici aussi.
        //
        // Conséquence voulue : une réponse dont le parent n'est pas admis est
        // écartée elle aussi. `shape_feed` le fait déjà pour tout ancêtre
        // manquant — « le contexte serait incomplet, on écarte ».
        let admitted: std::collections::HashSet<&str> =
            scored.iter().map(|s| s.tweet_id.as_str()).collect();
        let tweet_map: HashMap<&str, &RawTweet> = deduped
            .iter()
            .filter(|t| admitted.contains(t.id.as_str()))
            .map(|t| (t.id.as_str(), t))
            .collect();

        // Mise en forme du fil : plafonne les réponses et fait précéder chacune
        // du tweet auquel elle répond.
        let all_ids = shape_feed(&scored, &tweet_map);
        // Puis étalement par auteur — appliqué AVANT la mise en cache, donc les
        // pages servies depuis Redis en héritent aussi.
        let all_ids = spread_by_author(all_ids, &tweet_map);

        let pairs: Vec<(&RawTweet, &ScoredTweet)> = scored
            .iter()
            .filter_map(|s| tweet_map.get(s.tweet_id.as_str()).map(|t| (*t, s)))
            .collect();

        debug!("Computing feed quality metrics...");
        let metrics = compute_feed_metrics(&pairs);
        debug!(
            diversity_score = metrics.diversity_score,
            freshness_score = metrics.freshness_score,
            relevance_score = metrics.relevance_score,
            viral_potential = metrics.viral_potential,
            novelty_score = metrics.novelty_score,
            "Feed metrics calculated"
        );

        // Le lien de fil est figé ICI, tant qu'on a encore les tweets complets :
        // après la mise en cache il ne reste que des identifiants.
        let score_by_id: HashMap<&str, f64> =
            scored.iter().map(|s| (s.tweet_id.as_str(), s.score)).collect();
        let all_entries = as_feed_entries(&all_ids, &tweet_map, &score_by_id, &profile);

        let adaptive_ttl = adaptive_ttl(&profile, &mode);
        debug!(ttl_seconds = adaptive_ttl, "Setting cache TTL");
        self.cache
            .set_recommendations_ttl(&req.user_id, mode_str, &all_entries, adaptive_ttl)
            .await;

        let total_available = self.count_available(&req.user_id).await.unwrap_or(1000);
        let page: Vec<FeedEntry> = all_entries.into_iter().skip(offset).take(limit).collect();
        let threads = thread_links(&page);
        let page_scores = page_scores(&page);
        let page_ids: Vec<String> = page.into_iter().map(|entry| entry.id).collect();
        let count = page_ids.len();
        debug!(
            pagination_offset = offset,
            pagination_limit = limit,
            page_size = count,
            total_available,
            "Pagination applied"
        );

        // ── Mémoriser les impressions servies pour l'entraînement CTR ────────────
        // Uniquement les tweets réellement renvoyés : ceux qui sortent de la
        // pagination n'ont jamais été exposés au lecteur, les compter en
        // négatif fabriquerait des rejets qui n'ont pas eu lieu.
        self.record_impressions(&req.user_id, &page_ids, &scored, offset)
            .await;
        let experiment_assignments = if req.enable_experiments.unwrap_or(false) {
            experiments::assign_variants(&self.pg, &req.user_id, &page_ids)
                .await
                .unwrap_or_else(|error| {
                    warn!(error = ?error, "A/B assignment failed on recommendations");
                    Vec::new()
                })
        } else {
            Vec::new()
        };

        info!(
            user_id = %req.user_id, mode = mode_str,
            candidates = total_candidates, deduped = deduped_count,
            returned = count,
            latency_ms = start.elapsed().as_millis(),
            "NeuralRank recommendations computed"
        );

        // Voir le commentaire équivalent sur le chemin servi depuis le cache,
        // plus haut : uniquement en `for_you`.
        let ads = if mode_str == "for_you" {
            crate::ads::select_for_feed(&self.pg, &self.cache, &req.user_id, &profile, count).await
        } else {
            Vec::new()
        };

        Ok(RecommendResponse {
            success: true,
            user_id: req.user_id.clone(),
            tweet_ids: page_ids,
            threads,
            scores: page_scores,
            ads,
            count,
            algorithm: "NeuralRank Fusion",
            algorithm_version: "2.2.0 — 8 dimensions + ML CTR + bandit + adaptive A/B",
            mode: mode_str.to_string(),
            latency_ms: start.elapsed().as_millis() as u64,
            cache_hit: false,
            experiments: experiment_assignments,
            metadata: RecommendMetadata {
                candidates_collected: total_candidates,
                sources: SourceStats {
                    deduplicated_total: deduped_count,
                    ..source_stats
                },
                user_profile: UserProfileSummary {
                    user_type: format!("{:?}", profile.user_type),
                    confidence: profile.profile_confidence,
                    personality: format!("{:?}", profile.personality_type),
                    engagement_velocity: profile.engagement_velocity,
                    engagement_trend: if profile.engagement_trend > 1.2 {
                        "increasing".into()
                    } else if profile.engagement_trend < 0.8 {
                        "decreasing".into()
                    } else {
                        "stable".into()
                    },
                    network_influence: profile.network_influence,
                    most_active_hour: profile.most_active_hour,
                    churn_risk: profile.churn_risk,
                },
                quality_metrics: QualityMetrics {
                    diversity_score: metrics.diversity_score,
                    freshness_score: metrics.freshness_score,
                    relevance_score: metrics.relevance_score,
                    viral_potential: metrics.viral_potential,
                    novelty_score: metrics.novelty_score,
                },
                pagination: Pagination {
                    limit: limit as i32,
                    offset: offset as i32,
                    has_more: offset + count < total_available as usize,
                    total_available,
                },
            },
        })
    }

    /// Facteurs de renfort d'affinité, indexés par id de tweet.
    ///
    /// Vide partout SAUF en mode Trending : c'est une décision de produit, pas
    /// une optimisation. `ForYou` est déjà personnalisé par le graphe social et
    /// l'affinité d'auteur, et `Discover` cherche délibérément l'inverse (il
    /// déprécie les comptes suivis) — y ajouter ce renfort brouillerait ce que
    /// chaque mode est censé faire. Voir `TRENDING_TASTE_BOOST_MAX`.
    ///
    /// Vide aussi quand le lecteur n'a pas encore de vecteur de goût (compte
    /// neuf, ou likes pas encore embeddés) : la page reste alors exactement ce
    /// qu'elle était avant ce renfort.
    async fn taste_boosts(
        &self,
        mode: &RecommendMode,
        profile: &UserProfile,
        tweets: &[RawTweet],
    ) -> HashMap<String, f64> {
        if *mode != RecommendMode::Trending {
            return HashMap::new();
        }
        let Some(taste) = profile.taste_vector.as_ref() else {
            debug!(
                user_id = %profile.user_id,
                "Trending: pas de vecteur de goût, aucun renfort d'affinité"
            );
            return HashMap::new();
        };

        let ids: Vec<String> = tweets.iter().map(|t| t.id.clone()).collect();
        let similarities = match crate::embeddings::taste_similarities(&self.pg, taste, &ids).await
        {
            Ok(s) => s,
            Err(e) => {
                // Silencieux à dessein : le renfort est un complément, son
                // absence doit laisser une page de tendances normale.
                debug!(user_id = %profile.user_id, error = %e,
                       "Trending: similarités de goût indisponibles");
                return HashMap::new();
            }
        };

        let factors = crate::algorithm::trending::taste_boost_factors(
            &similarities,
            TRENDING_TASTE_BOOST_MAX,
        );
        debug!(
            user_id = %profile.user_id,
            mesures = similarities.len(),
            renforces = factors.len(),
            candidats = tweets.len(),
            "Trending: renfort d'affinité calculé"
        );
        factors
    }

    fn score_all(
        &self,
        tweets: &[RawTweet],
        profile: &UserProfile,
        mode: &RecommendMode,
        weights: &crate::admin::AlgoWeights,
        velocity_throttles: &HashMap<String, f64>,
        realtime_author_boosts: &HashMap<String, f64>,
        arm_stats: &HashMap<String, (f64, u32)>,
        taste_boosts: &HashMap<String, f64>,
        neural_scores: &HashMap<String, f64>,
        auto_strike_candidates: &mut Vec<AutoStrikeCandidate>,
    ) -> Vec<ScoredTweet> {
        // Compteurs indexés par référence dans `tweets` et non par `String`
        // clonée : la boucle clonait l'identifiant de l'auteur de CHAQUE
        // candidat pour l'insérer, puis le thème de chaque tweet annoté. 1700
        // allocations par recommandation pour deux compteurs.
        let mut author_count: HashMap<&str, u32> = HashMap::new();
        // Anti-répétition THÉMATIQUE — voir `theme_diversity_multiplier`. Le
        // champ `theme` (annotation LLM) était déjà chargé et disponible ;
        // rien ne le lisait jamais pour de la diversité, seulement D9 pour
        // repérer deux catégories dégradantes.
        let mut theme_count: HashMap<&str, u32> = HashMap::new();
        let mut scored_feed: Vec<ScoredTweet> = Vec::with_capacity(tweets.len());
        // Forme du fil tenue au fil de l'eau — voir `FeedShape`. D6 la
        // recalculait en parcourant tout le fil deja score a chaque candidat.
        let mut feed_shape = FeedShape::empty();
        let (ctr_samples, _) = self.ctr_predictor.stats();
        // Activer ML CTR seulement si suffisamment de données (évite overfitting cold-start)
        let use_ml = ctr_samples >= crate::algorithm::scoring::ML_MIN_SAMPLES;
        let (dwell_samples, _) = self.dwell_predictor.stats();
        // Même garde-fou que le CTR, seuil identique : les deux modèles partagent
        // le même vecteur de features et démarrent d'un prior tout aussi vide.
        let use_dwell = dwell_samples >= crate::algorithm::scoring::ML_MIN_SAMPLES;
        // Le melange depend de la SURFACE — voir `BlendProfile`. Resolu une
        // fois pour tout le lot : le mode ne change pas d'un candidat a
        // l'autre.
        // ── Position du lecteur dans l'espace collaboratif ─────────────────
        //
        // Resolue UNE fois pour tout le lot : elle ne depend que du lecteur.
        // `None` quand l'espace est trop maigre ou que ce lecteur n'a aucun
        // auteur place — le trait retombe alors sur sa valeur neutre, et le
        // modele se comporte comme s'il n'existait pas.
        let collab_space = self.collab.read().unwrap();
        let reader_collab = collab_space
            .is_usable()
            .then(|| collab_space.reader(&profile.top_authors))
            .flatten();

        let blend_profile = match mode {
            RecommendMode::Trending => BlendProfile::TRENDING,
            RecommendMode::Feed => BlendProfile::FOLLOWING,
            RecommendMode::ForYou | RecommendMode::Discover => BlendProfile::BALANCED,
        };
        let detector = GarbageContentDetector::new();
        let enforcer = ShadowbanEnforcer::new();
        // Instant de référence du lot — voir `score_tweet_with_weights_at`.
        let now = chrono::Utc::now();
        let mut dropped_garbage = 0usize;
        trace!(mode = ?mode, use_ml, ctr_samples, use_dwell, dwell_samples, "Scoring all tweets with mode adjustments");

        for (idx, tweet) in tweets.iter().enumerate() {
            // ── Admission par surface ────────────────────────────────────────────
            //
            // Deux verdicts distincts, appliqués ensemble : l'éligibilité de CE
            // post, et le niveau de restriction de son AUTEUR. Le premier tombe
            // sur du spam isolé publié par un compte sain, le second sur un compte
            // qui accumule les avertissements — les confondre revenait à ne
            // traiter correctement ni l'un ni l'autre.
            //
            // Ce qui change par rapport au filtre précédent :
            //
            // - Le seuil ne retire plus le tweet du pipeline entier, seulement des
            //   surfaces où l'on POUSSE du contenu vers des gens qui n'ont rien
            //   demandé. Un abonné voit toujours ce que publie le compte qu'il
            //   suit, quel qu'en soit l'état — c'est l'invariant de `admit()`.
            // - Le niveau de compte ferme enfin les surfaces pour de bon.
            //   `excludes_trending()` existait déjà mais n'était appelé nulle
            //   part : un compte `Ghosted` n'était que rétrogradé (×0,05), donc
            //   toujours capable d'atteindre les Tendances avec un pic
            //   d'engagement suffisant.
            let follows_author = profile.follows(&tweet.user_id);
            let surface = ShadowbanEnforcer::effective_surface(tweet, mode, follows_author);
            let signals = detector.detect(tweet);
            let eligibility = content_eligibility(tweet, &signals);
            // Un motif d'inéligibilité qui compte au niveau du compte (voir
            // `IneligibilityReason::policy_for`) devient un candidat à
            // l'avertissement automatique — traité en aval, après le scoring,
            // pour ne pas mêler d'écriture Redis à cette boucle synchrone. Ça
            // vaut pour toute surface, pas seulement celles fermées à CE
            // lecteur : un compte suivi qui publie du spam doit accumuler des
            // avertissements même si son abonné continue de le voir.
            if let ContentEligibility::NotRecommended(reason) = eligibility {
                let toxicity_category = tweet.llm.as_ref().map(|l| l.toxicity_category.as_str());
                if let Some(policy) = reason.policy_for(toxicity_category) {
                    auto_strike_candidates.push(AutoStrikeCandidate {
                        tweet_id: tweet.id.clone(),
                        author_id: tweet.user_id.clone(),
                        policy,
                        reason_label: reason.label(),
                    });
                }
            }
            let verdict = enforcer.admit(
                tweet.author_shadowban_level,
                eligibility,
                surface,
                follows_author,
            );
            if !verdict.allowed {
                dropped_garbage += 1;
                trace!(tweet_id = %tweet.id, surface = surface.label(),
                       blocked_by = verdict.blocked_by.unwrap_or("unknown"),
                       "Écarté de cette surface (reste visible des abonnés)");
                continue;
            }

            // Explorer (Trending) n'applique jamais la pénalité de répétition
            // d'auteur : demande explicite du 2026-08-21, « aucune pénalité
            // dans explore » — voir le bloc `RecommendMode::Trending`
            // ci-dessous pour le reste du traitement dédié à ce mode.
            let ac = if *mode == RecommendMode::Trending {
                0
            } else {
                *author_count.get(tweet.user_id.as_str()).unwrap_or(&0)
            };
            // Boost temps réel (30 min, voir `services::feedback_loop`) : réagit
            // à un like/skip de CETTE session, pas seulement au profil rechargé
            // toutes les 300s ou au prochain TTL de cache de feed.
            let realtime_boost = realtime_author_boosts
                .get(&tweet.user_id)
                .copied()
                .unwrap_or(0.0);
            // Phase 2+3: score_tweet_ml_with_weights intègre poids actifs + CTR predictor + realtime boost
            let mut s = score_tweet_ml_with_weights(
                tweet,
                profile,
                ac,
                feed_shape,
                if use_ml {
                    Some(&self.ctr_predictor)
                } else {
                    None
                },
                if use_dwell {
                    Some(&self.dwell_predictor)
                } else {
                    None
                },
                // Chaque tête porte son propre seuil de démarrage à froid en
                // interne : passer le prédicteur ne suffit pas à le faire
                // peser, il faut qu'il ait appris.
                Some(&self.objectives),
                realtime_boost,
                weights,
                blend_profile,
                reader_collab
                    .as_ref()
                    .and_then(|v| collab_space.affinity(v, &tweet.user_id))
                    .unwrap_or(0.5),
                // `is_warm` : tant que la moyenne courante n'a pas assez
                // d'observations, le `lift` n'a pas de sens et la tete reste
                // muette — meme garde que le seuil de demarrage a froid des
                // tetes `objectives`.
                neural_scores
                    .get(&tweet.id)
                    .filter(|_| self.neural.is_warm())
                    .map(|p| self.neural.lift(*p)),
                // Instant du lot, et signaux de contenu poubelle déjà calculés
                // quelques lignes plus haut pour l'admission par surface.
                ScoringContext::at(now).with_garbage(signals),
            );
            let base_score = s.score;

            match mode {
                RecommendMode::Trending => {
                    // Explorer, 100% personnalisé : demande explicite du
                    // 2026-08-21 (« aucune pénalité dans explore »). Fini le
                    // mélange 40% base / 60% vélocité impersonnelle — le score
                    // de base (déjà calculé plus haut avec `ac` forcé à 0, donc
                    // sans pénalité de répétition d'auteur) porte seul le
                    // classement. La pénalité anti-répétition thématique,
                    // appliquée plus bas pour les autres modes, est également
                    // sautée pour Trending — voir le test sur `*mode` à cet
                    // endroit.
                    //
                    // Fatigue d'exposition et bonus média restent : ce ne sont
                    // pas des mécanismes anti-bulle, juste du confort de
                    // grille (ne pas rabâcher le même tweet, mettre en valeur
                    // les médias qu'une grille sait le mieux montrer).
                    let mut score = s.score;

                    let fatigue = impression_fatigue(tweet.viewer_impressions);
                    score *= fatigue;

                    if tweet.has_media {
                        score *= TRENDING_MEDIA_BOOST;
                    }

                    // ── Affinité de goût, propre à ce mode ────────────────────
                    // Renfort pré-existant (branche explorer-taste-boost) :
                    // absent de la map = 1,0, jamais de pénalité — voir
                    // `taste_boost_factors`. Redondant en pratique maintenant
                    // que le score est déjà 100% personnalisé ci-dessus, mais
                    // inoffensif (plafonné à ×1,18) donc conservé plutôt que
                    // supprimé au merge.
                    let taste_boost = taste_boosts.get(&tweet.id).copied().unwrap_or(1.0);
                    score *= taste_boost;

                    s.score = score.clamp(0.0, 1.0);
                    trace!(tweet_id = %tweet.id, base_score, fatigue, taste_boost,
                           has_media = tweet.has_media, final_score = s.score,
                           "Trending/Explorer: 100% personnalisé, aucune pénalité anti-bulle");
                }
                RecommendMode::Discover => {
                    let mut multiplier = 1.0;
                    if profile.follows(&tweet.user_id) {
                        multiplier *= 0.65;
                        trace!(tweet_id = %tweet.id, "Discover: user follows author, reducing score by 35%");
                    }
                    if tweet.source == TweetSource::Discovery {
                        multiplier *= 1.30;
                        trace!(tweet_id = %tweet.id, "Discover: from Discovery source, boosting by 30%");
                    }
                    s.score = (s.score * multiplier).min(1.0);
                }
                RecommendMode::Feed => {
                    s.score = apply_follow_boost(s.score, tweet, profile, "Feed");
                }
                RecommendMode::ForYou => {
                    // C'est le mode que demandent réellement les applications.
                    // Il n'appliquait aucun ajustement : le boost abonnés
                    // n'existait que dans `Feed`, que personne n'appelle. Un
                    // abonnement ne pesait donc que sa part de D3, invisible
                    // derrière l'engagement.
                    s.score = apply_follow_boost(s.score, tweet, profile, "ForYou");
                }
            }

            // ── Réponse explicite : « ça ne m'intéresse pas » ─────────────────
            // Appliqué APRÈS l'ajustement de mode, et à tous les modes : une
            // réponse donnée à la main n'a pas à être rejouée différemment
            // selon l'onglet où on se trouve.
            //
            // Porte sur l'AUTEUR, pas sur le seul tweet refusé. Mesuré par
            // Mozilla sur YouTube : le bouton « pas intéressé » n'évite que
            // ~11 % des recommandations non voulues, quand « ne plus
            // recommander cette chaîne » en évite 43 %. Écarter un tweet parmi
            // les mille du même compte ne change rien de perceptible, et
            // l'utilisateur en conclut — à raison — que le bouton est décoratif.
            if !profile.damped_authors.is_empty() {
                if let Some(&strikes) = profile.damped_authors.get(&tweet.user_id) {
                    let damping = author_damping(strikes);
                    s.score *= damping;
                    trace!(tweet_id = %tweet.id, author = %tweet.user_id, strikes, damping,
                           "Author damped by explicit disinterest");
                }
            }

            // ── Frein de vélocité ──────────────────────────────────────────────
            // Appliqué à tous les modes, comme la sourdine ci-dessus. Un simple
            // multiplicateur, jamais un filtre dur : contrairement au shadowban,
            // ce frein ne ferme aucune surface — il retarde, il n'exclut pas.
            if let Some(&mult) = velocity_throttles.get(&tweet.user_id) {
                s.score *= mult;
                trace!(tweet_id = %tweet.id, author = %tweet.user_id, mult,
                       "Velocity throttle applied");
            }

            // ── Anti-répétition thématique ───────────────────────────────────
            // Sautée pour Trending/Explorer (« aucune pénalité dans explore »,
            // 2026-08-21) : ce mode ne doit plus jamais être freiné par un
            // thème trop présent dans le fil.
            //
            // Uniquement sur un tweet ANNOTÉ : un thème vide/inconnu partagé
            // par tout le non-annoté formerait un faux « sujet » géant et
            // pénaliserait des tweets qui n'ont rien en commun.
            if *mode != RecommendMode::Trending {
                if let Some(theme) = tweet
                    .llm
                    .as_ref()
                    .map(|l| l.theme.as_str())
                    .filter(|t| !t.is_empty())
                {
                    let tc = *theme_count.get(theme).unwrap_or(&0);
                    if tc > 0 {
                        let theme_mult = theme_diversity_multiplier(tc);
                        s.score *= theme_mult;
                        trace!(tweet_id = %tweet.id, theme, tc, theme_mult, "Theme diversity applied");
                    }
                    *theme_count.entry(theme).or_insert(0) += 1;
                }
            }

            if idx < 3 {
                trace!(idx, tweet_id = %tweet.id, base_score, final_score = s.score, "Sample scored tweet");
            }

            *author_count.entry(tweet.user_id.as_str()).or_insert(0) += 1;
            feed_shape.push(tweet.has_media);
            scored_feed.push(s);
        }

        if dropped_garbage > 0 {
            debug!(dropped = dropped_garbage, kept = scored_feed.len(), mode = ?mode,
                   "Filtre d'admission par surface appliqué");
        }

        debug!(
            "Sorting {} scored tweets by final score...",
            scored_feed.len()
        );
        // `total_cmp` et non `partial_cmp().unwrap_or(Equal)` : même ordre sur
        // toute valeur finie, mais une comparaison d'entiers au lieu d'un
        // `Option` construit puis déballé à chaque paire — et un ordre TOTAL,
        // donc un `NaN` échappé du scoring ne peut plus rendre le tri
        // incohérent (avec `Equal` partout, l'ordre final dépendait de
        // l'algorithme de tri). Le tri reste STABLE : deux tweets à score égal
        // gardent leur ordre d'arrivée, ce dont dépend l'adjacence parent/
        // réponse construite plus bas.
        scored_feed.sort_by(|a, b| b.score.total_cmp(&a.score));

        let top_scores: Vec<_> = scored_feed
            .iter()
            .take(3)
            .map(|s| (&s.tweet_id, s.score))
            .collect();
        debug!("Top 3 final scores: {:?}", top_scores);

        // Trending seul : réordonnancement aléatoire pondéré par score (Gumbel-max),
        // tiré à neuf à CHAQUE appel de `score_all` — donc à chaque calcul non
        // servi depuis le cache Rust (recompute déclenché par `force_refresh` ou
        // expiration du TTL). `.score` n'est jamais modifié : seul l'ORDRE de
        // `scored_feed` change, tout le reste du pipeline (métriques, logs,
        // impressions CTR) continue de voir les vrais scores calculés.
        //
        // Les autres modes gardent le tri strict + bandit ci-dessous : Trending
        // ne passe jamais par le bandit (personnalisé par construction, sans
        // rapport avec « ce qui prend de l'ampleur en ce moment »).
        if *mode == RecommendMode::Trending {
            scored_feed = trending_draw(scored_feed);
            let shuffled_top: Vec<_> = scored_feed
                .iter()
                .take(3)
                .map(|s| (&s.tweet_id, s.score))
                .collect();
            debug!(
                "Trending: two-temperature draw applied, new top 3: {:?}",
                shuffled_top
            );
        }

        // Phase 3: Contextual Bandit — réorganise le feed (80% exploit / 20% explore)
        // Seulement en mode ForYou/Feed, pas en Trending (qui a son propre
        // réordonnancement aléatoire ci-dessus)
        if matches!(mode, RecommendMode::ForYou | RecommendMode::Feed) {
            let selection =
                bandit_select(&scored_feed, tweets, profile, scored_feed.len(), arm_stats);
            debug!(
                exploit = selection.exploit_count,
                explore = selection.explore_count,
                "Phase 3: Bandit reordering applied (80% exploit + 20% explore)"
            );
            // Réordonner scored_feed selon l'ordre du bandit.
            //
            // Par POSITIONS et non par une table `String -> ScoredTweet` : la
            // version d'avant clonait l'identifiant de chaque tweet pour s'en
            // servir de clé, soit un millier et demi d'allocations dont pas une
            // ne survivait à la ligne suivante. L'index est bâti sur des
            // références, puis relâché avant que `scored_feed` soit consommé.
            let order: Vec<usize> = {
                let position: crate::utils::FxHashMap<&str, usize> = scored_feed
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.tweet_id.as_str(), i))
                    .collect();
                selection
                    .tweet_ids
                    .iter()
                    .filter_map(|id| position.get(id.as_str()).copied())
                    .collect()
            };
            // `take()` et non un simple index : si le bandit citait deux fois
            // le même tweet, la table précédente ne l'aurait rendu qu'une fois
            // (`remove`). Le trou laissé derrière préserve ce comportement.
            let mut slots: Vec<Option<ScoredTweet>> =
                scored_feed.into_iter().map(Some).collect();
            return order
                .into_iter()
                .filter_map(|i| slots[i].take())
                .collect();
        }

        scored_feed
    }

    // `pub(crate)` : le rattrapage hors-ligne du CTR (`services::ctr_backfill`)
    // reconstruit un profil pour chaque lecteur historique avec cette même
    // méthode — celle qui alimente déjà chaque recommandation servie.
    pub(crate) async fn build_user_profile(&self, user_id: &str) -> Result<UserProfile> {
        let cache_key = format!("twitninf:profile:{}", user_id);
        if let Some(mut cached) = self.cache.get_profile(&cache_key).await {
            debug!(user_id, "Profile loaded from cache");
            // Les index d'appartenance ne sont pas sérialisés (`serde(skip)`) :
            // un profil relu du cache arrive avec des ensembles VIDES, et
            // `follows()` répondrait alors « non » pour tout le monde — le
            // boost d'abonnement et D3 tomberaient silencieusement à zéro pour
            // tous les lecteurs dont le profil est en cache, c'est-à-dire la
            // quasi-totalité. Reconstruire ici n'est pas une optimisation,
            // c'est la condition de justesse du montage.
            cached.rebuild_indexes();
            return Ok(cached);
        }

        debug!(user_id, "Building user profile from database...");
        let client = self.pg.get().await?;
        let uid = uuid::Uuid::parse_str(user_id)?;

        // Pipeline : tokio-postgres multiplexe plusieurs requêtes concurrentes sur
        // une même connexion. On lance les requêtes de profil en parallèle via
        // join! au lieu de les enchaîner — la latence devient celle de la plus
        // lente, pas la somme. Toutes sont paramétrées ($1), plus construites
        // par `format!`.
        //
        // `status = 'active'` est décisif : sans ce filtre, un compte bloqué ou
        // mis en sourdine restait compté comme « suivi », donc boosté de +30 %
        // en mode Feed et de +0.55 en D3. On se retrouvait à pousser en tête de
        // fil le contenu des comptes que l'utilisateur avait explicitement
        // écartés.
        const SQL_SOCIAL: &str = "SELECT following_id::text, EXISTS(SELECT 1 FROM user_follows f2 WHERE f2.follower_id = user_follows.following_id AND f2.following_id = $1 AND f2.status = 'active') AS is_mutual FROM user_follows WHERE follower_id = $1 AND status = 'active' LIMIT 1000";
        const SQL_ENGAGEMENT: &str = "SELECT SUM(CASE WHEN created_at > NOW() - INTERVAL '1 day' THEN 1 ELSE 0 END) AS daily, SUM(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END) AS weekly FROM tweet_likes WHERE user_id = $1";
        const SQL_TEMPORAL: &str = "SELECT EXTRACT(HOUR FROM created_at)::int AS h, EXTRACT(DOW FROM created_at)::int AS d, COUNT(*) AS cnt FROM tweet_likes WHERE user_id = $1 AND created_at > NOW() - INTERVAL '60 days' GROUP BY h, d ORDER BY h, d";
        const SQL_CONTENT: &str = "SELECT AVG(LENGTH(t.content))::float8 AS avg_len, SUM(CASE WHEN t.media_urls IS NOT NULL AND t.media_urls != '[]'::jsonb THEN 1 ELSE 0 END)::float8 / GREATEST(COUNT(*), 1) AS media_ratio, AVG(COALESCE(jsonb_array_length(t.hashtags), 0))::float8 AS avg_hashtags FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id WHERE tl.user_id = $1 AND tl.created_at > NOW() - INTERVAL '30 days'";
        const SQL_BEHAVIOR: &str = "SELECT (SELECT COUNT(*) FROM tweet_retweets WHERE user_id = $1) AS rt_count, (SELECT COUNT(*) FROM tweet_likes WHERE user_id = $1) AS like_count, (SELECT COUNT(*) FROM tweets WHERE user_id = $1 AND parent_tweet_id IS NOT NULL) AS reply_count, (SELECT COUNT(*) FROM user_follows WHERE following_id = $1 AND status = 'active') AS followers, (SELECT COUNT(*) FROM user_follows WHERE follower_id = $1 AND status = 'active') AS following";
        // Affinité par auteur : avant, UNIQUEMENT les likes — un lecteur qui
        // lit beaucoup et like peu (profil majoritaire) restait invisible
        // pour ce classement. `user_behavior_data` porte déjà le dwell réel
        // (`action_type='time_spent'`, écrit par `mirrorDwell` côté API,
        // voir la passation NeuralRank §3.7b) : unioné ici, ramené en
        // équivalent-minutes pour rester du même ordre de grandeur qu'un
        // compte de likes plutôt que de le noyer sous des millisecondes.
        // `LEAST(...,600000)` : un événement peut remonter jusqu'à ~10h côté
        // client (app restée ouverte en arrière-plan) — piège déjà payé sur
        // l'algo Scout, plafonné à la même valeur qu'à l'écriture
        // (`dwellMirror.js::DWELL_CAP_MS`) en défense contre les lignes
        // antérieures à ce plafond.
        const SQL_AFFINITY: &str = "SELECT author_id, SUM(affinity)::float8 AS affinity FROM ( \
            SELECT t.user_id::text AS author_id, COUNT(*)::float8 AS affinity \
            FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id \
            WHERE tl.user_id = $1 AND tl.created_at > NOW() - INTERVAL '60 days' \
            GROUP BY t.user_id \
            UNION ALL \
            SELECT t.user_id::text AS author_id, \
                   (SUM(LEAST(COALESCE((b.context_data->>'time_spent_ms')::bigint, 0), 600000))::float8 / 60000.0) AS affinity \
            FROM user_behavior_data b JOIN tweets t ON t.id::text = b.target_id \
            WHERE b.user_id = $1 AND b.action_type = 'time_spent' AND b.target_type = 'tweet' \
              AND COALESCE(b.is_data_test, false) = false AND t.user_id <> $1 \
              AND b.timestamp > NOW() - INTERVAL '60 days' \
            GROUP BY t.user_id \
        ) combined GROUP BY author_id ORDER BY affinity DESC LIMIT 20";
        const SQL_SEEN: &str = "SELECT tweet_id::text FROM tweet_likes WHERE user_id = $1 ORDER BY created_at DESC LIMIT 500";
        const SQL_RETWEETED: &str = "SELECT tweet_id::text FROM tweet_retweets WHERE user_id = $1 ORDER BY created_at DESC LIMIT 300";
        // Contenu réellement aimé — sert à reconstruire les centres d'intérêt
        // (`top_words`) et le style préféré, deux champs que le profil laissait
        // vides et que D2 comme D8 lisent pourtant.
        const SQL_LIKED_TEXT: &str = "SELECT COALESCE(t.content, '') FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id WHERE tl.user_id = $1 AND tl.created_at > NOW() - INTERVAL '90 days' ORDER BY tl.created_at DESC LIMIT 200";
        const SQL_SECOND_DEGREE: &str = "SELECT DISTINCT f2.following_id::text FROM user_follows f JOIN user_follows f2 ON f2.follower_id = f.following_id AND f2.status = 'active' WHERE f.follower_id = $1 AND f.status = 'active' AND f2.following_id <> $1 AND f2.following_id NOT IN (SELECT following_id FROM user_follows WHERE follower_id = $1 AND status = 'active') LIMIT 200";
        // Comptes bloqués dans un sens ou l'autre — voir `UserProfile::blocked_ids`.
        const SQL_BLOCKED: &str = "SELECT following_id::text FROM user_follows WHERE follower_id = $1 AND status = 'blocked' UNION SELECT follower_id::text FROM user_follows WHERE following_id = $1 AND status = 'blocked'";

        // Lié à une variable : `&[&uid]` en argument direct crée un temporaire
        // détruit avant la fin du `join!`.
        let params: [&(dyn tokio_postgres::types::ToSql + Sync); 1] = [&uid];

        let (
            social_res,
            engagement_res,
            temporal_res,
            content_pref_res,
            behavior_res,
            author_affinity_res,
            seen_ids_res,
            retweeted_res,
            liked_text_res,
            second_degree_res,
            blocked_res,
        ) = join!(
            client.query(SQL_SOCIAL, &params),
            client.query(SQL_ENGAGEMENT, &params),
            client.query(SQL_TEMPORAL, &params),
            client.query(SQL_CONTENT, &params),
            client.query(SQL_BEHAVIOR, &params),
            client.query(SQL_AFFINITY, &params),
            client.query(SQL_SEEN, &params),
            client.query(SQL_RETWEETED, &params),
            client.query(SQL_LIKED_TEXT, &params),
            client.query(SQL_SECOND_DEGREE, &params),
            client.query(SQL_BLOCKED, &params),
        );

        let mut profile = UserProfile::default();
        profile.user_id = user_id.to_string();

        if let Ok(rows) = social_res {
            for row in &rows {
                let fid: String = row.try_get(0).unwrap_or_default();
                let is_mutual: bool = row.try_get(1).unwrap_or(false);
                if !fid.is_empty() {
                    profile.following_ids.push(fid.clone());
                    if is_mutual {
                        profile.mutual_follow_ids.push(fid);
                    }
                }
            }
            trace!(
                following = profile.following_ids.len(),
                mutual = profile.mutual_follow_ids.len(),
                "Social graph loaded"
            );
        }

        if let Ok(rows) = behavior_res {
            if let Some(row) = rows.first() {
                let like_count: i64 = row.try_get(1).unwrap_or(0);
                let follower_count: i64 = row.try_get(3).unwrap_or(0);
                let following_count: i64 = row.try_get(4).unwrap_or(0);

                profile.follower_count = follower_count;
                profile.following_count = following_count;
                profile.network_influence =
                    ((follower_count as f64).ln().max(0.0) * 10.0).min(100.0);

                profile.user_type = if like_count > 200 {
                    UserType::PowerUser
                } else if like_count > 30 {
                    UserType::Regular
                } else {
                    UserType::Casual
                };

                profile.profile_confidence = (0.3 + (like_count as f64 / 400.0).min(0.7)).min(1.0);
                profile.churn_risk = 0.2;
                trace!(like_count, follower_count, following_count, user_type = ?profile.user_type, "Behavior metrics loaded");
            }
        }

        if let Ok(rows) = engagement_res {
            if let Some(row) = rows.first() {
                let daily: i64 = row.try_get(0).unwrap_or(0);
                let weekly: i64 = row.try_get(1).unwrap_or(0);
                let weekly_per_day = weekly as f64 / 7.0;
                profile.engagement_velocity = daily as f64;
                profile.engagement_trend = if weekly_per_day > 0.0 {
                    daily as f64 / weekly_per_day
                } else {
                    1.0
                };
                trace!(
                    daily_engagement = daily,
                    engagement_trend = profile.engagement_trend,
                    "Engagement metrics loaded"
                );
            }
        }

        if let Ok(rows) = temporal_res {
            let mut hourly = [0.0_f64; 24];
            let mut daily = [0.0_f64; 7];
            for row in &rows {
                let h: i32 = row.try_get(0).unwrap_or(0);
                let d: i32 = row.try_get(1).unwrap_or(0);
                let cnt: i64 = row.try_get(2).unwrap_or(0);
                if (0..24).contains(&h) {
                    hourly[h as usize] += cnt as f64;
                }
                if (0..7).contains(&d) {
                    daily[d as usize] += cnt as f64;
                }
            }
            let h_max = hourly
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1.0);
            let d_max = daily
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1.0);
            for i in 0..24 {
                profile.hourly_activity[i] = hourly[i] / h_max;
            }
            for i in 0..7 {
                profile.daily_activity[i] = daily[i] / d_max;
            }
            profile.most_active_hour = hourly
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(12) as u32;
            trace!(
                most_active_hour = profile.most_active_hour,
                "Temporal activity patterns loaded"
            );
        }

        if let Ok(rows) = content_pref_res {
            if let Some(row) = rows.first() {
                let avg_len: f64 = row.try_get(0).unwrap_or(100.0);
                let media_ratio: f64 = row.try_get(1).unwrap_or(0.3);
                let avg_ht: f64 = row.try_get(2).unwrap_or(1.0);
                profile.avg_content_length = avg_len;
                profile.prefers_media = media_ratio > 0.35;
                profile.avg_hashtag_count = avg_ht;
                profile.preferred_content_length = if avg_len < 80.0 {
                    ContentLength::Short
                } else if avg_len > 200.0 {
                    ContentLength::Long
                } else {
                    ContentLength::Medium
                };
                trace!(avg_content_length = avg_len, media_ratio, content_preference = ?profile.preferred_content_length, "Content preferences loaded");
            }
        }

        if let Ok(rows) = author_affinity_res {
            let max_aff = rows
                .first()
                .and_then(|r| r.try_get::<_, f64>(1).ok())
                .unwrap_or(1.0)
                .max(1.0);
            profile.top_authors = rows
                .iter()
                .filter_map(|r| {
                    let uid: String = r.try_get(0).ok()?;
                    let aff: f64 = r.try_get(1).ok()?;
                    Some((uid, aff / max_aff))
                })
                .collect();
            trace!(
                top_authors_count = profile.top_authors.len(),
                "Top authors affinity loaded"
            );
        }

        if let Ok(rows) = seen_ids_res {
            profile.liked_tweet_ids = rows
                .iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            trace!(
                liked_tweets_count = profile.liked_tweet_ids.len(),
                "Liked tweet history loaded"
            );
        }
        profile.seen_tweet_ids = self.cache.get_seen_tweet_ids(user_id).await;
        profile.damped_authors = self.cache.get_damped_authors(user_id).await;
        // Absent du `join!` ci-dessus : ne coûte qu'un aller-retour de plus sur
        // un profil déjà mis en cache 300s, et une panne ici (modèle
        // désactivé, requête échouée) ne doit pas faire échouer tout le reste
        // du profil — juste laisser `taste_vector` à `None`.
        match crate::embeddings::user_taste_vector(&self.pg, user_id).await {
            Ok(v) => profile.taste_vector = v,
            Err(e) => debug!(user_id, error = %e, "Vecteur de goût indisponible"),
        }
        // Vecteur de goût EXPLICITE (voir `crate::calibration`), distinct du
        // vecteur naturel ci-dessus : un compte qui vient de recalibrer
        // volontairement sur 5 tours a dit quelque chose de plus net que
        // trois mois de likes dispersés dans l'activité normale, donc pèse
        // plus quand les deux existent (`calibration::blend_taste`).
        if let Some(calib) = self.cache.calibration_load_taste(user_id).await {
            profile.taste_vector = Some(match profile.taste_vector.take() {
                Some(natural) => crate::calibration::blend_taste(&calib, &natural),
                None => calib,
            });
        }
        if !profile.damped_authors.is_empty() {
            debug!(
                user_id,
                authors = profile.damped_authors.len(),
                "Damped authors loaded"
            );
        }

        if let Ok(rows) = retweeted_res {
            // Sans ces ids, `profile_retweet_rate` valait 0/N = 0 dans D5 et le
            // bonus « prédiction de partage » ne se déclenchait jamais.
            profile.retweeted_tweet_ids = rows
                .iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            trace!(
                retweeted = profile.retweeted_tweet_ids.len(),
                "Retweet history loaded"
            );
        }

        if let Ok(rows) = liked_text_res {
            let texts: Vec<String> = rows
                .iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            let (words, personality, positivity) = profile_from_liked_text(&texts);
            profile.top_words = words;
            profile.personality_type = personality;
            profile.emotional_positivity = positivity;
            trace!(top_words = profile.top_words.len(), positivity,
                   personality = ?profile.personality_type, "Interests derived from liked content");
        }

        if let Ok(rows) = second_degree_res {
            profile.second_degree_ids = rows
                .iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            trace!(
                second_degree_count = profile.second_degree_ids.len(),
                "Second degree network loaded"
            );
        }

        if let Ok(rows) = blocked_res {
            profile.blocked_ids = rows
                .iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            trace!(blocked_count = profile.blocked_ids.len(), "Blocked accounts loaded");
        }

        // Index d'appartenance : construits une fois ici, relus des milliers de
        // fois pendant le scoring. Voir `UserProfile::rebuild_indexes`.
        profile.rebuild_indexes();

        debug!(
            profile_confidence = profile.profile_confidence,
            following_indexed = profile.following_set.len(),
            seen_indexed = profile.seen_set.len(),
            "User profile built and cached"
        );
        self.cache.set_profile(&cache_key, &profile).await;
        Ok(profile)
    }

    /// Collecte les candidats des 8 sources en UNE requête paramétrée.
    ///
    /// L'ancienne version envoyait 8 requêtes construites par `format!`, chacune
    /// répétant 5 sous-requêtes corrélées par ligne (jusqu'à ~1700 lignes), et
    /// ne calculait l'engagement récent que pour 2 sources sur 8 — le même tweet
    /// obtenait donc un D1 différent selon la source qui l'avait trouvé. Ici les
    /// sources ne sélectionnent que des ids, la déduplication se fait en SQL, et
    /// les métriques sont calculées une seule fois par candidat unique.
    async fn collect_candidates(
        &self,
        user_id: &str,
        profile: &UserProfile,
        mode: &RecommendMode,
        banned_set: &std::collections::HashSet<String>,
    ) -> Result<(Vec<RawTweet>, SourceStats)> {
        // ⚠ La fenêtre DÉCOUVERTE est volontairement bien plus large que les
        // autres. C'est la seule source qui tire au hasard parmi les auteurs
        // NON suivis : c'est elle, et elle seule, qui fait entrer des comptes
        // nouveaux dans le vivier.
        //
        // Elle valait 96 h, ce qui paraît généreux mais ne l'est pas du tout à
        // l'échelle réelle de la plateforme. Relevé en prod le 2026-07-29 :
        //   24 h →  7 auteurs ·  72 h → 10 auteurs
        //    7 j → 16 auteurs ·  30 j → 69 auteurs
        // Le vivier ne contenait donc que ~10 auteurs, dont un qui publie
        // presque la moitié du corpus — et le fil servi comptait 37 tweets du
        // même compte sur 50, avec des séries de 38 d'affilée. Aucun réglage de
        // score ni plafond de mise en page ne peut corriger ça : on ne
        // diversifie pas un vivier qui ne contient personne d'autre.
        //
        // Les fenêtres trending / social / viral restent courtes : c'est ce qui
        // garde le haut du fil frais. Seule la découverte remonte loin, et son
        // poids (0.05) fait qu'elle apporte de la variété sans noyer l'actualité.
        let (window_trending, window_social, window_discover, window_viral): (i32, i32, i32, i32) =
            match mode {
                RecommendMode::Trending => (6, 24, 24 * 7, 3),
                RecommendMode::Feed => (12, 72, 24 * 14, 6),
                RecommendMode::Discover => (24, 48, 24 * 30, 12),
                RecommendMode::ForYou => (72, 72, 24 * 30, 24),
            };
        // Horizon global : aucune source ne remonte plus loin, le CTE `visible`
        // n'a donc pas à balayer toute la table.
        let horizon = window_trending
            .max(window_social)
            .max(window_discover)
            .max(window_viral)
            .max(24 * 7);
        debug!(mode = ?mode, trending_window = window_trending, social_window = window_social,
               discover_window = window_discover, viral_window = window_viral, "Collecting candidates with time windows");

        let active_hour = profile.most_active_hour as i32;

        // Les listes d'ids partent en paramètres `uuid[]`, plus jamais
        // concaténées dans le SQL : une liste vide donne `= ANY('{}')` qui vaut
        // simplement `false`, sans sentinelle bidon ni branche spéciale.
        let to_uuids = |ids: &[String]| -> Vec<uuid::Uuid> {
            ids.iter()
                .filter_map(|id| uuid::Uuid::parse_str(id).ok())
                .collect()
        };
        let uid = uuid::Uuid::parse_str(user_id)?;
        let following_uuids = to_uuids(&profile.following_ids);
        let top_author_uuids = to_uuids(
            &profile
                .top_authors
                .iter()
                .take(10)
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>(),
        );
        let banned_uuids: Vec<uuid::Uuid> = banned_set
            .iter()
            .filter_map(|id| uuid::Uuid::parse_str(id).ok())
            .collect();

        let client = self.pg.get().await?;

        // ── Fenêtres élargies si le vivier est trop maigre ───────────────────
        // Les fenêtres courtes de Trending (6 h) supposent un flux de
        // publication soutenu. Quand il ne l'est pas, elles ne contiennent
        // presque rien : la page de découverte affiche la même poignée de
        // tweets toute la journée, et il n'y a aucune raison d'y revenir. On
        // retente alors UNE fois, plus large — un aller de plus en base
        // seulement dans ce cas, jamais sur le chemin nominal.
        //
        // Les autres modes n'ont qu'une tentative : leurs fenêtres sont déjà
        // larges, et c'est Trending qui porte la page de découverte.
        let mut attempts: Vec<(i32, i32, i32, i32)> = vec![(
            window_trending,
            window_social,
            window_discover,
            window_viral,
        )];
        if *mode == RecommendMode::Trending {
            attempts.push((
                window_trending * TRENDING_WIDEN_FACTOR,
                window_social * TRENDING_WIDEN_FACTOR,
                window_discover,
                window_viral * TRENDING_WIDEN_FACTOR,
            ));
        }

        let mut all = Vec::new();
        for (attempt, (wt, ws, wd, wv)) in attempts.into_iter().enumerate() {
            let attempt_horizon = wt.max(ws).max(wd).max(wv).max(horizon);
            let rows = client
                .query(
                    CANDIDATES_SQL.as_str(),
                    &[
                        &uid,                       // $1
                        &following_uuids,           // $2
                        &top_author_uuids,          // $3
                        &banned_uuids,              // $4
                        &wt,                        // $5
                        &ws,                        // $6
                        &wd,                        // $7
                        &wv,                        // $8
                        &attempt_horizon,           // $9
                        &active_hour,               // $10
                        &MAX_CANDIDATES_PER_AUTHOR, // $11
                    ],
                )
                .await?;

            all = map_rows(rows);
            if all.len() >= TRENDING_MIN_POOL {
                break;
            }
            debug!(
                attempt,
                candidates = all.len(),
                floor = TRENDING_MIN_POOL,
                "Candidate pool below floor"
            );
        }

        let mut stats = SourceStats::default();
        for tweet in &all {
            match tweet.source {
                TweetSource::Trending => stats.trending += 1,
                TweetSource::SocialGraph => stats.social_graph += 1,
                TweetSource::Viral => stats.viral += 1,
                TweetSource::Discovery => stats.discovery += 1,
                TweetSource::Temporal => stats.temporal += 1,
                TweetSource::Influencer => stats.influencer += 1,
                TweetSource::Personalized => stats.personalized += 1,
                TweetSource::Quality => stats.quality += 1,
            }
        }
        stats.deduplicated_total = all.len();

        debug!(
            candidates = all.len(),
            trending = stats.trending,
            social_graph = stats.social_graph,
            viral = stats.viral,
            discovery = stats.discovery,
            temporal = stats.temporal,
            influencer = stats.influencer,
            personalized = stats.personalized,
            quality = stats.quality,
            "Candidates collected (single parameterized query, deduped in SQL)"
        );

        Ok((all, stats))
    }

    /// Remonte les parents manquants des réponses candidates.
    ///
    /// Sans cette passe, `shape_feed` écartait la quasi-totalité des réponses :
    /// il exige que toute la chaîne d'ancêtres soit dans le vivier, or aucune
    /// des 8 sources ne collecte un tweet parce qu'il est le parent d'un autre.
    /// Une réponse n'était donc servie que si son parent avait été retenu par
    /// ailleurs, par pure coïncidence. Le fil de discussion existait dans le
    /// code sans jamais atteindre l'écran.
    ///
    /// La remontée est itérative — le parent d'une réponse peut lui-même être
    /// une réponse — et bornée par `MAX_THREAD_DEPTH`, la même profondeur que
    /// celle que `shape_feed` accepte d'afficher : remonter plus loin
    /// ramènerait des tweets qu'il écarterait de toute façon.
    async fn hydrate_thread_parents(
        &self,
        user_id: &str,
        tweets: &mut Vec<RawTweet>,
        banned_set: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let uid = uuid::Uuid::parse_str(user_id)?;
        let banned_uuids: Vec<uuid::Uuid> = banned_set
            .iter()
            .filter_map(|id| uuid::Uuid::parse_str(id).ok())
            .collect();

        let mut known: HashSet<String> = tweets.iter().map(|t| t.id.clone()).collect();
        let mut added = 0usize;

        for _ in 0..MAX_THREAD_DEPTH {
            let missing: Vec<uuid::Uuid> = tweets
                .iter()
                .filter_map(|t| t.parent_tweet_id.as_deref())
                .filter(|id| !known.contains(*id))
                .filter_map(|id| uuid::Uuid::parse_str(id).ok())
                .collect::<HashSet<_>>() // un même parent peut manquer à plusieurs réponses
                .into_iter()
                .collect();
            if missing.is_empty() {
                break;
            }

            let client = self.pg.get().await?;
            let rows = client
                .query(PARENTS_SQL.as_str(), &[&uid, &missing, &banned_uuids])
                .await?;
            let parents = map_rows(rows);
            if parents.is_empty() {
                // Les parents restants sont invisibles pour ce lecteur.
                // `shape_feed` écartera leurs réponses, ce qui est le
                // comportement voulu : insister ne ferait que reposer la même
                // question à la base.
                break;
            }

            // Les identifiants demandés qui n'ont RIEN renvoyé sont marqués
            // connus malgré tout : sans ça, la passe suivante les redemanderait
            // à l'identique jusqu'à épuiser les tours.
            for id in &missing {
                known.insert(id.to_string());
            }
            for parent in parents {
                known.insert(parent.id.clone());
                tweets.push(parent);
                added += 1;
            }
        }

        if added > 0 {
            debug!(
                parents_added = added,
                "Parents de fil remontés pour rendre les réponses lisibles"
            );
        }
        Ok(added)
    }

    /// Candidats trouvés par similarité de CONTENU avec le vecteur de goût du
    /// lecteur — voir `crate::embeddings`. Absent des 8 sources SQL de
    /// `collect_candidates`, qui ne comparent que récence/popularité/follow,
    /// jamais ce dont un tweet parle réellement.
    ///
    /// `None`/erreur silencieuse à chaque étape (pas de vecteur de goût, modèle
    /// désactivé, requête échouée) : cette source est un COMPLÉMENT, jamais un
    /// prérequis — un compte neuf ou un moteur sans embeddings doit continuer
    /// à recevoir un feed normal par les 8 autres sources.
    async fn hydrate_semantic_candidates(
        &self,
        user_id: &str,
        profile: &UserProfile,
        tweets: &mut Vec<RawTweet>,
        banned_set: &std::collections::HashSet<String>,
    ) -> usize {
        let Some(taste) = profile.taste_vector.as_ref() else {
            return 0;
        };

        let ids = match crate::embeddings::nearest_tweets(&self.pg, taste, user_id, 40).await {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => return 0,
            Err(e) => {
                debug!(user_id, error = %e, "Candidats sémantiques indisponibles");
                return 0;
            }
        };

        let known: HashSet<String> = tweets.iter().map(|t| t.id.clone()).collect();
        let missing: Vec<uuid::Uuid> = ids
            .into_iter()
            .filter(|id| !known.contains(id))
            .filter_map(|id| uuid::Uuid::parse_str(&id).ok())
            .collect();
        if missing.is_empty() {
            return 0;
        }

        let Ok(uid) = uuid::Uuid::parse_str(user_id) else {
            return 0;
        };
        let banned_uuids: Vec<uuid::Uuid> = banned_set
            .iter()
            .filter_map(|id| uuid::Uuid::parse_str(id).ok())
            .collect();

        let client = match self.pg.get().await {
            Ok(c) => c,
            Err(_) => return 0,
        };
        let rows = match client
            .query(SEMANTIC_SQL.as_str(), &[&uid, &missing, &banned_uuids])
            .await
        {
            Ok(r) => r,
            Err(e) => {
                debug!(user_id, error = %e, "Hydratation des candidats sémantiques échouée");
                return 0;
            }
        };
        let added = map_rows(rows);
        let count = added.len();
        if count > 0 {
            debug!(user_id, count, "Candidats sémantiques ajoutés");
        }
        tweets.extend(added);
        count
    }

    /// Candidats trouvés par co-occurrence — voir `crate::cooccurrence`.
    /// Même prudence que la source sémantique : un compte qui n'a encore
    /// aucun auteur favori (`top_authors` vide, compte tout neuf) obtient
    /// simplement zéro candidat de cette source, jamais une erreur.
    async fn hydrate_cooccurrence_candidates(
        &self,
        profile: &UserProfile,
        tweets: &mut Vec<RawTweet>,
        banned_set: &std::collections::HashSet<String>,
    ) -> usize {
        if profile.top_authors.is_empty() {
            return 0;
        }
        let seeds: Vec<String> = profile
            .top_authors
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        let co_authors = self.cache.co_liked_authors(&seeds, 15).await;
        if co_authors.is_empty() {
            return 0;
        }

        let known: HashSet<String> = tweets.iter().map(|t| t.id.clone()).collect();
        let author_uuids: Vec<uuid::Uuid> = co_authors
            .iter()
            .filter_map(|id| uuid::Uuid::parse_str(id).ok())
            .collect();
        if author_uuids.is_empty() {
            return 0;
        }

        let Ok(uid) = uuid::Uuid::parse_str(&profile.user_id) else {
            return 0;
        };
        let banned_uuids: Vec<uuid::Uuid> = banned_set
            .iter()
            .filter_map(|id| uuid::Uuid::parse_str(id).ok())
            .collect();

        let client = match self.pg.get().await {
            Ok(c) => c,
            Err(_) => return 0,
        };
        let rows = match client
            .query(COOCCUR_SQL.as_str(), &[&uid, &author_uuids, &banned_uuids])
            .await
        {
            Ok(r) => r,
            Err(e) => {
                debug!(user_id = %profile.user_id, error = %e, "Hydratation par co-occurrence échouée");
                return 0;
            }
        };
        let added: Vec<RawTweet> = map_rows(rows)
            .into_iter()
            .filter(|t| !known.contains(&t.id))
            .collect();
        let count = added.len();
        if count > 0 {
            debug!(user_id = %profile.user_id, count, "Candidats par co-occurrence ajoutés");
        }
        tweets.extend(added);
        count
    }

    async fn count_available(&self, user_id: &str) -> Result<i64> {
        let client = self.pg.get().await?;
        let uid = uuid::Uuid::parse_str(user_id)?;
        // Aligné sur les filtres de `visible` : compter des tweets privés ou de
        // comptes suspendus faussait `has_more`, qui promettait des pages
        // supplémentaires que la pagination ne pouvait pas servir.
        let row = client
            .query_one(
                "SELECT COUNT(*) FROM tweets t JOIN users u ON u.id = t.user_id \
             WHERE t.deleted_at IS NULL AND t.moderation_status = 'approved' \
               AND t.is_private = false AND COALESCE(t.is_data_test, false) = false \
               AND u.is_active = true AND COALESCE(u.is_suspended, false) = false \
               AND t.user_id <> $1",
                &[&uid],
            )
            .await?;
        Ok(row.get(0))
    }

    fn build_empty_response(
        &self,
        user_id: &str,
        tweet_ids: Vec<String>,
        count: usize,
        mode: &str,
        latency_ms: u64,
        cache_hit: bool,
    ) -> RecommendResponse {
        RecommendResponse {
            success: true,
            user_id: user_id.to_string(),
            tweet_ids,
            threads: Vec::new(),
            scores: Vec::new(),
            count,
            algorithm: "NeuralRank Fusion",
            algorithm_version: "2.2.0 — 8 dimensions + ML CTR + bandit + adaptive A/B",
            mode: mode.to_string(),
            latency_ms,
            cache_hit,
            experiments: Vec::new(),
            ads: Vec::new(),
            metadata: RecommendMetadata {
                candidates_collected: 0,
                sources: SourceStats::default(),
                user_profile: UserProfileSummary {
                    user_type: "cached".into(),
                    confidence: 1.0,
                    personality: "cached".into(),
                    engagement_velocity: 0.0,
                    engagement_trend: "cached".into(),
                    network_influence: 0.0,
                    most_active_hour: 12,
                    churn_risk: 0.0,
                },
                quality_metrics: QualityMetrics {
                    diversity_score: 0.0,
                    freshness_score: 0.0,
                    relevance_score: 0.0,
                    viral_potential: 0.0,
                    novelty_score: 0.0,
                },
                pagination: Pagination {
                    limit: 50,
                    offset: 0,
                    has_more: false,
                    total_available: 0,
                },
            },
        }
    }
}

// ─── SQL constants ────────────────────────────────────────────────────────────

/// Sélection des candidats : 8 sources, dédupliquées et plafonnées par auteur.
///
/// Se termine sur le CTE `picked (id, src, w)` que `PROJECTION_SQL` vient lire.
/// La coupure est là, et pas ailleurs, parce que c'est le seul point où la
/// requête de collecte et celle d'hydratation des parents se rejoignent : les
/// deux produisent un `picked`, les deux l'habillent du même projection.
///
/// Paramètres :
///   $1 user_id · $2 following[] · $3 top_authors[] · $4 hard_banned[]
///   $5 fenêtre trending (h) · $6 social (h) · $7 discovery (h) · $8 viral (h)
///   $9 horizon global (h) · $10 heure d'activité de l'utilisateur
///   $11 plafond de candidats par auteur (voir `MAX_CANDIDATES_PER_AUTHOR`)
const CANDIDATES_CTE: &str = r#"
WITH visible AS (
    SELECT t.id, t.user_id, t.created_at, u.verified, u.premium
    FROM tweets t
    JOIN users u ON u.id = t.user_id
    WHERE t.deleted_at IS NULL
      AND t.moderation_status = 'approved'
      -- Un tweet privé, de compte désactivé ou suspendu était collecté puis
      -- jeté par la couche Node : autant de places de feed gaspillées.
      AND t.is_private = false
      AND COALESCE(t.is_data_test, false) = false
      AND u.is_active = true
      AND COALESCE(u.is_suspended, false) = false
      AND t.user_id <> $1
      AND NOT (t.user_id = ANY($4))
      AND t.created_at > NOW() - make_interval(hours => $9)
      -- Un retweet pur sans original affichable ne rend qu'une carte vide.
      AND (
        COALESCE(t.content, '') <> ''
        OR (t.media_urls IS NOT NULL AND t.media_urls <> '[]'::jsonb)
        -- Un message vocal se suffit à lui-même : le composeur comme l'API
        -- acceptent un tweet sans texte ni média dès lors qu'il en porte un.
        -- Sans cette ligne, ce tweet-là n'entrait jamais dans le pool de
        -- candidats — publié, visible sur le profil, absent du fil.
        OR t.audio_url IS NOT NULL
        OR EXISTS (
             SELECT 1 FROM tweets o
             WHERE o.id = t.original_tweet_id
               AND o.deleted_at IS NULL
               AND o.moderation_status = 'approved'
               AND o.is_private = false
           )
      )
      -- Une réponse doit avoir au moins un like pour entrer dans le pool.
      -- Sans engagement, elle n'apporte rien hors de son fil : le lecteur voit
      -- un bout de conversation sans le contexte. L'exception « auteur suivi »
      -- a été retirée : suivre quelqu'un ne rend pas sa réponse vide
      -- intéressante, et c'est par là que passait l'essentiel du bruit.
      AND (
        t.parent_tweet_id IS NULL
        OR (SELECT COUNT(*) FROM tweet_likes WHERE tweet_id = t.id) >= 1
      )
      -- Son parent doit être affichable POUR CE LECTEUR : il sera injecté
      -- au-dessus d'elle dans le feed, donc il doit passer exactement les mêmes
      -- contrôles que n'importe quel candidat. Sans la vérification sur
      -- l'auteur, répondre à un compte bloqué ou suspendu aurait suffi à faire
      -- réapparaître son tweet dans le fil par la porte de derrière.
      AND (
        t.parent_tweet_id IS NULL
        OR EXISTS (
             SELECT 1 FROM tweets p
             JOIN users pu ON pu.id = p.user_id
             WHERE p.id = t.parent_tweet_id
               AND p.deleted_at IS NULL
               AND p.moderation_status = 'approved'
               AND p.is_private = false
               AND pu.is_active = true
               AND COALESCE(pu.is_suspended, false) = false
               AND NOT (p.user_id = ANY($4))
           )
      )
),
cand AS (
        (SELECT id, 1 AS src, 0.15::float8 AS w FROM visible v
          WHERE v.created_at > NOW() - make_interval(hours => $5)
          ORDER BY (SELECT COUNT(*) FROM tweet_likes l
                     WHERE l.tweet_id = v.id AND l.created_at > NOW() - INTERVAL '1 hour') DESC,
                   v.created_at DESC
          LIMIT 400)
  UNION ALL
        (SELECT id, 2, 0.12 FROM visible v
          WHERE v.user_id = ANY($2) AND v.created_at > NOW() - make_interval(hours => $6)
          ORDER BY v.created_at DESC LIMIT 300)
  UNION ALL
        (SELECT id, 3, 0.08 FROM visible v
          WHERE v.created_at > NOW() - make_interval(hours => $8)
          ORDER BY (SELECT COUNT(*) FROM tweet_likes l
                     WHERE l.tweet_id = v.id AND l.created_at > NOW() - INTERVAL '6 hours') DESC
          LIMIT 250)
  UNION ALL
        -- Borner AVANT de mélanger, pas après : `ORDER BY RANDOM()` trie
        -- l'intégralité des lignes qui passent le WHERE, sans qu'aucun index
        -- puisse l'aider — son coût suit le volume de tweets récents de TOUTE
        -- la plateforme, pas la taille du résultat qu'on en tire. À 40
        -- lecteurs ça ne se voit pas ; au premier vrai pic de publication ça
        -- devient le poste le plus cher de toute la requête, sur CHAQUE appel
        -- de feed. Le sous-select prend d'abord les 2000 plus récents (même
        -- index que `v.created_at DESC` utilisé partout ailleurs ici), et
        -- c'est seulement ce lot borné que `RANDOM()` mélange : le coût du tri
        -- aléatoire reste constant, quelle que soit l'échelle de la
        -- plateforme. En dessous de 2000 lignes candidates, le comportement
        -- est identique à l'ancienne requête.
        (SELECT id, 4, 0.05 FROM (
              SELECT v.id FROM visible v
              WHERE NOT (v.user_id = ANY($2))
                AND v.created_at > NOW() - make_interval(hours => $7)
              ORDER BY v.created_at DESC
              LIMIT 2000
            ) discovery_pool
          ORDER BY RANDOM() LIMIT 150)
  UNION ALL
        (SELECT id, 5, 0.06 FROM visible v
          WHERE v.created_at > NOW() - INTERVAL '72 hours'
            AND EXTRACT(HOUR FROM v.created_at) BETWEEN GREATEST($10 - 1, 0) AND LEAST($10 + 1, 23)
          ORDER BY v.created_at DESC LIMIT 150)
  UNION ALL
        (SELECT id, 6, 0.04 FROM visible v
          WHERE v.created_at > NOW() - INTERVAL '48 hours'
            AND (v.verified = true OR v.premium = true)
          ORDER BY v.created_at DESC LIMIT 150)
  UNION ALL
        (SELECT id, 7, 0.10 FROM visible v
          WHERE v.user_id = ANY($3) AND v.created_at > NOW() - INTERVAL '7 days'
          ORDER BY v.created_at DESC LIMIT 200)
  UNION ALL
        -- « Qualité » sélectionne désormais sur le taux d'engagement réel.
        -- L'ancienne version reprenait le filtre verified/premium de la source
        -- influenceur : deux sources sur huit renvoyaient les mêmes lignes.
        (SELECT id, 8, 0.02 FROM visible v
          WHERE v.created_at > NOW() - INTERVAL '72 hours'
          ORDER BY (SELECT COUNT(*) FROM tweet_likes l WHERE l.tweet_id = v.id)::float8
                 / GREATEST((SELECT COALESCE(view_count, 0) FROM tweets x WHERE x.id = v.id), 10)::float8 DESC
          LIMIT 100)
),
merged AS (
    SELECT id, MAX(w) AS w, (ARRAY_AGG(src ORDER BY w DESC))[1] AS src
    FROM cand GROUP BY id
),
-- ⚠ Plafond par auteur SUR LE VIVIER, toutes sources confondues.
--
-- Sans lui, un compte qui publie beaucoup entrait des centaines de fois : la
-- source « social » (comptes suivis, LIMIT 300) suffit à elle seule à saturer
-- le vivier quand le lecteur suit un compte prolifique. Relevé en prod le
-- 2026-07-29 sur un lecteur qui suit le plus gros publieur : 32 tweets du même
-- auteur sur 50 servis, malgré la pénalité de diversité au score ET le plafond
-- de mise en page. Les deux arrivent trop tard — on ne diversifie pas une liste
-- où un auteur occupe déjà les deux tiers des candidats.
--
-- Le tri interne garde les MEILLEURS candidats de l'auteur (poids de source
-- décroissant, puis les plus récents) : on borne sa place, on ne dégrade pas ce
-- qu'il a de mieux.
picked AS (
    SELECT id, w, src FROM (
        SELECT m.id, m.w, m.src,
               ROW_NUMBER() OVER (
                   PARTITION BY t.user_id
                   ORDER BY m.w DESC, t.created_at DESC
               ) AS rn
        FROM merged m
        JOIN tweets t ON t.id = m.id
    ) ranked
    WHERE rn <= $11
)
"#;

/// Habillage commun : transforme un `picked (id, src, w)` en lignes `RawTweet`.
///
/// Partagé mot pour mot entre la collecte et l'hydratation des parents, pour
/// que `map_rows` puisse indexer les colonnes par position sans jamais se
/// demander de laquelle des deux requêtes vient la ligne. Toute colonne ajoutée
/// ici profite aux deux — et surtout, aucune ne peut n'en servir qu'une.
///
/// Paramètre lu : `$1` (le lecteur, pour ses impressions déjà servies).
const PROJECTION_SQL: &str = r#"
SELECT
    t.id::text, t.user_id::text,
    COALESCE(t.content, '') AS content,
    t.created_at,
    COALESCE(t.view_count, 0)::bigint AS view_count,
    lk.like_count, cm.comment_count, rt.retweet_count,
    COALESCE(rp.report_count, 0) AS report_count,
    (t.media_urls IS NOT NULL AND t.media_urls <> '[]'::jsonb) AS has_media,
    COALESCE(jsonb_array_length(t.hashtags), 0)::int AS hashtag_count,
    COALESCE(jsonb_array_length(t.mentions), 0)::int AS mention_count,
    LENGTH(COALESCE(t.content, ''))::int AS content_length,
    au.followers, au.following, au.tweet_count, au.age_days,
    COALESCE(u.verified, false) AS author_is_verified,
    COALESCE(u.premium, false)  AS author_is_premium,
    COALESCE(u.algorithmic_visibility_multiplier, 1.0)::float8 AS visibility_mult,
    COALESCE(t.moderation_status::text, 'pending') AS moderation_status,
    t.recommendation_group::text,
    lk.likes_1h, lk.likes_6h, cm.comments_1h, rt.retweets_1h,
    COALESCE(bh.impressions, 0) AS impressions,
    p.src, p.w,
    t.parent_tweet_id::text, t.original_tweet_id::text, COALESCE(t.is_retweet, false),
    -- Étiquettes de l'annotateur LLM. LEFT JOIN volontaire : un tweet non
    -- encore annoté doit rester servable, D9 se met alors en neutre.
    ll.theme, ll.toxicity_score::float8, ll.toxicity_category,
    ll.quality_score::float8, ll.tone, ll.confidence::float8,
    -- `subscription_tier` est un ENUM Postgres : le cast texte est obligatoire,
    -- tokio-postgres ne sait pas décoder un type enum applicatif.
    COALESCE(u.subscription_tier::text, 'free') AS author_tier
FROM picked p
JOIN tweets t ON t.id = p.id
JOIN users  u ON u.id = t.user_id
LEFT JOIN LATERAL (
    SELECT COUNT(*)::bigint AS like_count,
           COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour')::bigint  AS likes_1h,
           COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours')::bigint AS likes_6h
    FROM tweet_likes WHERE tweet_id = t.id
) lk ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::bigint AS comment_count,
           COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour')::bigint AS comments_1h
    FROM tweets r WHERE r.parent_tweet_id = t.id AND r.deleted_at IS NULL
) cm ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::bigint AS retweet_count,
           COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour')::bigint AS retweets_1h
    FROM tweet_retweets WHERE tweet_id = t.id
) rt ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::bigint AS report_count
    FROM reports WHERE target_type = 'tweet' AND target_id = t.id
) rp ON true
LEFT JOIN LATERAL (
    -- Impressions déjà servies à CE lecteur pour CE tweet.
    SELECT COUNT(*)::bigint AS impressions
    FROM user_behavior_data b
    WHERE b.user_id = $1 AND b.target_type = 'tweet'
      AND b.action_type = 'tweet_view' AND b.target_id = t.id::text
) bh ON true
LEFT JOIN LATERAL (
    SELECT
      (SELECT COUNT(*)::bigint FROM user_follows f
        WHERE f.following_id = t.user_id AND f.status = 'active') AS followers,
      (SELECT COUNT(*)::bigint FROM user_follows f
        WHERE f.follower_id  = t.user_id AND f.status = 'active') AS following,
      (SELECT COUNT(*)::bigint FROM tweets a
        WHERE a.user_id = t.user_id AND a.deleted_at IS NULL)     AS tweet_count,
      GREATEST(0, EXTRACT(DAY FROM NOW() - u.created_at))::int    AS age_days
) au ON true
-- Un retweet pur porte un contenu vide : c'est le tweet d'origine qui a été
-- annoté. On rattache donc le label à `original_tweet_id` quand il existe,
-- sinon au tweet lui-même.
LEFT JOIN tweet_llm_labels ll ON ll.tweet_id = COALESCE(t.original_tweet_id, t.id)
"#;

/// Sélectionne des tweets par identifiant, aux mêmes conditions de visibilité
/// que le vivier — pour remonter le parent d'une réponse bien classée.
///
/// ⚠ Les filtres sont recopiés plutôt que délégués au fait « son enfant est
/// passé ». Le vivier vérifie déjà que le parent est affichable, mais par une
/// clause distincte, qui pourrait diverger : ce serait alors par ici qu'un
/// tweet supprimé, privé ou d'un compte suspendu rentrerait dans le fil, sans
/// jamais avoir été candidat. Une porte dérobée ne se laisse pas ouverte au
/// motif qu'une autre porte est fermée.
///
/// Deux différences assumées avec `visible` :
///   * le tweet du LECTEUR est admis — « quelqu'un a répondu à ton message »
///     est précisément un fil qui mérite d'être lu ;
///   * le plafond par auteur n'est pas appliqué : un parent n'est pas là pour
///     concourir, il est là pour rendre sa réponse lisible. `spread_by_author`
///     borne de toute façon sa présence à l'écran.
///
/// Paramètres : $1 lecteur · $2 identifiants de parents · $3 hard_banned[]
const PARENTS_CTE: &str = r#"
WITH picked AS (
    SELECT t.id, 0 AS src, 0.05::float8 AS w
    FROM tweets t
    JOIN users u ON u.id = t.user_id
    WHERE t.id = ANY($2)
      AND t.deleted_at IS NULL
      AND t.moderation_status = 'approved'
      AND t.is_private = false
      AND COALESCE(t.is_data_test, false) = false
      AND u.is_active = true
      AND COALESCE(u.is_suspended, false) = false
      AND NOT (t.user_id = ANY($3))
)
"#;

/// Hydrate les résultats de `embeddings::nearest_tweets` — mêmes identifiants
/// que `PARENTS_CTE`, poids un peu plus haut qu'un parent (0.15 plutôt que
/// 0.05) : contrairement à un parent, un candidat sémantique concourt
/// réellement pour une place dans le fil, il n'est pas là pour la lisibilité
/// d'un autre tweet. Étiqueté `src=4` (Discovery) : c'est déjà la source
/// « contenu qu'on n'a pas explicitement demandé », et ajouter une neuvième
/// variante à `TweetSource` pour une seule ligne de statistiques ne se
/// justifiait pas — le classement, lui, ne regarde que `w`.
///
/// Paramètres : $1 lecteur · $2 identifiants trouvés par similarité · $3 hard_banned[]
const SEMANTIC_CTE: &str = r#"
WITH picked AS (
    SELECT t.id, 4 AS src, 0.15::float8 AS w
    FROM tweets t
    JOIN users u ON u.id = t.user_id
    WHERE t.id = ANY($2)
      AND t.deleted_at IS NULL
      AND t.moderation_status = 'approved'
      AND t.is_private = false
      AND COALESCE(t.is_data_test, false) = false
      AND u.is_active = true
      AND COALESCE(u.is_suspended, false) = false
      AND NOT (t.user_id = ANY($3))
)
"#;

/// Hydrate les auteurs renvoyés par `CacheManager::co_liked_authors` — voir
/// `crate::cooccurrence`. Contrairement à `SEMANTIC_CTE` (des tweets
/// précis), on reçoit ici des AUTEURS : on prend leurs quelques tweets les
/// plus récents chacun (`ROW_NUMBER() OVER (PARTITION BY ...)`), pas
/// « tout ce qu'ils ont publié ». Poids légèrement sous le sémantique :
/// c'est un signal agrégé sur d'autres lecteurs, un cran plus indirect que
/// « ce tweet précis ressemble à ce que TU aimes ».
///
/// Paramètres : $1 lecteur (inutilisé, gardé pour la même forme que les
/// autres CTE) · $2 identifiants d'auteurs co-aimés · $3 hard_banned[]
const COOCCUR_CTE: &str = r#"
WITH ranked AS (
    SELECT t.id, t.user_id,
           ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY t.created_at DESC) AS rn
    FROM tweets t
    JOIN users u ON u.id = t.user_id
    WHERE t.user_id = ANY($2)
      AND t.deleted_at IS NULL
      AND t.moderation_status = 'approved'
      AND t.is_private = false
      AND COALESCE(t.is_data_test, false) = false
      AND u.is_active = true
      AND COALESCE(u.is_suspended, false) = false
      AND NOT (t.user_id = ANY($3))
      AND t.created_at > NOW() - INTERVAL '7 days'
),
picked AS (
    SELECT id, 4 AS src, 0.12::float8 AS w FROM ranked WHERE rn <= 3
)
"#;

/// Les requêtes complètes, assemblées une fois pour toutes.
///
/// `concat!` ne sait pas coller des constantes, et refaire le `format!` à
/// chaque requête réallouerait plusieurs kilo-octets par recommandation.
static CANDIDATES_SQL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{CANDIDATES_CTE}{PROJECTION_SQL}"));
static PARENTS_SQL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{PARENTS_CTE}{PROJECTION_SQL}"));
static SEMANTIC_SQL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{SEMANTIC_CTE}{PROJECTION_SQL}"));
static COOCCUR_SQL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{COOCCUR_CTE}{PROJECTION_SQL}"));

/// Hydratation par identifiants explicites — pour le rattrapage hors-ligne du
/// CTR (`services::ctr_backfill`), qui a besoin de tweets PRÉCIS (ceux d'une
/// interaction passée), pas d'une sélection de candidats.
///
/// `$1` reste le lecteur (lu par la jointure `impressions` de `PROJECTION_SQL`),
/// `$2` la liste d'identifiants. `src = 4` : valeur qui retombe sur
/// `TweetSource::Discovery` dans `map_rows` — neutre, aucune des huit sources
/// n'a de sens pour un tweet reconstruit après coup.
const BY_IDS_CTE: &str = r#"
WITH picked AS (
    SELECT id, 4 AS src, 0.05::float8 AS w
    FROM tweets
    WHERE id = ANY($2)
)
"#;
static BY_IDS_SQL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{BY_IDS_CTE}{PROJECTION_SQL}"));

// ─── Mapping rows → RawTweet ──────────────────────────────────────────────────

/// Renfort accordé aux abonnements tant que le compte n'a presque pas d'histoire.
///
/// À l'inscription, les comptes suivis sont le seul signal existant : pas de
/// like, pas d'auteur favori, pas d'heure d'activité. Le renfort décroît
/// linéairement jusqu'à disparaître une fois `COLD_START_INTERACTION_FLOOR`
/// interactions atteintes — sinon un choix fait en dix secondes à l'inscription
/// continuerait de dominer le fil des mois plus tard.
fn cold_start_follow_multiplier(profile: &UserProfile) -> f64 {
    let interactions = (profile.liked_tweet_ids.len()
        + profile.retweeted_tweet_ids.len()
        + profile.replied_to_tweet_ids.len()
        + profile.bookmarked_tweet_ids.len()) as f64;

    if interactions >= COLD_START_INTERACTION_FLOOR {
        return 1.0;
    }
    let novelty = 1.0 - (interactions / COLD_START_INTERACTION_FLOOR);
    1.0 + novelty * (COLD_START_FOLLOW_BOOST_MAX - 1.0)
}

/// Multiplicateur « je suis ce compte » appliqué au score final.
///
/// Partagé par les modes Feed et ForYou : deux forces différentes pour le même
/// signal rendraient le fil incompréhensible selon l'onglet ouvert.
fn apply_follow_boost(score: f64, tweet: &RawTweet, profile: &UserProfile, mode: &str) -> f64 {
    let mut boost = 1.0;

    if profile.follows(&tweet.user_id) {
        boost *= FOLLOW_FEED_BOOST;
        let cold = cold_start_follow_multiplier(profile);
        boost *= cold;
        trace!(tweet_id = %tweet.id, mode, cold_start = cold, "Follow boost applied");
    }
    if profile.is_mutual(&tweet.user_id) {
        boost *= FOLLOW_MUTUAL_BOOST;
        trace!(tweet_id = %tweet.id, mode, "Mutual follow boost applied");
    }

    (score * boost).min(1.0)
}

/// Multiplicateur de score d'un auteur explicitement refusé par le lecteur.
///
/// Décroissance géométrique plutôt qu'un bannissement sec : un refus unique
/// peut viser CE tweet-là plus que le compte (on tombe sur un sujet qui
/// n'intéresse pas, d'un auteur qu'on lit par ailleurs). Un premier « non »
/// divise donc la visibilité par ~3, et trois « non » l'effacent à peu près —
/// ce qui reste réversible, puisque le compteur expire (30 jours).
///
/// Jamais zéro : un score nul sortirait l'auteur de tout classement, y compris
/// pour un tweet exceptionnel, et rendrait le refus indistinguable d'un blocage.
fn author_damping(strikes: f64) -> f64 {
    const PER_STRIKE: f64 = 0.32;
    const FLOOR: f64 = 0.02;
    PER_STRIKE.powf(strikes.max(0.0)).max(FLOOR)
}

/// Tirage du mode Trending : une température pour l'ouverture, une pour la suite.
///
/// Le mélange pondéré par score (Gumbel-max) évite qu'un rafraîchissement
/// resserve exactement le même ordre. Mais avec une température UNIQUE, la
/// position 1 est tirée aussi au hasard que la position 50 — or ces premières
/// cartes décident si la personne continue ou repart. Une mauvaise ouverture
/// tirée au sort gâche une page dont la suite était bonne.
///
/// D'où deux temps :
///   * l'OUVERTURE (`TRENDING_HOOK_SIZE` cartes) est échantillonnée à
///     température réduite dans le haut du classement (`TRENDING_HOOK_POOL`) —
///     donc du contenu solide, mais pas la même sélection à chaque tirage ;
///   * la SUITE reprend la température pleine, qui est ce qui rend le défilement
///     imprévisible et donne sa chance à ce qui n'est pas en tête.
///
/// Les tweets du vivier non retenus pour l'ouverture ne sont pas relégués
/// derrière la queue : ils la rejoignent et sont retirés avec elle.
///
/// `.score` n'est jamais modifié — seul l'ORDRE change.
fn trending_draw(scored: Vec<ScoredTweet>) -> Vec<ScoredTweet> {
    if scored.len() <= 1 {
        return scored;
    }
    let mut rng = rand::thread_rng();

    // Clé Gumbel-max : trier sur `ln(score) + bruit × température` équivaut à
    // tirer sans remise proportionnellement au score. Température basse → colle
    // au classement ; haute → s'en écarte.
    let mut key_of = |score: f64, temperature: f64| -> f64 {
        let u: f64 = rng.gen_range(1e-9..1.0 - 1e-9);
        let gumbel_noise = -(-u.ln()).ln();
        score.max(1e-9).ln() + gumbel_noise * temperature
    };

    // `scored` arrive trié par score décroissant : le vivier d'ouverture est
    // simplement sa tête.
    let pool_len = TRENDING_HOOK_POOL.min(scored.len());
    let mut pool = scored;
    let tail = pool.split_off(pool_len);

    let mut keyed_pool: Vec<(f64, ScoredTweet)> = pool
        .into_iter()
        .map(|s| (key_of(s.score, TRENDING_HOOK_TEMPERATURE), s))
        .collect();
    keyed_pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let hook_len = TRENDING_HOOK_SIZE.min(keyed_pool.len());
    let leftovers = keyed_pool.split_off(hook_len);
    let hook: Vec<ScoredTweet> = keyed_pool.into_iter().map(|(_, s)| s).collect();

    let mut rest: Vec<ScoredTweet> = leftovers.into_iter().map(|(_, s)| s).collect();
    rest.extend(tail);
    let mut keyed_rest: Vec<(f64, ScoredTweet)> = rest
        .into_iter()
        .map(|s| (key_of(s.score, TRENDING_SHUFFLE_TEMPERATURE), s))
        .collect();
    keyed_rest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    hook.into_iter()
        .chain(keyed_rest.into_iter().map(|(_, s)| s))
        .collect()
}

fn map_rows(rows: Vec<tokio_postgres::Row>) -> Vec<RawTweet> {
    rows.into_iter()
        .filter_map(|r| {
            let id: String = r.try_get(0).ok()?;
            let user_id: String = r.try_get(1).ok()?;
            let content: String = r.try_get(2).unwrap_or_default();

            // Émojis, ponctuation, URLs et mots : un seul passage sur le texte,
            // et les minuscules gardées pour le scoring au lieu d'être
            // recalculées à chaque recommandation. Voir `crate::content`.
            let text = crate::content::analyze_content(&content);
            let emoji_count = text.emoji_count;
            let exclamation_count = text.exclamation_count;
            let question_count = text.question_count;
            let url_count = text.url_count;

            Some(RawTweet {
                id,
                user_id,
                content,
                created_at: r.try_get(3).ok()?,
                view_count: r.try_get(4).unwrap_or(0),
                like_count: r.try_get(5).unwrap_or(0),
                comment_count: r.try_get(6).unwrap_or(0),
                retweet_count: r.try_get(7).unwrap_or(0),
                report_count: r.try_get(8).unwrap_or(0),
                has_media: r.try_get(9).unwrap_or(false),
                hashtag_count: r.try_get(10).unwrap_or(0),
                mention_count: r.try_get(11).unwrap_or(0),
                content_length: r.try_get(12).unwrap_or(0),
                author_followers: r.try_get(13).unwrap_or(0),
                author_following: r.try_get(14).unwrap_or(0),
                author_tweet_count: r.try_get(15).unwrap_or(0),
                author_account_age_days: r.try_get(16).unwrap_or(0),
                author_is_verified: r.try_get(17).unwrap_or(false),
                author_is_premium: r.try_get(18).unwrap_or(false),
                author_tier: AuthorTier::resolve(
                    r.try_get::<_, &str>(38).unwrap_or("free"),
                    r.try_get(18).unwrap_or(false),
                ),
                author_visibility_multiplier: r.try_get::<_, f64>(19).unwrap_or(1.0),
                moderation_status: r.try_get(20).unwrap_or_else(|_| "approved".into()),
                recommendation_group: r.try_get(21).ok().flatten(),
                likes_1h: r.try_get(22).unwrap_or(0),
                likes_6h: r.try_get(23).unwrap_or(0),
                comments_1h: r.try_get(24).unwrap_or(0),
                retweets_1h: r.try_get(25).unwrap_or(0),
                viewer_impressions: r.try_get(26).unwrap_or(0),
                // Aucune table `tweet_shares` / `tweet_bookmarks` n'existe dans ce
                // schéma : ces compteurs restent structurellement nuls (les termes
                // correspondants de D1 sont donc inertes, ce n'est pas une omission).
                share_count: 0,
                bookmark_count: 0,
                emoji_count,
                exclamation_count,
                question_count,
                url_count,
                text,
                source: match r.try_get::<_, i32>(27).unwrap_or(4) {
                    1 => TweetSource::Trending,
                    2 => TweetSource::SocialGraph,
                    3 => TweetSource::Viral,
                    5 => TweetSource::Temporal,
                    6 => TweetSource::Influencer,
                    7 => TweetSource::Personalized,
                    8 => TweetSource::Quality,
                    _ => TweetSource::Discovery,
                },
                source_weight: r.try_get::<_, f64>(28).unwrap_or(0.05),
                parent_tweet_id: r.try_get::<_, Option<String>>(29).ok().flatten(),
                original_tweet_id: r.try_get::<_, Option<String>>(30).ok().flatten(),
                is_retweet: r.try_get(31).unwrap_or(false),
                // Chargé depuis Redis/DB par le job de qualité ; Clean par défaut
                author_shadowban_level: crate::shadowban::ShadowbanLevel::Clean,
                // `theme` NULL ⇒ pas de ligne dans tweet_llm_labels ⇒ tweet non
                // annoté. On laisse `None` plutôt que d'inventer des valeurs par
                // défaut, pour que D9 puisse rester neutre en connaissance de cause.
                llm: r
                    .try_get::<_, Option<String>>(32)
                    .ok()
                    .flatten()
                    .map(|theme| LlmLabels {
                        theme,
                        toxicity_score: r.try_get::<_, f64>(33).unwrap_or(0.0),
                        toxicity_category: r.try_get(34).unwrap_or_else(|_| "aucune".into()),
                        quality_score: r.try_get::<_, f64>(35).unwrap_or(0.5),
                        tone: r.try_get(36).unwrap_or_else(|_| "neutre".into()),
                        confidence: r.try_get::<_, f64>(37).unwrap_or(0.5),
                    }),
            })
        })
        .collect()
}

/// Part maximale de réponses dans le fil. Au-delà, le lecteur a l'impression de
/// lire des bouts de conversation plutôt qu'un fil d'actualité.
const MAX_REPLY_RATIO: f64 = 0.25;

/// Profondeur maximale de remontée d'un fil de réponses. Borne les chaînes
/// longues et protège d'un cycle si la base contenait une boucle parent/enfant.
const MAX_THREAD_DEPTH: usize = 4;

/// Nombre maximum de tweets qu'un même auteur peut placer dans le VIVIER de
/// candidats, toutes sources confondues.
///
/// 12 laisse de quoi remplir les trois premières pages à raison de trois par
/// page (voir `MAX_PER_AUTHOR_PER_PAGE`) sans qu'un seul compte puisse occuper
/// la moitié des candidats. C'est le premier des trois verrous de diversité, et
/// le seul qui agisse avant le scoring : les deux autres (pénalité de score et
/// étalement de mise en page) ne peuvent que réordonner ce que celui-ci laisse
/// passer.
/// ⚠ `i64` et non `i32` : il est comparé à `ROW_NUMBER()`, qui vaut `bigint`
/// côté Postgres. Avec un `i32`, tokio-postgres refuse la sérialisation du
/// paramètre (« cannot convert between the Rust type i32 and the Postgres type
/// int8 ») et TOUTE recommandation part en 500.
const MAX_CANDIDATES_PER_AUTHOR: i64 = 12;

/// Nombre maximum de tweets d'un même auteur sur une page de fil.
///
/// Trois, c'est assez pour comprendre qu'un compte publie beaucoup sur un sujet,
/// et trop peu pour qu'il occupe la page. Le score seul n'y suffisait pas :
/// `diversity_multiplier` DÉCOURAGE la répétition, il ne l'interdit pas — un
/// compte très en forme face à peu de concurrence gardait un score dégradé mais
/// toujours supérieur au reste, et raflait dix places d'affilée.
const MAX_PER_AUTHOR_PER_PAGE: u32 = 3;

/// Taille de page servant de fenêtre au plafond ci-dessus.
///
/// Alignée sur la valeur demandée par l'app (`limit = 50` côté API, voir
/// `neuralRankRoutes.js`). Le plafond est appliqué à la CONSTRUCTION de la liste
/// complète, donc avant la mise en cache : toutes les pages en héritent, y
/// compris celles servies depuis Redis.
const PAGE_WINDOW: usize = 50;

/// Met en forme la liste finale d'identifiants servie au client.
///
/// Trois règles, appliquées dans l'ordre du classement :
///
/// 1. **Un fil n'occupe qu'une entrée.** Soit le tweet racine, soit une de ses
///    réponses — jamais les deux, et jamais deux réponses au même tweet.
/// 2. Une réponse n'est servie que si tous ses ancêtres figurent parmi les
///    candidats retenus. Sinon son contexte est inaffichable (parent supprimé,
///    privé, auteur bloqué) et le lecteur verrait une repartie sans savoir à
///    quoi elle répond.
/// 3. Les réponses sont plafonnées à `MAX_REPLY_RATIO` du fil.
///
/// Le parent **est** émis dans la liste, juste avant sa réponse.
///
/// C'est le contrat que les clients appliquent réellement : mobile comme
/// Windows rendent une liste plate et déduisent le fil de l'ADJACENCE
/// (`isThreadParent` = « l'élément suivant a mon id pour parent »). Aucun ne
/// reconstruit un fil depuis un champ de contexte.
///
/// Une version antérieure n'émettait que la réponse et marquait ses ancêtres
/// comme « à l'écran », en comptant sur le client pour dessiner une carte de
/// contexte au-dessus. Personne ne la dessinait : la réponse arrivait seule,
/// illisible, et l'API devait ensuite l'écarter. On émet donc la chaîne
/// entière — le fil est du ressort de celui qui décide de l'ordre.
///
/// ⚠ L'invariant tient tant que rien ne réordonne la liste ensuite :
/// `spread_by_author` déplace les fils par BLOCS pour cette raison.
pub fn shape_feed<'a>(scored: &[ScoredTweet], tweets: &HashMap<&str, &'a RawTweet>) -> Vec<String> {
    let max_replies = ((scored.len() as f64) * MAX_REPLY_RATIO).ceil() as usize;

    let mut out: Vec<String> = Vec::with_capacity(scored.len());
    // Tweets déjà visibles à l'écran, que ce soit comme entrée de feed ou comme
    // carte de contexte rendue par le client au-dessus d'une réponse.
    // Des references et non des `String` : chaque identifiant emis etait
    // recopie DEUX fois, une pour la liste de sortie et une pour cet ensemble.
    // La seconde ne survivait pas a la fonction. Les identifiants vivent dans
    // les tweets, qui vivent plus longtemps que la mise en forme.
    let mut shown: crate::utils::FxHashSet<&'a str> =
        crate::utils::FxHashSet::with_capacity_and_hasher(scored.len(), Default::default());
    let mut replies_kept = 0usize;
    let mut replies_dropped = 0usize;

    for s in scored {
        let Some(t) = tweets.get(s.tweet_id.as_str()) else {
            continue;
        };
        if shown.contains(s.tweet_id.as_str()) {
            continue;
        }

        // Remonte le fil : [tweet, parent, …, racine].
        let mut chain: Vec<&RawTweet> = vec![t];
        let mut cursor = *t;
        let mut complete = true;
        while let Some(parent_id) = cursor.parent_tweet_id.as_deref() {
            if chain.len() >= MAX_THREAD_DEPTH {
                complete = false;
                break;
            }
            match tweets.get(parent_id) {
                Some(p) => {
                    chain.push(*p);
                    cursor = *p;
                }
                // Ancêtre absent des candidats : filtré en amont ou non
                // collecté. Le contexte serait incomplet, on écarte.
                None => {
                    complete = false;
                    break;
                }
            }
        }

        let is_reply = t.parent_tweet_id.is_some();

        if is_reply && !complete {
            replies_dropped += 1;
            continue;
        }
        if is_reply && replies_kept >= max_replies {
            replies_dropped += 1;
            continue;
        }

        // Le parent vient juste d'être émis : la réponse PROLONGE ce fil au lieu
        // d'en ouvrir un second. On l'ajoute telle quelle, sans réémettre la
        // chaîne — c'est le cas d'un tweet et de sa réponse tous deux bien
        // classés, où le fil se lit naturellement de haut en bas.
        let extends_last = chain
            .get(1)
            .zip(out.last())
            .is_some_and(|(parent, previous)| parent.id == *previous);
        if extends_last {
            out.push(t.id.clone());
            shown.insert(t.id.as_str());
            replies_kept += 1;
            continue;
        }

        // Un ancêtre est à l'écran, mais PLUS HAUT : réémettre la réponse ici la
        // séparerait de son parent de plusieurs tweets, ce qui la rend illisible
        // — et le client ne tracerait aucun trait de conversation entre les
        // deux. Le fil est déjà représenté, on passe.
        if chain.iter().skip(1).any(|a| shown.contains(a.id.as_str())) {
            if is_reply {
                replies_dropped += 1;
            }
            continue;
        }

        // Émission de la racine vers la feuille : `chain` a été construite en
        // remontant, elle se lit donc à l'envers. C'est cet ordre-là qui fait
        // le fil à l'écran.
        for a in chain.iter().rev() {
            out.push(a.id.clone());
            shown.insert(a.id.as_str());
        }
        if is_reply {
            replies_kept += 1;
        }
    }

    if replies_dropped > 0 || replies_kept > 0 {
        debug!(
            replies_kept,
            replies_dropped,
            max_replies,
            total = out.len(),
            "Mise en forme du fil : un fil = une entrée"
        );
    }
    out
}

/// Répartit le fil pour qu'aucun auteur n'occupe plus de
/// `MAX_PER_AUTHOR_PER_PAGE` places par page de `PAGE_WINDOW`.
///
/// Le surplus n'est PAS jeté : il est reporté à la page suivante. C'est la
/// différence entre « diversifier » et « censurer » — un compte prolifique reste
/// entièrement lisible, il est simplement étalé au lieu d'être servi en bloc.
/// Jeter le surplus aurait aussi rendu le nombre de tweets disponibles
/// dépendant de la composition du fil, donc la pagination incohérente.
///
/// L'ordre par score est conservé dans chaque page : on ne remonte jamais un
/// tweet moins bon devant un meilleur, on descend seulement celui qui ne tient
/// pas dans le quota de son auteur. Une page peut donc contenir moins de
/// `PAGE_WINDOW` entrées quand il n'y a pas assez d'auteurs distincts — c'est le
/// prix assumé du plafond, et il vaut mieux une page courte et variée qu'une
/// page pleine signée trois fois par la même personne.
/// ⚠ L'unité déplacée est le FIL, pas le tweet. `shape_feed` vient de garantir
/// qu'une réponse suit immédiatement son parent ; déplacer les tweets un par un
/// aurait cassé cette adjacence dès que le parent et la réponse ont le même
/// auteur — le parent partait en page suivante et la réponse restait seule,
/// exactement l'orpheline qu'on cherche à éviter. Un fil compte donc pour ses
/// auteurs, mais se déplace d'un bloc.
pub fn spread_by_author<'a>(ids: Vec<String>, tweets: &HashMap<&str, &'a RawTweet>) -> Vec<String> {
    let author_of = |id: &str| -> Option<&'a str> { tweets.get(id).map(|t| t.user_id.as_str()) };

    let total = ids.len();
    let mut out: Vec<String> = Vec::with_capacity(total);
    // Auteurs présents dans les `PAGE_WINDOW` dernières positions émises, et
    // leur compte. Tenu à jour au fil de l'eau plutôt que recalculé : le
    // recalcul à chaque candidat rendait la fonction quadratique en la fenêtre.
    let mut window: crate::utils::FxHashMap<&'a str, u32> = Default::default();
    let blocks: Vec<Vec<String>> = group_threads(ids, tweets);
    let mut deferrals = 0usize;
    let mut forced = 0usize;

    // ── Ce que chaque bloc DEMANDE, calculé une seule fois ───────────────────
    //
    // Un fil ne passe que si CHACUN de ses auteurs a encore de la place ; un
    // fil de trois tweets du même auteur compte pour trois, sinon un compte
    // prolifique reprendrait par le fil ce que le plafond lui refuse au tweet.
    //
    // ⚠ Ce besoin était recalculé — table de hachage allouée comprise — à
    // CHAQUE fois qu'on regardait si un bloc tenait dans la fenêtre, c'est-à-
    // dire une fois par bloc restant et par place du fil. Mesuré sur le vivier
    // réel (1700 candidats, une dizaine d'auteurs distincts) : **339 ms**, soit
    // plus de cent fois le coût du scoring des mêmes 1700 tweets. C'était, et
    // de loin, le premier poste de calcul d'une recommandation.
    //
    // Un `Vec` et non une table : un bloc porte quatre auteurs au plus
    // (`MAX_THREAD_DEPTH`), un balayage linéaire y est plus rapide qu'un
    // hachage, et surtout il n'alloue pas.
    let needs: Vec<Vec<(&'a str, u32)>> = blocks
        .iter()
        .map(|block| {
            let mut need: Vec<(&'a str, u32)> = Vec::new();
            for author in block.iter().filter_map(|id| author_of(id)) {
                match need.iter_mut().find(|(a, _)| *a == author) {
                    Some((_, n)) => *n += 1,
                    None => need.push((author, 1)),
                }
            }
            need
        })
        .collect();

    let fits = |need: &[(&str, u32)], window: &crate::utils::FxHashMap<&str, u32>| -> bool {
        need.iter()
            .all(|(a, n)| window.get(a).copied().unwrap_or(0) + n <= MAX_PER_AUTHOR_PER_PAGE)
    };

    // ── Les blocs, rangés par auteur ─────────────────────────────────────────
    //
    // La boucle cherchait « le premier bloc qui tient » en balayant TOUT ce qui
    // reste à placer, et quand plus rien ne tenait — l'état permanent dès que
    // le vivier compte peu d'auteurs — elle le rebalayait une seconde fois pour
    // choisir le moins mauvais. Deux parcours complets par place servie.
    //
    // Or deux blocs du MÊME auteur demandant la même chose sont
    // interchangeables du point de vue du quota : si le premier ne tient pas,
    // aucun autre ne tiendra, et si l'on doit forcer, c'est le premier qu'on
    // prend. Il suffit donc de regarder la TÊTE de chaque file d'auteur — une
    // dizaine de tests au lieu de mille sept cents.
    //
    // Les blocs qui ne rentrent pas dans ce raisonnement (un fil à plusieurs
    // auteurs, ou plusieurs tweets du même auteur) restent examinés un par un.
    // Ils sont rares : un fil compte pour un quart du feed au plus, et la
    // plupart n'ont qu'un auteur.
    let mut simples: crate::utils::FxHashMap<&'a str, std::collections::VecDeque<usize>> =
        Default::default();
    let mut composes: Vec<usize> = Vec::new();
    for (i, need) in needs.iter().enumerate() {
        match need.as_slice() {
            [(author, 1)] => simples.entry(author).or_default().push_back(i),
            _ => composes.push(i),
        }
    }

    let mut pending: Vec<Option<Vec<String>>> = blocks.into_iter().map(Some).collect();
    let mut restants = pending.len();
    // Plus petit index encore en attente. Le curseur ne recule jamais, donc son
    // avancée coûte au total le nombre de blocs, pas un balayage par place.
    let mut premier = 0usize;

    while restants > 0 {
        while pending[premier].is_none() {
            premier += 1;
        }

        // ── Chemin rapide : le meilleur candidat restant tient ───────────────
        //
        // C'est le cas courant tant que la fenêtre n'est pas saturée, et c'est
        // exactement ce que faisait la version d'avant en temps constant. Sans
        // ce raccourci, l'indexation par auteur ferait PERDRE du temps sur un
        // vivier varié : on paierait un test par auteur là où le tout premier
        // bloc convenait.
        let pick: Option<usize> = if fits(&needs[premier], &window) {
            Some(premier)
        } else {
            // Premier candidat, dans l'ordre du score, dont l'auteur n'a pas
            // encore saturé son quota sur la fenêtre courante.
            let mut trouve: Option<usize> = None;
            for (author, file) in &simples {
                let Some(&i) = file.front() else { continue };
                // Le bloc ne demande qu'une place, d'où le `<` plutôt que le
                // `+ 1 <=` de `fits` : c'est la même condition écrite sans
                // l'addition.
                if window.get(author).copied().unwrap_or(0) < MAX_PER_AUTHOR_PER_PAGE {
                    trouve = Some(trouve.map_or(i, |best: usize| best.min(i)));
                }
            }
            for &i in &composes {
                if fits(&needs[i], &window) {
                    trouve = Some(trouve.map_or(i, |best: usize| best.min(i)));
                    // `composes` est en ordre croissant : le premier qui tient
                    // est le plus petit, inutile de continuer.
                    break;
                }
            }
            trouve
        };

        let index = match pick {
            Some(i) => {
                if i > premier {
                    deferrals += 1;
                }
                i
            }
            // Plus AUCUN candidat ne respecte le quota : il ne reste que des
            // auteurs saturés. Ça arrive dès qu'il y a moins de
            // PAGE_WINDOW / MAX_PER_AUTHOR_PER_PAGE auteurs distincts — 17 pour
            // une fenêtre de 50 — donc en permanence sur une petite communauté.
            //
            // On sert quand même : laisser un trou n'aurait aucun sens, et
            // refuser ferait dépendre la taille de la page du nombre d'auteurs
            // disponibles, ce qui casserait la pagination du client.
            //
            // ⚠ Mais surtout PAS dans l'ordre brut. Reprendre le premier vidait
            // le premier auteur d'un coup : on obtenait « 3 de chacun » puis
            // « 7 d'affilée du premier », soit pire qu'un simple tour de rôle.
            // On prend donc le candidat dont l'auteur est le MOINS présent dans
            // la fenêtre, ce qui dégrade en round-robin propre au lieu de
            // retomber en blocs.
            None => {
                forced += 1;
                // Présence de l'auteur le PLUS installé du fil : c'est lui qui
                // décide si le servir maintenant déséquilibre la fenêtre. À
                // égalité de présence, l'ordre du score tranche — d'où la clé
                // `(présence, index)`, qui est un ordre TOTAL : le parcours
                // d'une table de hachage n'a pas d'ordre, mais un minimum sur
                // une clé totale, si.
                let mut best: Option<(u32, usize)> = None;
                let mut retenir = |seen: u32, i: usize| {
                    if best.is_none_or(|b| (seen, i) < b) {
                        best = Some((seen, i));
                    }
                };
                for (author, file) in &simples {
                    if let Some(&i) = file.front() {
                        retenir(window.get(author).copied().unwrap_or(0), i);
                    }
                }
                for &i in &composes {
                    let seen = needs[i]
                        .iter()
                        .map(|(a, _)| window.get(a).copied().unwrap_or(0))
                        .max()
                        .unwrap_or(0);
                    retenir(seen, i);
                }
                best.map_or(premier, |(_, i)| i)
            }
        };

        // Retirer le bloc de la structure où il attendait. Une tête de file se
        // retire en temps constant ; un bloc composé se retire d'une liste qui
        // reste courte.
        match needs[index].as_slice() {
            [(author, 1)] => {
                if let Some(file) = simples.get_mut(author) {
                    file.pop_front();
                    if file.is_empty() {
                        simples.remove(author);
                    }
                }
            }
            _ => composes.retain(|&i| i != index),
        }

        // Le bloc entier part d'un coup : c'est ce qui garde la réponse collée
        // à son parent.
        let Some(block) = pending[index].take() else {
            // Impossible : un index ne sort qu'une fois de sa file.
            continue;
        };
        restants -= 1;
        for id in block {
            if let Some(a) = author_of(&id) {
                *window.entry(a).or_insert(0) += 1;
            }
            out.push(id);

            // La fenêtre glisse : l'auteur qui sort des `PAGE_WINDOW` dernières
            // positions retrouve son quota. C'est ce glissement qui rend la
            // garantie valable pour N'IMPORTE quelle page, quels que soient
            // `offset` et `limit` — et pas seulement pour un découpage aligné
            // sur 50.
            if out.len() > PAGE_WINDOW {
                let leaving = &out[out.len() - PAGE_WINDOW - 1];
                if let Some(a) = author_of(leaving) {
                    if let Some(count) = window.get_mut(a) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            window.remove(a);
                        }
                    }
                }
            }
        }
    }

    debug_assert_eq!(out.len(), total, "l'étalement réordonne, il ne filtre pas");

    if deferrals > 0 || forced > 0 {
        debug!(
            deferrals,
            forced,
            window = PAGE_WINDOW,
            max_per_author = MAX_PER_AUTHOR_PER_PAGE,
            "Étalement par auteur appliqué"
        );
    }

    out
}

/// Annote la liste finale du lien de fil de chaque entrée.
///
/// Le parent n'est retenu que s'il occupe la position PRÉCÉDENTE. Un parent
/// présent ailleurs dans la liste ne compte pas : ce champ décrit ce que
/// l'écran montre — « le tweet juste au-dessus est celui auquel je réponds » —
/// et pas la généalogie du tweet, que la base connaît déjà.
fn as_feed_entries(
    ids: &[String],
    tweets: &HashMap<&str, &RawTweet>,
    scores: &HashMap<&str, f64>,
    profile: &UserProfile,
) -> Vec<FeedEntry> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let tweet = tweets.get(id.as_str());
            let parent_id = tweet
                .and_then(|t| t.parent_tweet_id.as_deref())
                .filter(|parent| i > 0 && ids[i - 1] == *parent)
                .map(String::from);
            FeedEntry {
                id: id.clone(),
                parent_id,
                score: scores.get(id.as_str()).copied().unwrap_or(0.0),
                // Calculée ICI, au même endroit et pour la même raison que le
                // lien de fil : c'est le dernier instant où l'on dispose
                // encore du tweet complet ET du profil. Après la mise en
                // cache il ne reste que des identifiants.
                confidence: tweet
                    .map(|t| crate::algorithm::scoring::ranking_confidence(t, profile))
                    .unwrap_or(0.0),
            }
        })
        .collect()
}

/// Traduit les entrées d'une page en scores exposables.
///
/// Séparé de `thread_links` bien que les deux parcourent la même page : l'un
/// décrit la structure du fil, l'autre ce que le moteur a pensé de chaque
/// tweet. Les fusionner obligerait tout client qui veut l'un à comprendre
/// l'autre.
fn page_scores(page: &[FeedEntry]) -> Vec<TweetScore> {
    page.iter()
        .map(|entry| TweetScore {
            tweet_id: entry.id.clone(),
            score: entry.score,
            confidence: entry.confidence,
        })
        .collect()
}

/// Traduit les entrées d'une PAGE en liens de conversation exposables.
///
/// Travaille sur la page servie, pas sur la liste complète : un lien vers un
/// parent que le client n'a pas reçu ne lui servirait à rien, sinon à lui faire
/// afficher un fil troué.
fn thread_links(page: &[FeedEntry]) -> Vec<ThreadLink> {
    // Racine de chaque fil rencontré, propagée de proche en proche : la racine
    // d'une réponse est celle de son parent, ou le parent lui-même.
    let mut root_of: HashMap<&str, &str> = HashMap::new();
    let mut depth_of: HashMap<&str, usize> = HashMap::new();
    let mut links = Vec::new();

    for entry in page {
        let Some(parent) = entry.parent_id.as_deref() else {
            continue;
        };
        let root = root_of.get(parent).copied().unwrap_or(parent);
        let depth = depth_of.get(parent).copied().unwrap_or(0) + 1;
        root_of.insert(entry.id.as_str(), root);
        depth_of.insert(entry.id.as_str(), depth);
        links.push(ThreadLink {
            tweet_id: entry.id.clone(),
            parent_id: parent.to_string(),
            root_id: root.to_string(),
            depth,
        });
    }

    links
}

/// Regroupe une liste plate en blocs indéplaçables : un fil, ou un tweet seul.
///
/// Le fil est reconnu à l'ADJACENCE — un tweet dont le parent est l'élément qui
/// le précède immédiatement rejoint son bloc — et pas à une structure d'arbre.
/// C'est exactement la règle que les clients appliquent pour tracer le trait de
/// conversation : grouper autrement produirait des blocs que l'écran ne
/// dessinerait pas comme des fils.
fn group_threads(ids: Vec<String>, tweets: &HashMap<&str, &RawTweet>) -> Vec<Vec<String>> {
    let mut blocks: Vec<Vec<String>> = Vec::with_capacity(ids.len());

    for id in ids {
        let follows_previous = tweets
            .get(id.as_str())
            .and_then(|t| t.parent_tweet_id.as_deref())
            .zip(blocks.last().and_then(|b| b.last()))
            .is_some_and(|(parent, previous)| parent == previous.as_str());

        match blocks.last_mut() {
            Some(block) if follows_previous => block.push(id),
            _ => blocks.push(vec![id]),
        }
    }

    blocks
}

/// Identité de contenu d'un candidat : ce que le lecteur va réellement lire.
///
/// Un retweet n'a pas de texte propre — il réaffiche son original. Deux
/// retweets du même tweet, ou un tweet et son retweet, montrent donc exactement
/// la même chose et ne doivent occuper qu'une place dans le feed.
fn content_identity(tweet: &RawTweet) -> &str {
    // Uniquement pour un vrai retweet. Les réponses renseignent elles aussi
    // `original_tweet_id` (elles pointent la racine du fil) : s'en servir ici
    // ferait passer une réponse pour un doublon du tweet auquel elle répond, et
    // deux réponses distinctes au même tweet pour un seul contenu.
    if tweet.is_retweet {
        tweet.original_tweet_id.as_deref().unwrap_or(&tweet.id)
    } else {
        &tweet.id
    }
}

fn deduplicate(mut tweets: Vec<RawTweet>) -> Vec<RawTweet> {
    // Clés hachées vite : ce sont des UUID que la base a produits, pas des
    // chaînes qu'un utilisateur choisit — voir `crate::utils::fxhash`.
    let mut seen: crate::utils::FxHashMap<String, usize> =
        crate::utils::FxHashMap::with_capacity_and_hasher(tweets.len(), Default::default());
    let mut result: Vec<RawTweet> = Vec::with_capacity(tweets.len());

    for tweet in tweets.drain(..) {
        // Clé = tweet d'origine, pas l'id du candidat. L'ancienne version
        // dédupliquait sur `tweet.id` : un tweet et trois retweets de ce tweet
        // passaient pour quatre contenus distincts et saturaient le feed avec
        // la même carte.
        //
        // La clé n'est recopiée QUE si le tweet est retenu : la version d'avant
        // allouait une chaîne par candidat, y compris pour les doublons qu'elle
        // s'apprêtait à jeter.
        match seen.get(content_identity(&tweet)).copied() {
            None => {
                seen.insert(content_identity(&tweet).to_string(), result.len());
                result.push(tweet);
            }
            Some(idx) => {
                // On garde l'exemplaire déjà retenu, mais on lui laisse le
                // meilleur poids de source des doublons rencontrés : si le
                // tweet arrive à la fois par « suivi » et par « trending »,
                // il mérite le poids du canal le plus fort.
                if tweet.source_weight > result[idx].source_weight {
                    result[idx].source_weight = tweet.source_weight;
                    result[idx].source = tweet.source;
                }
                // Un original a toujours priorité sur son retweet : il porte le
                // texte, les compteurs et l'auteur réels.
                if result[idx].original_tweet_id.is_some() && tweet.original_tweet_id.is_none() {
                    let kept_weight = result[idx].source_weight;
                    let kept_source = result[idx].source;
                    result[idx] = tweet;
                    result[idx].source_weight = kept_weight;
                    result[idx].source = kept_source;
                }
            }
        }
    }
    result
}

/// Mots vides FR/EN — sans ce filtre, `top_words` serait saturé de « dans »,
/// « pour », « that »… qui apparaissent dans presque tous les tweets et ne
/// discriminent donc rien.
const STOPWORDS: &[&str] = &[
    "avec", "pour", "dans", "cette", "cela", "sont", "mais", "vous", "nous", "elle", "leur",
    "leurs", "être", "etre", "avoir", "fait", "faire", "plus", "moins", "tout", "tous", "toute",
    "toutes", "comme", "quand", "aussi", "encore", "alors", "donc", "chez", "sans", "sous", "très",
    "tres", "bien", "peut", "veut", "vais", "suis", "j'ai", "c'est", "n'est", "qu'il", "qu'elle",
    "parce", "depuis", "entre", "that", "this", "with", "from", "have", "will", "your", "they",
    "them", "their", "what", "when", "which", "there", "here", "been", "were", "would", "could",
    "should", "about", "just", "like", "really", "https", "http",
];

/// Les mêmes, indexés.
///
/// La liste était balayée en entier pour CHAQUE mot de CHAQUE tweet aimé —
/// soixante-dix comparaisons de chaînes par mot, sur des milliers de mots à
/// chaque reconstruction de profil. Construite une seule fois pour la vie du
/// processus.
static STOPWORDS_INDEX: std::sync::LazyLock<crate::utils::FxHashSet<&'static str>> =
    std::sync::LazyLock::new(|| STOPWORDS.iter().copied().collect());

/// Reconstruit les centres d'intérêt, le style et la positivité d'un lecteur à
/// partir du texte des tweets qu'il a aimés.
///
/// Ces trois champs (`top_words`, `personality_type`, `emotional_positivity`)
/// existaient dans `UserProfile` et étaient lus par D2 et D8, mais rien ne les
/// remplissait : la correspondance mots-clés et l'affinité d'intérêt valaient
/// donc 0 pour tout le monde, quelle que soit l'activité réelle.
fn profile_from_liked_text(texts: &[String]) -> (Vec<(String, u32)>, PersonalityType, f64) {
    if texts.is_empty() {
        return (Vec::new(), PersonalityType::Balanced, 0.5);
    }

    let mut counts: HashMap<String, u32> = HashMap::new();
    let (mut emoji, mut excl, mut quest, mut urls, mut long_form) = (0u32, 0u32, 0u32, 0u32, 0u32);

    for text in texts {
        if text.chars().count() > 180 {
            long_form += 1;
        }
        excl += text.matches('!').count() as u32;
        quest += text.matches('?').count() as u32;
        let lower = text.to_lowercase();
        urls += (lower.matches("http://").count() + lower.matches("https://").count()) as u32;
        emoji += text
            .chars()
            .filter(|c| {
                let u = *c as u32;
                (0x1F300..=0x1FAFF).contains(&u)
                    || (0x2600..=0x27BF).contains(&u)
                    || u == 0x2B50
                    || u == 0x2764
            })
            .count() as u32;

        for raw in lower.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-') {
            let word = raw.trim_matches(|c: char| c == '\'' || c == '-');
            // On garde les mots d'au moins 4 caractères : en dessous, le signal
            // est dominé par les articles et les abréviations.
            if word.chars().count() < 4 || word.chars().count() > 30 {
                continue;
            }
            if word.chars().all(|c| c.is_numeric()) {
                continue;
            }
            if STOPWORDS_INDEX.contains(word) {
                continue;
            }
            *counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut top_words: Vec<(String, u32)> = counts
        .into_iter()
        // Un mot vu une seule fois sur l'ensemble de l'historique n'est pas un
        // centre d'intérêt, c'est du bruit.
        .filter(|(_, n)| *n >= 2)
        .collect();
    top_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_words.truncate(30);

    let n = texts.len() as f64;
    let personality = if emoji as f64 / n > 1.0 || excl as f64 / n > 0.8 {
        PersonalityType::Enthusiastic
    } else if quest as f64 / n > 0.5 || urls as f64 / n > 0.4 {
        PersonalityType::Curious
    } else if long_form as f64 / n > 0.5 {
        PersonalityType::Thoughtful
    } else {
        PersonalityType::Balanced
    };

    // Proxy de positivité : densité d'émojis et d'exclamations dans ce que le
    // lecteur aime. Volontairement borné — c'est une heuristique, pas une
    // analyse de sentiment.
    let positivity = (0.5 + (emoji as f64 / n) * 0.2 + (excl as f64 / n) * 0.1).clamp(0.0, 1.0);

    (top_words, personality, positivity)
}

fn mode_label(mode: &RecommendMode) -> &'static str {
    match mode {
        RecommendMode::Feed => "feed",
        RecommendMode::Discover => "discover",
        RecommendMode::Trending => "trending",
        RecommendMode::ForYou => "for_you",
    }
}

fn adaptive_ttl(profile: &UserProfile, mode: &RecommendMode) -> u64 {
    let mut ttl = match profile.user_type {
        UserType::PowerUser => 45_u64,
        UserType::Regular => 90_u64,
        UserType::Casual => 180_u64,
    };
    if profile.engagement_trend > 1.5 {
        ttl = ttl.saturating_sub(20);
    }
    if *mode == RecommendMode::Trending {
        ttl = ttl.min(60);
    }
    if *mode == RecommendMode::Discover {
        ttl = ttl.min(120);
    }
    ttl.max(30)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // ─── Poids des abonnements ───────────────────────────────────────────────

    fn profile_following(author: &str, interactions: usize) -> UserProfile {
        let mut p = UserProfile {
            following_ids: vec![author.to_string()],
            liked_tweet_ids: (0..interactions).map(|i| format!("t{i}")).collect(),
            ..Default::default()
        };
        // Comme en production : un profil n'est utilisable qu'index construit.
        p.rebuild_indexes();
        p
    }

    fn tweet_by(author: &str) -> RawTweet {
        RawTweet {
            user_id: author.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn un_compte_neuf_beneficie_du_renfort_maximal() {
        let profile = profile_following("a", 0);
        assert!(
            (cold_start_follow_multiplier(&profile) - COLD_START_FOLLOW_BOOST_MAX).abs() < 1e-9
        );
    }

    #[test]
    fn le_renfort_de_demarrage_disparait_une_fois_le_compte_actif() {
        // Passe le plancher : les interactions reelles remplacent le choix
        // d'inscription, le renfort ne doit plus rien ajouter.
        let profile = profile_following("a", COLD_START_INTERACTION_FLOOR as usize + 5);
        assert!((cold_start_follow_multiplier(&profile) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn le_renfort_decroit_avec_l_activite() {
        let neuf = cold_start_follow_multiplier(&profile_following("a", 0));
        let tiede = cold_start_follow_multiplier(&profile_following("a", 10));
        let actif = cold_start_follow_multiplier(&profile_following("a", 25));
        assert!(neuf > tiede && tiede > actif);
    }

    #[test]
    fn suivre_un_auteur_remonte_reellement_son_tweet() {
        let suivi = profile_following("auteur", 0);
        let inconnu = UserProfile::default();
        let t = tweet_by("auteur");

        let avec = apply_follow_boost(0.40, &t, &suivi, "ForYou");
        let sans = apply_follow_boost(0.40, &t, &inconnu, "ForYou");

        assert!(sans == 0.40, "un auteur non suivi ne doit rien gagner");
        // Le point de depart du probleme : l'ecart doit etre franc, pas cosmetique.
        assert!(avec >= sans * 1.4, "avec={avec}, sans={sans}");
    }

    #[test]
    fn le_boost_ne_fait_jamais_deborder_le_score() {
        let suivi = profile_following("auteur", 0);
        assert!(apply_follow_boost(0.99, &tweet_by("auteur"), &suivi, "ForYou") <= 1.0);
    }

    #[test]
    fn les_poids_de_dimensions_somment_a_un() {
        // Une somme differente de 1 fait deriver tous les scores et casse les
        // seuils calibres ailleurs (garbage, shadowban, bandit).
        let total: f64 = crate::admin::models::AlgoWeights::default()
            .as_array()
            .iter()
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "somme des poids = {total}");
    }

    // ─── Mise en forme du fil ────────────────────────────────────────────────

    fn tweet(id: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            ..Default::default()
        }
    }

    fn by(id: &str, author: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            user_id: author.to_string(),
            ..Default::default()
        }
    }

    // ─── Étalement par auteur ────────────────────────────────────────────────

    /// Construit la table `id -> tweet` attendue par `spread_by_author`.

    /// Les deux tests de lien de fil ne s'intéressent qu'à l'adjacence
    /// parent/réponse. Score et confiance n'entrent pas dans ce calcul :
    /// ce raccourci évite de les fabriquer pour rien.
    fn entries_for_test<'a>(
        ids: &[String],
        tweets: &HashMap<&'a str, &'a RawTweet>,
    ) -> Vec<FeedEntry> {
        as_feed_entries(ids, tweets, &HashMap::new(), &UserProfile::default())
    }

    fn index<'a>(raw: &'a [RawTweet]) -> HashMap<&'a str, &'a RawTweet> {
        raw.iter().map(|t| (t.id.as_str(), t)).collect()
    }

    // ─── Équivalence de l'étalement indexé ───────────────────────────────────
    //
    // `spread_by_author` a été réécrit : il cherchait « le premier bloc qui
    // tient » en balayant tout ce qui restait à placer, deux fois par place
    // quand rien ne tenait. Sur le vivier réel — 1700 candidats, une dizaine
    // d'auteurs distincts, donc une fenêtre saturée en permanence — ça coûtait
    // 339 ms, plus de cent fois le scoring des mêmes tweets.
    //
    // La version indexée ne regarde que la tête de la file de chaque auteur.
    // Le raisonnement qui la justifie : deux blocs du même auteur demandant la
    // même chose sont interchangeables pour le quota, donc si la tête ne tient
    // pas, aucun autre ne tient — et s'il faut forcer, c'est la tête qu'on
    // prend, puisque c'est le plus petit index.
    //
    // Ce raisonnement se démontre, mais l'ordre du fil de tous les lecteurs en
    // dépend. On garde donc l'implémentation d'origine ici et on compare les
    // deux sorties, à l'identique, sur des milliers de configurations.

    /// L'étalement tel qu'il était écrit avant l'indexation par auteur.
    /// N'existe plus que comme référence de ce test.
    fn spread_de_reference(ids: Vec<String>, tweets: &HashMap<&str, &RawTweet>) -> Vec<String> {
        let author_of = |id: &str| -> Option<&str> { tweets.get(id).map(|t| t.user_id.as_str()) };

        let mut out: Vec<String> = Vec::with_capacity(ids.len());
        let mut window: HashMap<String, u32> = HashMap::new();
        let mut pending: Vec<Vec<String>> = group_threads(ids, tweets);

        let fits = |block: &[String], window: &HashMap<String, u32>| -> bool {
            let mut need: HashMap<&str, u32> = HashMap::new();
            block.iter().filter_map(|id| author_of(id)).for_each(|a| {
                *need.entry(a).or_insert(0) += 1;
            });
            need.iter()
                .all(|(a, n)| window.get(*a).unwrap_or(&0) + n <= MAX_PER_AUTHOR_PER_PAGE)
        };

        while !pending.is_empty() {
            let pick = pending.iter().position(|block| fits(block, &window));
            let index = match pick {
                Some(i) => i,
                None => pending
                    .iter()
                    .enumerate()
                    .min_by_key(|(i, block)| {
                        let seen = block
                            .iter()
                            .filter_map(|id| author_of(id))
                            .map(|a| *window.get(a).unwrap_or(&0))
                            .max()
                            .unwrap_or(0);
                        (seen, *i)
                    })
                    .map_or(0, |(i, _)| i),
            };

            for id in pending.remove(index) {
                if let Some(a) = author_of(&id) {
                    *window.entry(a.to_string()).or_insert(0) += 1;
                }
                out.push(id);
                if out.len() > PAGE_WINDOW {
                    let leaving = &out[out.len() - PAGE_WINDOW - 1];
                    if let Some(a) = author_of(leaving) {
                        if let Some(count) = window.get_mut(a) {
                            *count = count.saturating_sub(1);
                            if *count == 0 {
                                window.remove(a);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Xorshift : un tirage rejouable. Un test d'équivalence qui échoue une
    /// fois sur cent sans qu'on puisse reproduire le cas ne sert à rien.
    struct Tirage(u64);

    impl Tirage {
        fn suivant(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn dans(&mut self, n: usize) -> usize {
            (self.suivant() % n as u64) as usize
        }
    }

    /// Un vivier tiré au sort : `taille` tweets, `auteurs` comptes, et une
    /// proportion de réponses qui forment des fils (donc des blocs à plusieurs
    /// tweets, parfois de plusieurs auteurs).
    fn vivier_aleatoire(
        tirage: &mut Tirage,
        taille: usize,
        auteurs: usize,
        part_reponses: usize,
    ) -> Vec<RawTweet> {
        let mut tweets: Vec<RawTweet> = Vec::with_capacity(taille);
        for i in 0..taille {
            let parent = if i > 0 && tirage.dans(100) < part_reponses {
                Some(format!("t{}", tirage.dans(i)))
            } else {
                None
            };
            tweets.push(RawTweet {
                id: format!("t{i}"),
                user_id: format!("a{}", tirage.dans(auteurs)),
                parent_tweet_id: parent,
                ..Default::default()
            });
        }
        tweets
    }

    #[test]
    fn l_etalement_indexe_rend_exactement_le_meme_fil() {
        let mut tirage = Tirage(0x2026_0831_5EED_0001);
        // Les tailles couvrent le cas où la fenêtre ne sature jamais (beaucoup
        // d'auteurs) et celui où elle sature en permanence (un ou deux).
        for &(taille, auteurs, part_reponses) in &[
            (1usize, 1usize, 0usize),
            (5, 1, 0),
            (20, 1, 0),
            (20, 2, 0),
            (60, 3, 0),
            (60, 3, 30),
            (120, 5, 25),
            (120, 40, 25),
            (200, 8, 40),
            (200, 60, 10),
            (300, 2, 50),
        ] {
            for essai in 0..12 {
                let tweets = vivier_aleatoire(&mut tirage, taille, auteurs, part_reponses);
                let carte = index(&tweets);
                let ids: Vec<String> = tweets.iter().map(|t| t.id.clone()).collect();

                let attendu = spread_de_reference(ids.clone(), &carte);
                let obtenu = spread_by_author(ids.clone(), &carte);

                assert_eq!(
                    obtenu, attendu,
                    "taille={taille} auteurs={auteurs} reponses={part_reponses}% essai={essai}"
                );
                // Et l'invariant de base : on réordonne, on ne filtre pas.
                assert_eq!(obtenu.len(), ids.len());
            }
        }
    }

    #[test]
    fn l_etalement_ne_perd_ni_ne_duplique_aucun_tweet() {
        let mut tirage = Tirage(0x2026_0831_5EED_0002);
        let tweets = vivier_aleatoire(&mut tirage, 400, 6, 35);
        let carte = index(&tweets);
        let ids: Vec<String> = tweets.iter().map(|t| t.id.clone()).collect();
        let out = spread_by_author(ids.clone(), &carte);

        let avant: std::collections::BTreeSet<&String> = ids.iter().collect();
        let apres: std::collections::BTreeSet<&String> = out.iter().collect();
        assert_eq!(avant, apres);
        assert_eq!(out.len(), ids.len(), "aucun doublon");
    }

    /// Pire concentration d'un auteur sur toute fenêtre glissante de
    /// `PAGE_WINDOW` positions, dans les `depth` premières entrées.
    ///
    /// Fenêtre GLISSANTE et non découpage aligné : une page peut commencer à
    /// n'importe quel `offset`, la garantie doit valoir partout.
    ///
    /// `depth` borne la mesure à ce qui est réellement servi. Le fond de liste
    /// se concentre forcément : quand il ne reste que les tweets de deux
    /// auteurs, aucun ordonnancement n'y peut rien. Personne ne descend à la
    /// 300ᵉ recommandation, et y garantir la variété coûterait l'ordre par
    /// score sur les pages que tout le monde lit.
    fn worst_window_concentration(out: &[String], raw: &[RawTweet], depth: usize) -> u32 {
        let author_of: HashMap<&str, &str> = raw
            .iter()
            .map(|t| (t.id.as_str(), t.user_id.as_str()))
            .collect();
        let head = &out[..depth.min(out.len())];
        head.windows(PAGE_WINDOW.min(head.len().max(1)))
            .map(|window| {
                let mut counts: HashMap<&str, u32> = HashMap::new();
                for id in window {
                    *counts.entry(author_of[id.as_str()]).or_insert(0) += 1;
                }
                counts.values().copied().max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn spread_caps_an_author_when_others_are_available() {
        // 6 tweets d'un compte prolifique en tête, puis 6 autres auteurs.
        // Sans étalement, les 6 premiers occupaient le haut de page.
        let mut raw: Vec<RawTweet> = (0..6).map(|i| by(&format!("s{i}"), "spammy")).collect();
        raw.extend((0..6).map(|i| by(&format!("o{i}"), &format!("autre{i}"))));
        let ids: Vec<String> = raw.iter().map(|t| t.id.clone()).collect();

        let out = spread_by_author(ids.clone(), &index(&raw));

        assert_eq!(out.len(), ids.len(), "aucun tweet ne doit être perdu");
        // Les 3 premiers de « spammy » passent, les suivants sont repoussés
        // derrière les autres auteurs.
        assert_eq!(&out[..3], &texts(&["s0", "s1", "s2"])[..]);
        assert_eq!(
            &out[3..9],
            &texts(&["o0", "o1", "o2", "o3", "o4", "o5"])[..]
        );
        assert_eq!(&out[9..], &texts(&["s3", "s4", "s5"])[..]);
    }

    #[test]
    fn spread_keeps_score_order_when_nothing_exceeds_quota() {
        let raw = vec![by("a1", "a"), by("b1", "b"), by("c1", "c"), by("a2", "a")];
        let ids = texts(&["a1", "b1", "c1", "a2"]);

        let out = spread_by_author(ids.clone(), &index(&raw));

        assert_eq!(
            out, ids,
            "sans dépassement de quota, l'ordre par score est conservé"
        );
    }

    #[test]
    fn spread_serves_everything_when_a_single_author_remains() {
        // Un seul auteur : le plafond est intenable sans laisser des trous.
        // La dégradation attendue est « on sert quand même », pas « on ampute ».
        let raw: Vec<RawTweet> = (0..10).map(|i| by(&format!("t{i}"), "seul")).collect();
        let ids: Vec<String> = raw.iter().map(|t| t.id.clone()).collect();

        let out = spread_by_author(ids.clone(), &index(&raw));

        assert_eq!(
            out, ids,
            "aucune perte, ordre conservé, faute d'alternative"
        );
    }

    /// Profondeur réellement servie qu'on protège : trois pages.
    const SERVED_DEPTH: usize = PAGE_WINDOW * 3;

    #[test]
    fn spread_holds_the_cap_on_served_pages_when_enough_authors() {
        // ⚠ Le plafond n'est tenable que s'il existe au moins
        // ceil(PAGE_WINDOW / MAX_PER_AUTHOR_PER_PAGE) auteurs distincts : à 3
        // tweets chacun, il en faut 17 pour garnir une fenêtre de 50. En dessous,
        // aucun ordonnancement n'y arrive — c'est de l'arithmétique, pas un
        // défaut d'implémentation (voir le test de dégradation ci-dessous).
        //
        // 300 tweets sur 20 auteurs, GROUPÉS par auteur en entrée : le pire cas
        // de concentration qu'on puisse soumettre à l'étalement.
        let raw: Vec<RawTweet> = (0..300)
            .map(|i| by(&format!("t{i}"), &format!("auteur{}", i / 15)))
            .collect();
        let ids: Vec<String> = raw.iter().map(|t| t.id.clone()).collect();

        let out = spread_by_author(ids.clone(), &index(&raw));

        let mut sorted_in = ids.clone();
        let mut sorted_out = out.clone();
        sorted_in.sort();
        sorted_out.sort();
        assert_eq!(
            sorted_in, sorted_out,
            "l'étalement réordonne, il ne filtre pas"
        );
        assert_eq!(
            worst_window_concentration(&out, &raw, SERVED_DEPTH),
            MAX_PER_AUTHOR_PER_PAGE,
            "exactement {MAX_PER_AUTHOR_PER_PAGE} par fenêtre de {PAGE_WINDOW} sur les pages servies",
        );
    }

    #[test]
    fn spread_degrades_proportionally_when_authors_are_too_few() {
        // 4 auteurs pour une fenêtre de 50 : 4 × 3 = 12 places, la fenêtre ne
        // peut pas être garnie sans dépasser le quota. Le repli doit alors
        // dégrader en tour de rôle — donc tendre vers PAGE_WINDOW / auteurs —
        // et surtout pas resservir les auteurs par blocs.
        let raw: Vec<RawTweet> = (0..40)
            .map(|i| by(&format!("t{i}"), &format!("auteur{}", i / 10)))
            .collect();
        let ids: Vec<String> = raw.iter().map(|t| t.id.clone()).collect();

        let out = spread_by_author(ids.clone(), &index(&raw));

        assert_eq!(out.len(), ids.len(), "aucune perte, même en dégradation");
        let worst = worst_window_concentration(&out, &raw, SERVED_DEPTH);
        // 40 tweets, 4 auteurs : le minimum théorique sur la liste entière est
        // 10 par auteur. On doit l'atteindre, pas le dépasser.
        assert_eq!(worst, 10, "dégradation en tour de rôle, pas en blocs");
    }

    fn reply(id: &str, parent: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            parent_tweet_id: Some(parent.to_string()),
            ..Default::default()
        }
    }

    fn reply_by(id: &str, parent: &str, author: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            user_id: author.to_string(),
            parent_tweet_id: Some(parent.to_string()),
            ..Default::default()
        }
    }

    /// Vérifie l'invariant dont dépend le rendu du fil côté client : toute
    /// réponse servie est IMMÉDIATEMENT précédée de son parent.
    fn assert_replies_follow_their_parent(out: &[String], raw: &[RawTweet]) {
        let by_id: HashMap<&str, &RawTweet> = raw.iter().map(|t| (t.id.as_str(), t)).collect();
        for (i, id) in out.iter().enumerate() {
            let Some(parent) = by_id
                .get(id.as_str())
                .and_then(|t| t.parent_tweet_id.as_deref())
            else {
                continue;
            };
            assert_eq!(
                out.get(i.wrapping_sub(1)).map(String::as_str),
                Some(parent),
                "la reponse {id} n'est pas collee a son parent {parent} : {out:?}",
            );
        }
    }

    #[test]
    fn le_lien_de_fil_decrit_la_chaine_complete() {
        let raw = vec![
            tweet("racine"),
            reply("mid", "racine"),
            reply("feuille", "mid"),
            tweet("autre"),
        ];
        let ids: Vec<String> = ["racine", "mid", "feuille", "autre"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let links = thread_links(&entries_for_test(&ids, &index(&raw)));

        assert_eq!(links.len(), 2, "deux reponses, deux liens : {links:?}");
        assert_eq!(links[0].tweet_id, "mid");
        assert_eq!(links[0].parent_id, "racine");
        assert_eq!(links[0].root_id, "racine");
        assert_eq!(links[0].depth, 1);
        // La racine se propage : `feuille` répond à `mid`, mais le fil part
        // toujours de `racine`.
        assert_eq!(links[1].tweet_id, "feuille");
        assert_eq!(links[1].parent_id, "mid");
        assert_eq!(links[1].root_id, "racine");
        assert_eq!(links[1].depth, 2);
    }

    #[test]
    fn un_parent_hors_page_ne_produit_aucun_lien() {
        // Cas de la coupure de pagination : la page commence par une réponse
        // dont le parent fermait la page précédente. Sans parent à l'écran, il
        // n'y a pas de fil à annoncer — surtout pas un lien vers un tweet que
        // le client n'a pas reçu.
        let raw = vec![reply("rep", "parent_hors_page"), tweet("autre")];
        let ids: Vec<String> = ["rep", "autre"].iter().map(|s| s.to_string()).collect();

        let entries = entries_for_test(&ids, &index(&raw));
        assert_eq!(
            entries[0].parent_id, None,
            "aucun parent adjacent : {entries:?}"
        );
        assert!(thread_links(&entries).is_empty());
    }

    #[test]
    fn l_etalement_ne_separe_jamais_une_reponse_de_son_parent() {
        // Le fil et sa réponse sont du MÊME auteur : c'est le cas qui cassait,
        // puisque le plafond par auteur pousse le second tweet plus bas.
        // Assez de tweets pour saturer plusieurs fenêtres et forcer l'étalement.
        let mut raw: Vec<RawTweet> = Vec::new();
        let mut ids: Vec<String> = Vec::new();
        for i in 0..30 {
            let parent = format!("p{i}");
            let child = format!("r{i}");
            raw.push(by(&parent, &format!("auteur{}", i % 3)));
            raw.push(reply_by(&child, &parent, &format!("auteur{}", i % 3)));
            ids.push(parent);
            ids.push(child);
        }

        let out = spread_by_author(ids.clone(), &index(&raw));

        assert_eq!(out.len(), ids.len(), "aucune perte a l'etalement");
        assert_replies_follow_their_parent(&out, &raw);
    }

    #[test]
    fn une_reponse_prolonge_le_fil_quand_son_parent_vient_d_etre_servi() {
        // Parent et réponse tous deux bien classés : le fil se lit de haut en
        // bas, la réponse ne doit pas être écartée sous prétexte que son
        // ancêtre est « déjà à l'écran ».
        let raw = vec![tweet("parent"), reply("rep", "parent"), tweet("autre")];
        let out = shape(&scored_of(&["parent", "rep", "autre"]), &raw);

        assert_eq!(out, vec!["parent", "rep", "autre"]);
        assert_replies_follow_their_parent(&out, &raw);
    }

    #[test]
    fn une_reponse_n_est_pas_reservie_loin_de_son_parent() {
        // `parent` est servi en tête, `rep` n'est bien classée que beaucoup plus
        // bas : la resservir là-bas la couperait de son contexte. Mieux vaut
        // l'écarter que produire une orpheline.
        let mut raw = vec![tweet("parent"), reply("rep", "parent")];
        let mut ids = vec!["parent".to_string()];
        for i in 0..8 {
            let id = format!("f{i}");
            raw.push(tweet(&id));
            ids.push(id);
        }
        ids.push("rep".to_string());

        let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let out = shape(&scored_of(&refs), &raw);

        assert!(
            !out.contains(&"rep".to_string()),
            "reponse resservie loin de son parent : {out:?}"
        );
        assert_replies_follow_their_parent(&out, &raw);
    }

    fn retweet(id: &str, original: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            original_tweet_id: Some(original.to_string()),
            is_retweet: true,
            ..Default::default()
        }
    }

    // ── Mise en sourdine d'un auteur refusé ──────────────────────────────────

    #[test]
    fn un_auteur_jamais_refuse_n_est_pas_touche() {
        assert_eq!(author_damping(0.0), 1.0);
    }

    #[test]
    fn un_refus_reduit_fortement_sans_effacer() {
        let one = author_damping(1.0);
        assert!(one < 0.4, "un refus doit se voir tout de suite : {one}");
        assert!(one > 0.0, "mais jamais effacer l'auteur : {one}");
    }

    #[test]
    fn les_refus_s_accumulent_et_gardent_un_plancher() {
        let a = author_damping(1.0);
        let b = author_damping(2.0);
        let c = author_damping(3.0);
        assert!(
            b < a && c < b,
            "chaque refus doit peser davantage : {a}, {b}, {c}"
        );
        for strikes in [5.0, 20.0, 1_000.0] {
            let d = author_damping(strikes);
            assert!(d > 0.0, "jamais zéro, sinon c'est un blocage : {d}");
            assert!(d.is_finite());
        }
    }

    // ── Tirage Trending à deux températures ──────────────────────────────────

    /// `n` tweets de score strictement décroissant, comme `score_all` les rend.
    fn ranked(n: usize) -> Vec<ScoredTweet> {
        (0..n)
            .map(|i| ScoredTweet {
                tweet_id: format!("t{i:03}"),
                score: 1.0 - (i as f64) * 0.001,
                breakdown: Default::default(),
                ctr_features: None,
            })
            .collect()
    }

    #[test]
    fn le_tirage_ne_perd_ni_ne_duplique_aucun_tweet() {
        let input = ranked(120);
        let expected: std::collections::HashSet<String> =
            input.iter().map(|s| s.tweet_id.clone()).collect();

        let out = trending_draw(input);

        assert_eq!(
            out.len(),
            120,
            "le tirage doit rendre autant de tweets qu'il en reçoit"
        );
        let got: std::collections::HashSet<String> =
            out.iter().map(|s| s.tweet_id.clone()).collect();
        assert_eq!(
            got, expected,
            "aucun tweet ne doit apparaître ni disparaître"
        );
    }

    #[test]
    fn l_ouverture_est_toujours_tiree_dans_le_haut_du_classement() {
        // Le tirage est aléatoire : on le répète pour ne pas valider un coup de
        // chance. L'invariant, lui, doit tenir à chaque fois.
        for _ in 0..40 {
            let out = trending_draw(ranked(200));
            for opening in out.iter().take(TRENDING_HOOK_SIZE) {
                let rank: usize = opening.tweet_id[1..].parse().unwrap();
                assert!(
                    rank < TRENDING_HOOK_POOL,
                    "une carte d'ouverture vient du rang {rank}, hors du vivier de tête"
                );
            }
        }
    }

    #[test]
    fn deux_tirages_ne_donnent_pas_la_meme_ouverture() {
        // Sans cette propriété on retombe sur le défaut d'origine : la même
        // page à chaque rafraîchissement. Le test échoue si CENT tirages
        // successifs rendent tous exactement la même ouverture.
        let reference: Vec<String> = trending_draw(ranked(200))
            .into_iter()
            .take(TRENDING_HOOK_SIZE)
            .map(|s| s.tweet_id)
            .collect();

        let varied = (0..100).any(|_| {
            let draw: Vec<String> = trending_draw(ranked(200))
                .into_iter()
                .take(TRENDING_HOOK_SIZE)
                .map(|s| s.tweet_id)
                .collect();
            draw != reference
        });
        assert!(varied, "l'ouverture doit changer d'un tirage à l'autre");
    }

    #[test]
    fn un_lot_plus_petit_que_l_ouverture_ne_panique_pas() {
        for n in 0..=3 {
            let out = trending_draw(ranked(n));
            assert_eq!(out.len(), n, "lot de {n} tweet(s)");
        }
    }

    fn scored_of(ids: &[&str]) -> Vec<ScoredTweet> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| ScoredTweet {
                tweet_id: id.to_string(),
                score: 1.0 - (i as f64) * 0.01,
                breakdown: Default::default(),
                ctr_features: None,
            })
            .collect()
    }

    fn shape(scored: &[ScoredTweet], raw: &[RawTweet]) -> Vec<String> {
        let map: HashMap<&str, &RawTweet> = raw.iter().map(|t| (t.id.as_str(), t)).collect();
        shape_feed(scored, &map)
    }

    /// Le parent d'une réponse ne doit PAS entrer dans le fil quand il a été
    /// écarté par le filtre d'admission.
    ///
    /// C'est le défaut trouvé en production le 2026-08-22 : `tweet_map` était
    /// construite sur TOUS les candidats, pas seulement les admis, donc
    /// `shape_feed` ramenait le parent en remontant la chaîne. Un compte
    /// `Ghosted` sortait ainsi 3 tweets dans le fil d'un lecteur qui ne le
    /// suivait pas — tirés par leurs réponses, par la porte de derrière.
    ///
    /// Ici, « écarté » se traduit par « absent de la carte » : c'est
    /// exactement ce que fait l'appelant depuis le correctif.
    #[test]
    fn un_parent_non_admis_ne_revient_pas_par_sa_reponse() {
        let raw = vec![tweet("racine"), reply("rep", "banni")];
        // « banni » n'est PAS dans `raw` : il a été refusé à l'admission.
        let out = shape(&scored_of(&["rep", "racine"]), &raw);

        assert!(
            !out.iter().any(|x| x == "banni"),
            "le parent ecarte est revenu dans le fil : {out:?}"
        );
        assert!(
            !out.iter().any(|x| x == "rep"),
            "la reponse doit tomber avec son parent (contexte incomplet) : {out:?}"
        );
        assert_eq!(out, vec!["racine".to_string()]);
    }

    #[test]
    fn une_reponse_est_precedee_de_son_parent() {
        let raw = vec![tweet("racine"), reply("rep", "parent"), tweet("parent")];
        let out = shape(&scored_of(&["rep", "racine"]), &raw);

        let i_parent = out
            .iter()
            .position(|x| x == "parent")
            .expect("parent remonte");
        let i_rep = out
            .iter()
            .position(|x| x == "rep")
            .expect("reponse presente");
        assert!(
            i_parent < i_rep,
            "le parent doit passer avant la reponse : {out:?}"
        );
    }

    #[test]
    fn une_reponse_dont_le_parent_est_absent_est_ecartee() {
        // `parent` n'est pas dans les candidats : il a été filtré en amont
        // (supprime, prive, auteur bloque…). La reponse ne doit pas passer.
        let raw = vec![tweet("racine"), reply("rep", "parent")];
        let out = shape(&scored_of(&["rep", "racine"]), &raw);

        assert!(
            !out.contains(&"rep".to_string()),
            "reponse orpheline servie : {out:?}"
        );
        assert!(
            !out.contains(&"parent".to_string()),
            "un parent filtre ne doit jamais reapparaitre : {out:?}"
        );
        assert_eq!(out, vec!["racine"]);
    }

    #[test]
    fn une_reponse_a_une_reponse_amene_tout_le_fil_dans_l_ordre() {
        // Cas observe en prod : le parent etait lui-meme une reponse et
        // arrivait sans son propre parent.
        // Le fil compte 2 reponses : il faut assez de candidats pour que le
        // plafond de 25 % les autorise, sinon c'est lui qui tranche.
        let mut raw = vec![
            tweet("racine"),
            reply("mid", "racine"),
            reply("feuille", "mid"),
        ];
        let mut ids = vec!["feuille".to_string()];
        for i in 0..7 {
            let id = format!("f{i}");
            raw.push(tweet(&id));
            ids.push(id);
        }
        let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let out = shape(&scored_of(&refs), &raw);

        let pos = |id: &str| {
            out.iter()
                .position(|x| x == id)
                .unwrap_or_else(|| panic!("{id} absent de {out:?}"))
        };
        assert!(pos("racine") < pos("mid"), "racine avant mid : {out:?}");
        assert!(pos("mid") < pos("feuille"), "mid avant feuille : {out:?}");
    }

    #[test]
    fn une_reponse_a_un_retweet_n_est_pas_confondue_avec_l_original() {
        // Les reponses portent aussi `original_tweet_id` : la deduplication ne
        // doit surtout pas les collapser sur la racine du fil.
        let mut r1 = reply("rep1", "racine");
        r1.original_tweet_id = Some("racine".into());
        let mut r2 = reply("rep2", "racine");
        r2.original_tweet_id = Some("racine".into());

        let deduped = deduplicate(vec![tweet("racine"), r1, r2]);
        let ids: Vec<&str> = deduped.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            3,
            "racine + 2 reponses distinctes attendues : {ids:?}"
        );
    }

    #[test]
    fn le_parent_n_est_pas_duplique_s_il_est_deja_plus_haut() {
        let raw = vec![tweet("parent"), reply("rep", "parent")];
        let out = shape(&scored_of(&["parent", "rep"]), &raw);

        assert_eq!(
            out.iter().filter(|x| *x == "parent").count(),
            1,
            "le parent ne doit apparaitre qu'une fois : {out:?}"
        );
        assert_eq!(out, vec!["parent", "rep"]);
    }

    #[test]
    fn les_reponses_sont_plafonnees() {
        // 8 candidats classés dont 6 réponses → plafond de 25 % = 2 réponses.
        let mut raw = vec![tweet("t1"), tweet("t2")];
        let mut ids = vec!["t1".to_string(), "t2".to_string()];
        for i in 0..6 {
            let rid = format!("r{i}");
            let pid = format!("p{i}");
            raw.push(reply(&rid, &pid));
            raw.push(tweet(&pid));
            ids.push(rid);
        }
        let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let out = shape(&scored_of(&refs), &raw);

        let n_replies = out.iter().filter(|id| id.starts_with('r')).count();
        assert!(
            n_replies <= 2,
            "trop de reponses retenues : {n_replies} dans {out:?}"
        );
    }

    #[test]
    fn aucune_reponse_orpheline_dans_la_sortie() {
        // Invariant global : tout tweet servi qui a un parent doit avoir ce
        // parent quelque part avant lui.
        let raw = vec![
            tweet("a"),
            reply("b", "a"),
            reply("c", "b"),
            reply("orphelin", "inconnu"),
            tweet("d"),
        ];
        let out = shape(&scored_of(&["c", "orphelin", "d", "a", "b"]), &raw);

        let pos: HashMap<&str, usize> = out
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let by_id: HashMap<&str, &RawTweet> = raw.iter().map(|t| (t.id.as_str(), t)).collect();

        for (id, i) in &pos {
            if let Some(parent) = by_id.get(id).and_then(|t| t.parent_tweet_id.as_deref()) {
                let pp = pos
                    .get(parent)
                    .unwrap_or_else(|| panic!("parent {parent} absent pour {id} : {out:?}"));
                assert!(pp < i, "parent {parent} apres son enfant {id} : {out:?}");
            }
        }
        assert!(!out.contains(&"orphelin".to_string()));
    }

    #[test]
    fn un_retweet_du_meme_tweet_ne_passe_qu_une_fois() {
        // L'original et deux retweets de ce même original.
        let deduped = deduplicate(vec![
            retweet("rt1", "origine"),
            tweet("origine"),
            retweet("rt2", "origine"),
        ]);

        assert_eq!(
            deduped.len(),
            1,
            "un seul contenu attendu : {:?}",
            deduped.iter().map(|t| &t.id).collect::<Vec<_>>()
        );
        // L'original doit être conservé plutôt qu'un de ses retweets.
        assert_eq!(deduped[0].id, "origine");
    }

    #[test]
    fn deux_tweets_distincts_sont_conserves() {
        let deduped = deduplicate(vec![tweet("a"), tweet("b"), retweet("rt", "c"), tweet("c")]);
        let ids: Vec<&str> = deduped.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids.len(), 3, "a, b et c attendus : {ids:?}");
        assert!(ids.contains(&"a") && ids.contains(&"b") && ids.contains(&"c"));
    }

    #[test]
    fn top_words_ignore_les_mots_vides_et_les_hapax() {
        let liked = texts(&[
            "le casino avec la roue et les jetons dans le casino",
            "encore une soiree casino, la roue tourne pour tout le monde",
            "jetons perdus au casino ce soir",
        ]);
        let (words, _, _) = profile_from_liked_text(&liked);
        let map: std::collections::HashMap<_, _> = words.iter().cloned().collect();

        // « casino » revient 4 fois → centre d'intérêt retenu.
        assert_eq!(map.get("casino"), Some(&4));
        assert_eq!(map.get("jetons"), Some(&2));
        // Mots vides écartés même s'ils sont fréquents.
        assert!(!map.contains_key("dans"));
        assert!(!map.contains_key("pour"));
        // Vu une seule fois → bruit, pas un intérêt.
        assert!(!map.contains_key("soiree"));
    }

    #[test]
    fn profil_vide_ne_panique_pas_et_reste_neutre() {
        let (words, personality, positivity) = profile_from_liked_text(&[]);
        assert!(words.is_empty());
        assert!(matches!(personality, PersonalityType::Balanced));
        assert_eq!(positivity, 0.5);
    }

    #[test]
    fn personnalite_deduite_du_style_des_tweets_aimes() {
        let enthousiaste = texts(&["trop bien !! 🎉🔥", "enorme !! 😍", "wow !! 🚀🎰"]);
        assert!(matches!(
            profile_from_liked_text(&enthousiaste).1,
            PersonalityType::Enthusiastic
        ));

        let curieux = texts(&[
            "comment ca marche ? source https://exemple.fr",
            "pourquoi ce choix ? des donnees ?",
            "quelqu'un sait ? https://exemple.fr/doc",
        ]);
        assert!(matches!(
            profile_from_liked_text(&curieux).1,
            PersonalityType::Curious
        ));
    }

    #[test]
    fn positivite_augmente_avec_les_emojis_mais_reste_bornee() {
        let neutre = profile_from_liked_text(&texts(&["analyse posee du sujet"])).2;
        let joyeux = profile_from_liked_text(&texts(&["super 🎉🔥😍🚀 !!!"])).2;
        assert!(joyeux > neutre);
        assert!((0.0..=1.0).contains(&joyeux));
    }

    #[test]
    fn fatigue_impression_laisse_une_seconde_chance_puis_decroit() {
        use crate::algorithm::scoring::impression_fatigue;
        // Jusqu'à 2 expositions : aucune pénalité.
        assert_eq!(impression_fatigue(0), 1.0);
        assert_eq!(impression_fatigue(2), 1.0);
        // Ensuite décroissance stricte.
        assert!(impression_fatigue(3) < 1.0);
        assert!(impression_fatigue(4) < impression_fatigue(3));
        // Jamais un bannissement définitif.
        assert!(impression_fatigue(500) >= 0.05);
    }
}
