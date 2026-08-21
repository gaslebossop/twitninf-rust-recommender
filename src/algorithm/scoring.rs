/// NeuralRank Fusion — Moteur de scoring 8 dimensions + modificateurs
///
/// Contrairement au moteur JS qui utilise Math.random() pour le "ML",
/// chaque dimension est calculée à partir des données réelles extraites de la BDD.
///
/// Architecture en couches :
///   Phase 1 → Engagement Velocity     (vélocité réelle sur fenêtres 1h/6h/24h)
///   Phase 2 → Content Intelligence    (analyse textuelle réelle du contenu)
///   Phase 3 → Social Graph Dynamics   (graphe social multi-degrés)
///   Phase 4 → Temporal Dynamics       (patterns temporels réels)
///   Phase 5 → Behavioral Prediction   (historique réel de l'utilisateur)
///   Phase 6 → Content Diversity       (MMR - Maximal Marginal Relevance)
///   Phase 7 → Viral Prediction        (accélération de viralité)
///   Phase 8 → Personalization Depth   (affinités calculées)
///   Mod A   → Diversity Multiplier    (anti-bulle de filtre)
///   Mod B   → Moderation Penalty      (sécurité contenu)
use chrono::Utc;
use tracing::{debug, trace};

use crate::admin::AlgoWeights;
use crate::algorithm::d9_llm_understanding::{
    d9_llm_understanding as calculate_d9, quality_boost, toxicity_penalty,
};
use crate::algorithm::dwell::{MAX_BONUS as DWELL_MAX, SKIP_PENALTY as DWELL_MIN};
use crate::ml::ctr_predictor::{extract_features, CtrPredictor, N_FEATURES};
use crate::ml::dwell_predictor::DwellPredictor;
use crate::ml::user_weights::UserDimensionWeights;
use crate::models::{
    AuthorTier, ContentLength, PersonalityType, RawTweet, ScoreBreakdown, ScoredTweet, TweetSource,
    UserProfile, UserType,
};
use crate::shadowban::{GarbageContentDetector, ShadowbanEnforcer};

// Les poids par défaut des 8 dimensions vivent dans `AlgoWeights::default()`
// (source unique, aussi utilisée par l'admin et l'auto-tuner). Les constantes
// dupliquées qui traînaient ici pouvaient diverger silencieusement de ce
// défaut : deux barèmes pour le même calcul.

/// Au-delà de ce nombre d'expositions sans engagement, un tweet est considéré
/// comme refusé implicitement par le lecteur.
const IMPRESSION_FATIGUE_START: i64 = 2;

/// Coup de pouce de visibilité des comptes abonnés (Mod I).
///
/// Volontairement MINUSCULE : c'est un départage, pas un classement parallèle.
/// À +3 %/+6 %, un tweet d'abonné passe devant un tweet gratuit de qualité
/// équivalente, mais aucun contenu médiocre ne remonte au-dessus d'un bon —
/// l'écart entre deux tweets réellement différents se compte en dizaines de
/// pour cent sur ces dimensions. Le reste des modificateurs (fatigue,
/// shadowban, toxicité) continue de s'appliquer par-dessus, donc payer
/// n'achète pas l'immunité.
const SUBSCRIPTION_BOOST_PLUS: f64 = 1.03;
const SUBSCRIPTION_BOOST_PRO: f64 = 1.06;

/// Score final d'un tweet, toutes dimensions combinées.
/// `author_feed_count` = combien de fois cet auteur apparaît déjà dans le feed courant.
/// `ctr_predictor`     = modèle ML optionnel pour blending CTR prédit (Phase 2)
/// `realtime_boost`    = delta feedback loop Redis (Phase 3), 0.0 si non disponible
pub fn score_tweet(
    tweet: &RawTweet,
    profile: &UserProfile,
    author_feed_count: u32,
    feed_tweets_so_far: &[ScoredTweet],
) -> ScoredTweet {
    // Délègue à la variante pondérée avec les poids par défaut : les deux
    // fonctions calculaient auparavant la même chose en double, et toute
    // correction apportée à l'une manquait à l'autre.
    score_tweet_with_weights(
        tweet,
        profile,
        author_feed_count,
        feed_tweets_so_far,
        &AlgoWeights::default(),
    )
}

/// Variante de `score_tweet` avec poids de dimensions injectés (admin/auto-tuner).
pub fn score_tweet_with_weights(
    tweet: &RawTweet,
    profile: &UserProfile,
    author_feed_count: u32,
    feed_tweets_so_far: &[ScoredTweet],
    weights: &AlgoWeights,
) -> ScoredTweet {
    let (d1, ev_raw, ev_accel, viral_vel) = d1_engagement_velocity(tweet);
    let d2 = d2_content_intelligence(tweet, profile);
    let (d3, d_follow, d_mutual, d_second) = d3_social_graph(tweet, profile);
    let d4 = d4_temporal_dynamics(tweet, profile);
    let d5 = d5_behavioral_prediction(tweet, profile);
    let d6 = d6_content_diversity(tweet, profile, feed_tweets_so_far);
    let d7_raw = d7_viral_prediction(tweet);
    let d7 = if tweet.source == TweetSource::Trending {
        (d7_raw * 1.2).clamp(0.0, 1.0)
    } else {
        d7_raw
    };
    let d8 = d8_personalization_depth(tweet, profile);
    // D9 juge le texte lui-même via les étiquettes du LLM annotateur. Neutre
    // (0.5) tant qu'un tweet n'est pas annoté, donc sans effet sur le classement
    // existant.
    let d9 = calculate_d9(tweet);

    let user_weights = UserDimensionWeights::for_profile(profile);
    let user_weighted_score = user_weights.apply(d1, d2, d3, d4, d5, d6, d7, d8);

    // Pondération avec les poids actifs (admin/auto-tuner) au lieu des constantes
    let w = weights;
    let global_score = d1 * w.d1_engagement_velocity
        + d2 * w.d2_content_intelligence
        + d3 * w.d3_social_graph
        + d4 * w.d4_temporal
        + d5 * w.d5_behavioral
        + d6 * w.d6_diversity
        + d7 * w.d7_viral
        + d8 * w.d8_personalization
        + d9 * w.d9_llm_understanding;
    let base_score = global_score * 0.60 + user_weighted_score * 0.40;

    let diversity_mult = diversity_multiplier(author_feed_count);
    let mod_penalty = moderation_penalty(tweet);
    let source_bonus = match tweet.source {
        TweetSource::SocialGraph => 0.08,
        TweetSource::Personalized => 0.05,
        TweetSource::Viral => 0.04,
        TweetSource::Influencer => 0.03,
        TweetSource::Trending => 0.04,
        TweetSource::Temporal => 0.02,
        TweetSource::Discovery => 0.01,
        TweetSource::Quality => 0.01,
    };

    let score_before_shadowban =
        ((base_score + source_bonus) * diversity_mult + mod_penalty).clamp(0.0, 1.0);

    let enforcer = ShadowbanEnforcer::new();
    let shadowban_mult = enforcer.multiplier(tweet.author_shadowban_level);
    let score_after_shadowban =
        enforcer.apply_to_score(score_before_shadowban, tweet.author_shadowban_level);

    let garbage_signals = GarbageContentDetector::new().detect(tweet);
    let garbage_score = garbage_signals.score();
    let garbage_penalty = 1.0 - 0.60 * garbage_score;

    // ── Mod F : fatigue d'exposition ─────────────────────────────────────────
    // Un tweet déjà servi plusieurs fois à ce lecteur sans qu'il l'engage est
    // un refus implicite — c'est le signal le plus dense dont on dispose
    // (`user_behavior_data` compte 5× plus d'impressions que de likes). Le
    // lecteur reste libre de le revoir une fois : la pénalité ne démarre qu'à
    // partir de la 3e exposition, puis décroît géométriquement.
    let fatigue = impression_fatigue(tweet.viewer_impressions);

    // ── Mod G : levier de visibilité par compte ──────────────────────────────
    // `users.algorithmic_visibility_multiplier` existait en base et pilotait
    // déjà l'affichage ailleurs, mais aucun scoring ne le lisait : le réglage
    // n'avait donc aucun effet sur les recommandations.
    let visibility = tweet.author_visibility_multiplier.clamp(0.0, 3.0);

    // ── Mod H : toxicité détectée par le LLM ─────────────────────────────────
    // Multiplicateur séparé de D9, et non un terme de son score : mettre le
    // poids de D9 à zéro dans le panel admin ne doit pas désactiver au passage
    // la rétrogradation du contenu agressif.
    let toxicity_mult = toxicity_penalty(tweet);

    // ── Mod I : coup de pouce d'abonné ───────────────────────────────────────
    // Appliqué en dernier, sur le score déjà pénalisé : un compte abonné dont
    // le tweet est toxique ou sur-exposé reste rétrogradé, le boost porte alors
    // sur une valeur déjà écrasée.
    let subscription_boost = subscription_boost(tweet.author_tier);

    // ── Mod J : appui de qualité ─────────────────────────────────────────────
    // Pendant de Mod H : la toxicité rétrograde, la qualité relève. Multiplicatif
    // et non additif, donc il n'exhume pas un tweet par ailleurs pénalisé — un
    // contenu sur-exposé ou toxique reste écrasé, même bien écrit.
    let quality_mult = quality_boost(tweet);

    let final_score = (score_after_shadowban
        * garbage_penalty
        * fatigue
        * visibility
        * toxicity_mult
        * quality_mult
        * subscription_boost)
        .clamp(0.0, 1.0);

    if tweet.viewer_impressions > 0 || (visibility - 1.0).abs() > f64::EPSILON {
        trace!(tweet_id = %tweet.id, impressions = tweet.viewer_impressions, fatigue,
               visibility, "Mod F/G appliqués");
    }

    ScoredTweet {
        tweet_id: tweet.id.clone(),
        score: final_score,
        breakdown: ScoreBreakdown {
            engagement_velocity: d1,
            content_intelligence: d2,
            social_graph_dynamics: d3,
            temporal_dynamics: d4,
            behavioral_prediction: d5,
            content_diversity: d6,
            viral_prediction: d7,
            personalization_depth: d8,
            engagement_velocity_raw: ev_raw,
            engagement_acceleration: ev_accel,
            viral_velocity: viral_vel,
            direct_follow_boost: d_follow,
            mutual_follow_boost: d_mutual,
            second_degree_boost: d_second,
            diversity_multiplier: diversity_mult,
            moderation_penalty: mod_penalty,
            source_weight: tweet.source_weight,
            shadowban_multiplier: shadowban_mult,
            garbage_penalty,
            subscription_boost,
            has_media: tweet.has_media,
            final_score,
        },
        ctr_features: None,
    }
}

/// Version avec ML CTR Predictor (Phase 2) + realtime boost (Phase 3)
/// `ctr`           = CtrPredictor optionnel (None → score classique)
/// `realtime_boost`= delta feedback loop Redis, typiquement [-0.20, +0.20]
pub fn score_tweet_ml(
    tweet: &RawTweet,
    profile: &UserProfile,
    author_feed_count: u32,
    feed_tweets_so_far: &[ScoredTweet],
    ctr: Option<&CtrPredictor>,
    realtime_boost: f64,
) -> ScoredTweet {
    // Score de base (Phase 1 + Phase 2 user weights)
    let mut scored = score_tweet(tweet, profile, author_feed_count, feed_tweets_so_far);

    // Les features sont extraites systématiquement, même quand le blending ML
    // est encore désactivé : c'est ce vecteur qu'on mémorise pour entraîner le
    // modèle. Le gater sur `use_ml` créait un blocage circulaire — pas de
    // features stockées → pas d'entraînement → jamais assez de samples pour
    // activer le ML.
    let features = ctr_feature_vector(tweet, profile, &scored);
    scored.ctr_features = Some(features.to_vec());

    // Phase 2 : blending ML CTR predictor (40% ML + 60% règles)
    if let Some(ctr_model) = ctr {
        let ml_ctr = ctr_model.predict_ctr(&features);
        let blended = scored.score * 0.60 + ml_ctr * 0.40;
        debug!(
            base = scored.score,
            ml_ctr, blended, "Phase 2: ML CTR blend (60% rules + 40% ML)"
        );
        scored.score = blended;
    }

    // Phase 3 : realtime feedback boost/penalty
    if realtime_boost.abs() > 0.001 {
        scored.score = (scored.score + realtime_boost).clamp(0.0, 1.0);
        debug!(realtime_boost, final = scored.score, "Phase 3: Realtime feedback applied");
    }

    scored.breakdown.final_score = scored.score;
    scored
}

/// Variante ML avec poids de dimensions injectés (admin/auto-tuner).
///
/// `ctr` et `dwell` sont chacun gardés par leur PROPRE seuil de démarrage à
/// froid (voir `use_ml`/`use_dwell` dans `services::recommender`) — un modèle
/// tout juste réinitialisé ne doit jamais peser dans le mélange. Les poids du
/// mélange s'ajustent selon ce qui est réellement disponible ; les règles
/// gardent toujours au moins 50 % : un signal appris ne doit pas dominer une
/// dimension qui n'a encore rien appris à côté de lui.
pub fn score_tweet_ml_with_weights(
    tweet: &RawTweet,
    profile: &UserProfile,
    author_feed_count: u32,
    feed_tweets_so_far: &[ScoredTweet],
    ctr: Option<&CtrPredictor>,
    dwell: Option<&DwellPredictor>,
    realtime_boost: f64,
    weights: &AlgoWeights,
) -> ScoredTweet {
    let mut scored = score_tweet_with_weights(
        tweet,
        profile,
        author_feed_count,
        feed_tweets_so_far,
        weights,
    );

    let features = ctr_feature_vector(tweet, profile, &scored);
    scored.ctr_features = Some(features.to_vec());

    let ml_ctr = ctr.map(|m| m.predict_ctr(&features));
    // Le dwell prédit vit sur [SKIP_PENALTY, MAX_BONUS] (voir `algorithm::dwell`),
    // pas [0, 1] comme le CTR et le score de règles — reprojeté pour mélanger
    // sur la même échelle que les deux autres termes.
    let ml_dwell01 = dwell.map(|m| {
        let predicted = m.predict_dwell(&features);
        ((predicted - DWELL_MIN) / (DWELL_MAX - DWELL_MIN)).clamp(0.0, 1.0)
    });

    scored.score = match (ml_ctr, ml_dwell01) {
        (Some(c), Some(d)) => (scored.score * 0.50 + c * 0.30 + d * 0.20).clamp(0.0, 1.0),
        (Some(c), None) => (scored.score * 0.60 + c * 0.40).clamp(0.0, 1.0),
        (None, Some(d)) => (scored.score * 0.70 + d * 0.30).clamp(0.0, 1.0),
        (None, None) => scored.score,
    };

    if realtime_boost.abs() > 0.001 {
        scored.score = (scored.score + realtime_boost).clamp(0.0, 1.0);
    }

    scored.breakdown.final_score = scored.score;
    scored
}

/// Construit le vecteur de features CTR d'un tweet déjà scoré.
///
/// Source unique : la prédiction, la mémorisation de l'impression et
/// l'entraînement doivent tous partir d'ici. Deux constructions parallèles du
/// vecteur, c'était précisément le bug — le modèle s'entraînait sur des
/// constantes et prédisait sur les vraies dimensions.
// `pub(crate)` : le rattrapage hors-ligne du modèle CTR
// (`RecommenderService::backfill_ctr_model`) doit reconstruire EXACTEMENT le
// même vecteur que le chemin de service — une seconde implémentation aurait
// pu diverger de celle-ci sans qu'aucun test ne le remarque.
pub(crate) fn ctr_feature_vector(
    tweet: &RawTweet,
    profile: &UserProfile,
    scored: &ScoredTweet,
) -> [f64; N_FEATURES] {
    let age_h = age_hours(tweet);
    let (d1, _, ev_accel, _) = d1_engagement_velocity(tweet);
    // 20 likes/jour ≈ un lecteur très actif : `engagement_velocity` est un
    // compte brut (`SUM(... INTERVAL '1 day')`), sans plafond naturel.
    let reader_engagement = (profile.engagement_velocity / 20.0).clamp(0.0, 1.0);
    extract_features(
        d1,
        scored.breakdown.content_intelligence,
        scored.breakdown.social_graph_dynamics,
        scored.breakdown.temporal_dynamics,
        scored.breakdown.behavioral_prediction,
        scored.breakdown.content_diversity,
        scored.breakdown.viral_prediction,
        scored.breakdown.personalization_depth,
        age_h,
        tweet.source == TweetSource::Trending,
        tweet.has_media,
        tweet.author_followers,
        ev_accel,
        reader_engagement,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// D1 — ENGAGEMENT VELOCITY (25%)
// Mesure la vitesse d'accumulation d'engagement sur 3 fenêtres temporelles.
// Calcule aussi l'accélération (tendance haussière vs stable).
// ═══════════════════════════════════════════════════════════════════════════════
fn d1_engagement_velocity(t: &RawTweet) -> (f64, f64, f64, f64) {
    let age_h = age_hours(t).max(0.1);
    trace!(age_h, "D1 Age in hours");

    // Score brut pondéré par valeur d'action
    let eng_total = t.like_count     as f64 * 1.0
        + t.comment_count  as f64 * 3.5   // commentaires = signal fort
        + t.retweet_count  as f64 * 5.0   // retweet = amplification
        + t.share_count    as f64 * 4.0
        + t.bookmark_count as f64 * 2.5
        + t.view_count     as f64 * 0.05; // vues : volume élevé → faible poids
    trace!(
        likes = t.like_count,
        comments = t.comment_count,
        retweets = t.retweet_count,
        shares = t.share_count,
        bookmarks = t.bookmark_count,
        views = t.view_count,
        eng_total,
        "D1 Raw engagement total"
    );

    // Phase 1: boost multiplicatif pour les tweets très récents (CTR signal fort)
    let recency_boost = if age_h < 0.5 {
        1.5 // tweets < 30 min : engagement en cours → CTR maximal
    } else if age_h < 2.0 {
        1.3 // tweets < 2h : momentum fort
    } else {
        1.0
    };
    trace!(
        recency_boost,
        age_h,
        "D1 Recency boost multiplier (Phase 1)"
    );

    // Vélocité = engagement / âge pondéré logarithmiquement + recency boost
    let velocity_raw = (eng_total / (1.0 + (age_h / 6.0).ln().max(0.0) * 2.0)) * recency_boost;
    trace!(velocity_raw, "D1 Raw velocity (engagement/age)");

    // Engagement récent (1h) : signal d'accélération
    let recent_eng =
        t.likes_1h as f64 * 1.0 + t.comments_1h as f64 * 3.5 + t.retweets_1h as f64 * 5.0;
    trace!(
        recent_eng,
        likes_1h = t.likes_1h,
        comments_1h = t.comments_1h,
        retweets_1h = t.retweets_1h,
        "D1 Recent engagement (1h)"
    );

    // Accélération : compare 1h à 6h pour détecter montée en puissance
    let eng_6h =
        t.likes_6h as f64 * 1.0 + (t.comment_count.min(t.comments_1h * 6 + 1)) as f64 * 3.5;

    let acceleration = if eng_6h > 0.0 {
        // ratio 1h annualisé vs 6h : si > 1.0, trending en hausse
        ((recent_eng * 6.0) / eng_6h).clamp(0.0, 5.0) / 5.0
    } else if recent_eng > 0.0 {
        1.0 // entièrement récent → accélération maximale
    } else {
        0.0
    };
    trace!(acceleration, eng_6h, "D1 Acceleration (recent/6h ratio)");

    // Vitesse de viralité = engagement / heure, seuil calibré à 50 eng/h
    let viral_vel = sigmoid(velocity_raw / (age_h * 50.0));
    trace!(viral_vel, "D1 Viral velocity sigmoid");

    // Score D1 composite
    let ev_raw = sigmoid(velocity_raw / 200.0);
    let d1 = (ev_raw * 0.50 + acceleration * 0.30 + viral_vel * 0.20).clamp(0.0, 1.0);
    debug!(
        d1,
        ev_raw,
        acceleration,
        viral_vel,
        "D1 Final (50% raw_velocity + 30% acceleration + 20% viral_vel)"
    );

    (d1, ev_raw, acceleration, viral_vel)
}

// ═══════════════════════════════════════════════════════════════════════════════
// D2 — CONTENT INTELLIGENCE (20%)
// Analyse réelle du contenu : longueur idéale, richesse, format, style,
// correspondance avec les préférences de l'utilisateur.
// ═══════════════════════════════════════════════════════════════════════════════
fn d2_content_intelligence(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0_f64;
    trace!(content_length = t.content_length, personality = ?profile.personality_type, "D2 Start");

    // ── Longueur idéale ──────────────────────────────────────────────────────
    let len = t.content_length as f64;
    let len_score = match profile.preferred_content_length {
        ContentLength::Short => {
            if len < 80.0 {
                1.0
            } else {
                1.0 - ((len - 80.0) / 200.0).clamp(0.0, 0.8)
            }
        }
        ContentLength::Medium => gaussian(len, 150.0, 80.0),
        ContentLength::Long => {
            if len > 200.0 {
                1.0
            } else {
                len / 200.0
            }
        }
    };
    score += len_score * 0.25;
    trace!(len_score, "D2 Length score (25% weight)");

    // ── Richesse du contenu ──────────────────────────────────────────────────
    let mut richness_score = 0.0;
    if t.has_media {
        richness_score += 0.20;
        trace!("D2 Media +0.20");
    }
    let hashtag_score = (t.hashtag_count as f64 * 0.04).min(0.12);
    richness_score += hashtag_score;
    trace!(
        hashtag_count = t.hashtag_count,
        hashtag_score,
        "D2 Hashtags"
    );
    let mention_score = (t.mention_count as f64 * 0.03).min(0.09);
    richness_score += mention_score;
    trace!(
        mention_count = t.mention_count,
        mention_score,
        "D2 Mentions"
    );
    score += richness_score;

    // ── Style correspondant au profil ────────────────────────────────────────
    let personality_score = match &profile.personality_type {
        PersonalityType::Enthusiastic => {
            // Aime les contenus avec émojis et exclamations
            let emoji_bonus = (t.emoji_count as f64 * 0.03).min(0.10);
            let excl_bonus = (t.exclamation_count as f64 * 0.03).min(0.06);
            let ps = emoji_bonus + excl_bonus;
            trace!(
                emoji_count = t.emoji_count,
                emoji_bonus,
                exclamation_count = t.exclamation_count,
                excl_bonus,
                "D2 Enthusiastic"
            );
            ps
        }
        PersonalityType::Curious => {
            // Aime les questions, les URLs (liens sources)
            let question_bonus = (t.question_count as f64 * 0.04).min(0.10);
            let url_bonus = (t.url_count as f64 * 0.03).min(0.06);
            let ps = question_bonus + url_bonus;
            trace!(
                question_count = t.question_count,
                question_bonus,
                url_count = t.url_count,
                url_bonus,
                "D2 Curious"
            );
            ps
        }
        PersonalityType::Thoughtful => {
            // Aime les contenus longs, les URLs
            let long_bonus = if len > 150.0 { 0.10 } else { 0.0 };
            let url_bonus = (t.url_count as f64 * 0.04).min(0.08);
            let ps = long_bonus + url_bonus;
            trace!(
                long_bonus,
                url_count = t.url_count,
                url_bonus,
                "D2 Thoughtful"
            );
            ps
        }
        PersonalityType::Balanced => {
            // Polyvalent : léger bonus pour tout
            let emoji_bonus = (t.emoji_count as f64 * 0.01).min(0.05);
            let hashtag_bonus = (t.hashtag_count as f64 * 0.02).min(0.05);
            let ps = emoji_bonus + hashtag_bonus;
            trace!(emoji_bonus, hashtag_bonus, "D2 Balanced");
            ps
        }
    };
    score += personality_score;

    // ── Correspondance mots-clés préférés ────────────────────────────────────
    let content_lower = t.content.to_lowercase();
    let keyword_matches: usize = profile
        .top_words
        .iter()
        .filter(|(word, _)| content_lower.contains(word.as_str()))
        .count();
    let keyword_score = (keyword_matches as f64 * 0.03).min(0.15);
    trace!(keyword_matches, keyword_score, "D2 Keyword matches");
    score += keyword_score;

    let final_d2 = score.clamp(0.0, 1.0);
    debug!(
        final_d2,
        len_score, richness_score, personality_score, keyword_score, "D2 Final"
    );
    final_d2
}

// ═══════════════════════════════════════════════════════════════════════════════
// D3 — SOCIAL GRAPH DYNAMICS (15%)
// Proximité sociale multi-degrés : suivi direct, mutuel, amis d'amis.
// ═══════════════════════════════════════════════════════════════════════════════
fn d3_social_graph(t: &RawTweet, profile: &UserProfile) -> (f64, f64, f64, f64) {
    let mut score = 0.0_f64;
    trace!(author_id = %t.user_id, "D3 Social graph analysis");

    // Degré 1 : l'utilisateur suit directement l'auteur
    let direct = if profile.follows(&t.user_id) {
        0.55
    } else {
        0.0
    };
    score += direct;
    trace!(
        direct,
        is_following = profile.follows(&t.user_id),
        "D3 Degree 1: Direct follow"
    );

    // Degré 1.5 : follow mutuel (plus fort signal d'engagement)
    let mutual = if profile.is_mutual(&t.user_id) {
        0.25
    } else {
        0.0
    };
    score += mutual;
    trace!(
        mutual,
        is_mutual = profile.is_mutual(&t.user_id),
        "D3 Degree 1.5: Mutual follow"
    );

    // Degré 2 : ami d'ami (signal de découverte sociale)
    let second = if profile.is_second_degree(&t.user_id) {
        0.12
    } else {
        0.0
    };
    score += second;
    trace!(
        second,
        is_second_degree = profile.is_second_degree(&t.user_id),
        "D3 Degree 2: Second degree"
    );

    // Bonus si l'utilisateur a déjà interagi positivement avec l'auteur
    let prior_affinity = profile
        .top_authors
        .iter()
        .find(|(uid, _)| uid == &t.user_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let affinity_bonus = (prior_affinity * 0.20).min(0.20);
    score += affinity_bonus;
    trace!(prior_affinity, affinity_bonus, "D3 Prior author affinity");

    // Influence de l'auteur dans le réseau (followers de l'auteur)
    let author_influence = sigmoid((t.author_followers as f64).ln().max(0.0) / 10.0);
    let influence_bonus = author_influence * 0.08;
    score += influence_bonus;
    trace!(
        author_followers = t.author_followers,
        author_influence,
        influence_bonus,
        "D3 Author network influence"
    );

    let final_d3 = score.clamp(0.0, 1.0);
    debug!(
        final_d3,
        direct, mutual, second, "D3 Final (direct + mutual + second degree)"
    );
    (final_d3, direct, mutual, second)
}

// ═══════════════════════════════════════════════════════════════════════════════
// D4 — TEMPORAL DYNAMICS (10%)
// Pertinence temporelle réelle : récence + alignement heure d'activité
// + momentum de tendance + patterns jour/heure.
// ═══════════════════════════════════════════════════════════════════════════════
fn d4_temporal_dynamics(t: &RawTweet, profile: &UserProfile) -> f64 {
    let age_h = age_hours(t);
    trace!(age_h, "D4 Tweet age in hours");

    // Phase 1: demi-vie réduite de 6h → 4h pour favoriser contenu frais (+CTR)
    let recency = (-0.173 * age_h).exp(); // ln(2)/4h ≈ 0.173
    trace!(recency, "D4 Recency (4h half-life, Phase 1 optimization)");

    // Alignement heure d'activité utilisateur
    let pub_hour = t
        .created_at
        .format("%H")
        .to_string()
        .parse::<u32>()
        .unwrap_or(12) as usize;
    let user_hour_activity = profile.hourly_activity[pub_hour];
    // user_hour_activity est déjà normalisé [0,1]
    let hour_match = user_hour_activity;
    trace!(pub_hour, hour_match, "D4 Hourly activity match");

    // Alignement jour de la semaine
    let pub_day = t
        .created_at
        .format("%w")
        .to_string()
        .parse::<usize>()
        .unwrap_or(1);
    let day_match = profile.daily_activity[pub_day];
    trace!(pub_day, day_match, "D4 Daily activity match");

    // Momentum : score plus élevé si tweet très récent ET engagé
    let momentum = if age_h < 2.0 {
        let eng_total = t.like_count + t.comment_count + t.retweet_count;
        sigmoid(eng_total as f64 / 20.0) * 1.5
    } else if age_h < 6.0 {
        1.0
    } else {
        0.7
    };
    trace!(momentum, "D4 Momentum (recent + engaged)");

    let d4 = (recency * 0.45 + hour_match * 0.25 + day_match * 0.15 + momentum.min(1.0) * 0.15)
        .clamp(0.0, 1.0);
    debug!(
        d4,
        recency,
        hour_match,
        day_match,
        momentum,
        "D4 Final (45% recency + 25% hour + 15% day + 15% momentum)"
    );
    d4
}

// ═══════════════════════════════════════════════════════════════════════════════
// D5 — BEHAVIORAL PREDICTION (10%)
// Prédit la probabilité d'engagement basée sur l'historique réel.
// ═══════════════════════════════════════════════════════════════════════════════
fn d5_behavioral_prediction(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0_f64;
    trace!(user_type = ?profile.user_type, "D5 Behavioral prediction start");

    // Probabilité d'engagement basée sur la vélocité (utilisateurs actifs voient plus)
    let activity_bonus = match profile.user_type {
        UserType::PowerUser => 0.20,
        UserType::Regular => 0.12,
        UserType::Casual => 0.05,
    };
    score += activity_bonus;
    trace!(activity_bonus, "D5 User activity type bonus");

    // L'utilisateur engage-t-il souvent avec des médias ? → boost si tweet avec média
    let media_bonus = if profile.prefers_media && t.has_media {
        0.18
    } else {
        0.0
    };
    score += media_bonus;
    trace!(
        media_bonus,
        prefers_media = profile.prefers_media,
        has_media = t.has_media,
        "D5 Media preference"
    );

    // Longueur préférée = signal fort d'achèvement de lecture
    let length_match = match (&profile.preferred_content_length, t.content_length) {
        (ContentLength::Short, l) if l < 100 => 0.15,
        (ContentLength::Medium, l) if (80..=200).contains(&l) => 0.15,
        (ContentLength::Long, l) if l > 180 => 0.15,
        _ => 0.05,
    };
    score += length_match;
    trace!(
        length_match,
        content_length = t.content_length,
        "D5 Length preference match"
    );

    // Prédiction de partage : si utilisateur retweet beaucoup
    let profile_retweet_rate = if profile.liked_tweet_ids.len() > 0 {
        profile.retweeted_tweet_ids.len() as f64 / profile.liked_tweet_ids.len() as f64
    } else {
        0.1
    };
    // Si tweet très retweetable ET utilisateur retweet souvent
    let retweet_pred = sigmoid((t.retweet_count as f64 / t.view_count.max(1) as f64) * 100.0)
        * profile_retweet_rate;
    let retweet_bonus = (retweet_pred * 0.20).min(0.20);
    score += retweet_bonus;
    trace!(
        retweet_pred,
        profile_retweet_rate,
        retweet_bonus,
        "D5 Retweet prediction"
    );

    // Tendance d'engagement en hausse → utilisateurs actifs plus réceptifs
    let trend_bonus = if profile.engagement_trend > 1.2 {
        0.10
    } else {
        0.0
    };
    score += trend_bonus;
    trace!(
        engagement_trend = profile.engagement_trend,
        trend_bonus,
        "D5 Engagement trend"
    );

    // Risque churn inverse : utilisateur fidèle → plus de chances d'engager
    let loyalty_bonus = (1.0 - profile.churn_risk) * 0.12;
    score += loyalty_bonus;
    trace!(
        churn_risk = profile.churn_risk,
        loyalty_bonus,
        "D5 Churn risk (loyalty)"
    );

    let d5 = score.clamp(0.0, 1.0);
    debug!(
        d5,
        activity_bonus,
        media_bonus,
        length_match,
        retweet_bonus,
        trend_bonus,
        loyalty_bonus,
        "D5 Final"
    );
    d5
}

// ═══════════════════════════════════════════════════════════════════════════════
// D6 — CONTENT DIVERSITY (8%)
// Maximal Marginal Relevance (MMR) adapté :
// récompense les tweets qui apportent de la nouveauté au feed.
// ═══════════════════════════════════════════════════════════════════════════════
fn d6_content_diversity(t: &RawTweet, profile: &UserProfile, feed: &[ScoredTweet]) -> f64 {
    let feed_size = feed.len();
    trace!(feed_size, tweet_id = %t.id, "D6 Content diversity analysis");

    if feed_size == 0 {
        trace!("D6 First tweet in feed → max diversity");
        return 0.80;
    }

    // Score de nouveauté : bonus si peu de tweets de cet auteur dans le feed
    // (déjà géré par diversity_multiplier, ici on ajoute la diversité de format)

    let mut score = 0.70_f64; // base
    trace!(base_score = 0.70, "D6 Base score");

    // Bonus format : médias vs texte — ratio réel de tweets-média déjà dans le feed.
    // (auparavant un placeholder figé à 1.0 qui désactivait totalement ce bonus)
    let media_in_feed = feed.iter().filter(|s| s.breakdown.has_media).count();
    let media_ratio_in_feed = media_in_feed as f64 / feed_size.max(1) as f64;
    // Récompense la nouveauté de format : un tweet média dans un feed pauvre en média
    // (ou inversement, un tweet texte dans un feed saturé de médias).
    let media_bonus = if t.has_media && media_ratio_in_feed < 0.40 {
        0.15
    } else if !t.has_media && media_ratio_in_feed > 0.70 {
        0.10
    } else {
        0.0
    };
    score += media_bonus;
    trace!(
        media_bonus,
        media_ratio_in_feed,
        media_in_feed,
        "D6 Media format diversity (real ratio)"
    );

    // Bonus hashtags non vus (nouveaux sujets)
    let hashtag_novelty = if t.hashtag_count > 0 { 0.10 } else { 0.0 };
    score += hashtag_novelty;
    trace!(
        hashtag_novelty,
        hashtag_count = t.hashtag_count,
        "D6 Hashtag novelty"
    );

    // Contenu pas encore vu
    let novelty_bonus = if !profile.has_seen(&t.id) {
        0.05
    } else {
        0.0
    };
    score += novelty_bonus;
    trace!(
        novelty_bonus,
        already_seen = profile.has_seen(&t.id),
        "D6 Content novelty"
    );

    let d6 = score.clamp(0.0, 1.0);
    debug!(d6, "D6 Final");
    d6
}

// ═══════════════════════════════════════════════════════════════════════════════
// D7 — VIRAL PREDICTION (7%)
// Prédit le potentiel viral : accélération, shareabilité, cascade.
// ═══════════════════════════════════════════════════════════════════════════════
fn d7_viral_prediction(t: &RawTweet) -> f64 {
    let age_h = age_hours(t).max(0.1);
    let total_eng = t.like_count + t.comment_count + t.retweet_count + t.share_count;
    trace!(age_h, total_eng, "D7 Viral prediction start");

    if total_eng == 0 {
        trace!("D7 No engagement yet");
        return 0.05;
    }

    // Taux de partage (retweet/share par rapport au total d'engagement)
    let share_ratio = (t.retweet_count + t.share_count) as f64 / total_eng.max(1) as f64;
    trace!(
        share_ratio,
        retweets = t.retweet_count,
        shares = t.share_count,
        "D7 Share ratio"
    );

    // Cascade effect : plus de retweets que de likes → potentiel viral fort
    let cascade = if t.like_count > 0 {
        (t.retweet_count as f64 / t.like_count as f64).min(3.0) / 3.0
    } else if t.retweet_count > 0 {
        1.0
    } else {
        0.0
    };
    trace!(
        cascade,
        likes = t.like_count,
        "D7 Cascade effect (retweets/likes)"
    );

    // Vitesse de diffusion : eng/heure (viral = > 100 eng/h)
    let eng_per_hour = total_eng as f64 / age_h;
    let spread_velocity = sigmoid(eng_per_hour / 100.0);
    trace!(eng_per_hour, spread_velocity, "D7 Spread velocity");

    // Shareabilité : tweets avec médias et peu de rapports = plus partagés
    let shareability = if t.has_media { 0.3 } else { 0.1 }
        + if t.hashtag_count > 0 { 0.1 } else { 0.0 }
        + if t.url_count > 0 { 0.1 } else { 0.0 };
    trace!(
        shareability,
        has_media = t.has_media,
        hashtags = t.hashtag_count,
        urls = t.url_count,
        "D7 Shareability"
    );

    // Prédiction virale composite
    let d7 = (share_ratio * 0.30 + cascade * 0.25 + spread_velocity * 0.25 + shareability * 0.20)
        .clamp(0.0, 1.0);
    debug!(
        d7,
        share_ratio,
        cascade,
        spread_velocity,
        shareability,
        "D7 Final (30% share + 25% cascade + 25% velocity + 20% shareability)"
    );
    d7
}

// ═══════════════════════════════════════════════════════════════════════════════
// D8 — PERSONALIZATION DEPTH (5%)
// Affinité profonde : top auteurs, mots-clés, profil émotionnel.
// ═══════════════════════════════════════════════════════════════════════════════
fn d8_personalization_depth(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0_f64;
    trace!(author_id = %t.user_id, "D8 Personalization depth start");

    // Affinité auteur (score précalculé depuis l'historique d'interactions)
    let author_affinity = profile
        .top_authors
        .iter()
        .find(|(uid, _)| uid == &t.user_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let affinity_score = author_affinity * 0.40;
    score += affinity_score;
    trace!(
        author_affinity,
        affinity_score,
        "D8 Author affinity (40% weight)"
    );

    // Correspondance mots avec centres d'intérêt
    let content_lower = t.content.to_lowercase();
    let mut word_matches = 0;
    let interest_score: f64 = profile
        .top_words
        .iter()
        .take(20)
        .map(|(word, count)| {
            if content_lower.contains(word.as_str()) {
                word_matches += 1;
                (*count as f64).ln() / 5.0 // pondéré par fréquence
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .min(0.35);
    score += interest_score;
    trace!(
        interest_score,
        word_matches,
        top_words_count = profile.top_words.len(),
        "D8 Interest word matches"
    );

    // Profil émotionnel : si utilisateur positif, boost contenus enthousiastes
    let emotional_bonus = if profile.emotional_positivity > 0.6 && t.emoji_count > 0 {
        0.10
    } else {
        0.0
    };
    score += emotional_bonus;
    trace!(
        emotional_bonus,
        emotional_positivity = profile.emotional_positivity,
        emoji_count = t.emoji_count,
        "D8 Emotional positivity"
    );

    // Activité heure = match quasi parfait
    let pub_hour = t
        .created_at
        .format("%H")
        .to_string()
        .parse::<u32>()
        .unwrap_or(12);
    let hour_bonus = if pub_hour == profile.most_active_hour {
        0.15
    } else {
        0.0
    };
    score += hour_bonus;
    trace!(
        hour_bonus,
        pub_hour,
        most_active_hour = profile.most_active_hour,
        "D8 Peak activity hour match"
    );

    let d8 = score.clamp(0.0, 1.0);
    debug!(
        d8,
        author_affinity, interest_score, emotional_bonus, hour_bonus, "D8 Final"
    );
    d8
}

// ═══════════════════════════════════════════════════════════════════════════════
// Modificateurs globaux
// ═══════════════════════════════════════════════════════════════════════════════

/// Multiplicateur de fatigue d'exposition.
///
/// 0-2 impressions → 1.0 (aucune pénalité : on laisse une seconde chance,
/// un tweet manqué au scroll n'est pas un tweet refusé).
/// Ensuite chaque exposition supplémentaire retire 35 %, avec un plancher à
/// 0.05 pour ne jamais bannir définitivement un contenu de la sélection.
pub fn impression_fatigue(impressions: i64) -> f64 {
    if impressions <= IMPRESSION_FATIGUE_START {
        return 1.0;
    }
    let excess = (impressions - IMPRESSION_FATIGUE_START) as f64;
    (0.65_f64).powf(excess).max(0.05)
}

/// Multiplicateur de visibilité lié au palier d'abonnement de l'auteur.
///
/// Ce n'est pas une dimension de qualité : c'est un avantage acheté, assumé
/// comme tel, et gardé assez petit pour ne jamais faire remonter un mauvais
/// tweet — voir les constantes.
pub fn subscription_boost(tier: AuthorTier) -> f64 {
    match tier {
        AuthorTier::Free => 1.0,
        AuthorTier::Plus => SUBSCRIPTION_BOOST_PLUS,
        AuthorTier::Pro => SUBSCRIPTION_BOOST_PRO,
    }
}

/// Anti-bulle de filtre : pénalise progressivement les auteurs sur-représentés.
pub fn diversity_multiplier(author_count_in_feed: u32) -> f64 {
    // Décroissance rendue plus franche : au-delà de trois tweets, un auteur ne
    // doit plus concurrencer sérieusement les autres. Trois est aussi le
    // plafond dur appliqué à la mise en page (`MAX_PER_AUTHOR_PER_PAGE`), et les
    // deux mécanismes se complètent plutôt que de se doubler : celui-ci décide
    // de l'ORDRE — quel tweet mérite d'être vu en premier — et le plafond décide
    // du REMPLISSAGE. Sans la pente, le plafond se contentait de repousser à la
    // page suivante des tweets qui restaient malgré tout mieux classés que du
    // contenu varié ; sans le plafond, la pente ne faisait que ralentir un
    // compte qui finissait quand même par occuper la page faute de concurrence.
    let mult = match author_count_in_feed {
        0 | 1 => 1.00,
        2 => 0.82,
        3 => 0.62,
        4 => 0.36,
        5 => 0.20,
        _ => 0.10,
    };
    trace!(
        author_count_in_feed,
        mult,
        "Diversity multiplier applied (anti-filter bubble)"
    );
    mult
}

/// Anti-répétition THÉMATIQUE — le pendant de `diversity_multiplier`, un cran
/// au niveau du sujet plutôt que du compte.
///
/// `diversity_multiplier` empêche un même AUTEUR de saturer le fil ; rien
/// n'empêchait dix comptes différents de tenir dix tweets sur le même sujet,
/// alors que le champ `theme` (annotation LLM) était déjà chargé et
/// disponible — juste jamais lu pour ça, seulement pour repérer deux
/// catégories de contenu dégradant (voir `d9_llm_understanding`).
///
/// Pente nettement plus douce que la répétition d'auteur : un thème est une
/// catégorie large (« actualité », « sport »…), pas une identité — un fil
/// d'actualité un jour de gros événement DOIT pouvoir en montrer plusieurs
/// sans que chacun soit traité comme un quasi-doublon.
pub fn theme_diversity_multiplier(theme_count_in_feed: u32) -> f64 {
    let mult = match theme_count_in_feed {
        0 | 1 | 2 | 3 => 1.00,
        4 => 0.90,
        5 => 0.80,
        6 => 0.68,
        _ => 0.55,
    };
    trace!(
        theme_count_in_feed,
        mult,
        "Theme diversity multiplier applied"
    );
    mult
}

/// Pénalité pour contenu rapporté ou signalé.
fn moderation_penalty(t: &RawTweet) -> f64 {
    if t.report_count == 0 {
        return 0.0;
    }
    let report_ratio = t.report_count as f64 / t.view_count.max(100) as f64;
    let penalty = -(report_ratio * 3.0).min(0.50);
    trace!(
        report_count = t.report_count,
        report_ratio,
        penalty,
        "Moderation penalty calculated"
    );
    penalty // max -50% du score
}

// ═══════════════════════════════════════════════════════════════════════════════
// Métriques globales du feed
// ═══════════════════════════════════════════════════════════════════════════════

/// Calcule des métriques de qualité sur l'ensemble du feed final.
pub struct FeedQualityMetrics {
    pub diversity_score: f64,
    pub freshness_score: f64,
    pub relevance_score: f64,
    pub viral_potential: f64,
    pub novelty_score: f64,
}

pub fn compute_feed_metrics(tweets: &[(&RawTweet, &ScoredTweet)]) -> FeedQualityMetrics {
    if tweets.is_empty() {
        return FeedQualityMetrics {
            diversity_score: 0.0,
            freshness_score: 0.0,
            relevance_score: 0.0,
            viral_potential: 0.0,
            novelty_score: 0.0,
        };
    }

    let n = tweets.len() as f64;

    // Diversité : ratio d'auteurs uniques
    let unique_authors: std::collections::HashSet<&str> =
        tweets.iter().map(|(t, _)| t.user_id.as_str()).collect();
    let diversity_score = (unique_authors.len() as f64 / n).min(1.0);

    // Fraîcheur : moyenne des scores de fraîcheur
    let freshness_score = tweets
        .iter()
        .map(|(t, _)| (-0.115 * age_hours(t)).exp())
        .sum::<f64>()
        / n;

    // Pertinence : moyenne des scores D5 (behavioral prediction)
    let relevance_score = tweets
        .iter()
        .map(|(_, s)| s.breakdown.behavioral_prediction)
        .sum::<f64>()
        / n;

    // Potentiel viral : moyenne D7
    let viral_potential = tweets
        .iter()
        .map(|(_, s)| s.breakdown.viral_prediction)
        .sum::<f64>()
        / n;

    // Nouveauté : % de tweets avec scores de découverte (source != social)
    let novelty_score = tweets
        .iter()
        .map(|(t, _)| {
            if t.source == TweetSource::Discovery || t.source == TweetSource::Trending {
                1.0
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n;

    FeedQualityMetrics {
        diversity_score,
        freshness_score: freshness_score.clamp(0.0, 1.0),
        relevance_score: relevance_score.clamp(0.0, 1.0),
        viral_potential: viral_potential.clamp(0.0, 1.0),
        novelty_score: novelty_score.clamp(0.0, 1.0),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilitaires mathématiques
// ═══════════════════════════════════════════════════════════════════════════════

pub fn age_hours(t: &RawTweet) -> f64 {
    let diff = Utc::now().signed_duration_since(t.created_at);
    (diff.num_seconds() as f64 / 3600.0).max(0.0)
}

/// Sigmoid : σ(x) = 1 / (1 + e^-x)  →  σ(0) = 0.5
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Gaussienne normalisée : pic à 1.0 au centre μ, largeur σ
fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abonnes_devant_gratuits_a_qualite_egale() {
        assert!(subscription_boost(AuthorTier::Pro) > subscription_boost(AuthorTier::Plus));
        assert!(subscription_boost(AuthorTier::Plus) > subscription_boost(AuthorTier::Free));
        assert_eq!(subscription_boost(AuthorTier::Free), 1.0);
    }

    /// Le boost doit rester un départage. Un tweet gratuit nettement meilleur
    /// (ici +15 % de score brut) doit continuer de passer devant un tweet Pro :
    /// si ce test casse, c'est que l'avantage acheté est devenu un classement
    /// parallèle.
    #[test]
    fn un_meilleur_tweet_gratuit_reste_devant_un_tweet_pro() {
        let gratuit = 0.50 * subscription_boost(AuthorTier::Free);
        let pro = 0.435 * subscription_boost(AuthorTier::Pro);
        assert!(gratuit > pro, "gratuit={gratuit} pro={pro}");
    }

    #[test]
    fn palier_resolu_depuis_les_deux_colonnes() {
        assert_eq!(AuthorTier::resolve("pro", false), AuthorTier::Pro);
        assert_eq!(AuthorTier::resolve("plus", false), AuthorTier::Plus);
        // Compte historique : `premium` seul, sans palier explicite.
        assert_eq!(AuthorTier::resolve("free", true), AuthorTier::Pro);
        assert_eq!(AuthorTier::resolve("free", false), AuthorTier::Free);
    }
}
