use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::shadowban::ShadowbanLevel;

// ─── Requêtes entrantes ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RecommendRequest {
    pub user_id: String, // UUID
    pub limit: Option<i32>,
    pub offset: Option<i32>,
    pub mode: Option<RecommendMode>,
    pub exclude_seen: Option<bool>,
    pub force_refresh: Option<bool>,
    /// Déploiement progressif : seul le client Windows le demande pour le moment.
    pub enable_experiments: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RecommendMode {
    Feed,
    Discover,
    Trending,
    ForYou,
}

impl Default for RecommendMode {
    fn default() -> Self {
        RecommendMode::ForYou
    }
}

#[derive(Debug, Deserialize)]
pub struct TrackInteractionRequest {
    pub user_id: String,  // UUID
    pub tweet_id: String, // UUID
    pub interaction_type: InteractionType,
    pub dwell_ms: Option<u32>,
    /// Nature du contenu regardé, pour interpréter `dwell_ms`.
    ///
    /// Un temps brut est confondu avec la LONGUEUR du contenu — voir
    /// `algorithm::dwell`. Ces trois champs permettent de le rapporter au temps
    /// que ce contenu-là demandait. Tous facultatifs : un client qui ne les
    /// envoie pas retombe sur l'ancien calcul par paliers bruts.
    pub dwell_media: Option<crate::algorithm::dwell::DwellMedia>,
    pub content_chars: Option<u32>,
    pub video_duration_ms: Option<u32>,
    /// Auteur du tweet — nécessaire pour qu'un « ça ne m'intéresse pas » porte
    /// sur le compte et pas seulement sur le tweet refusé (voir `track_handler`).
    pub author_id: Option<String>,
    /// Indices renvoyés avec le tweet. Facultatifs pour rester compatible avec
    /// les anciens clients ; le moteur retombe alors sur l'affectation stockée.
    pub experiment_id: Option<String>,
    pub variant_id: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionType {
    Like,
    Unlike,
    Comment,
    Retweet,
    Unretweet,
    Share,
    View,
    Bookmark,
    ProfileView,
    Skip,
    Report,
    Block,
    /// Réponse explicite à la question posée dans le fil (« ça t'intéresse ? »).
    ///
    /// Distinct d'un like : on ne déclare pas aimer le tweet, on déclare vouloir
    /// — ou ne plus vouloir — ce GENRE de contenu. `NotInterested` pèse donc
    /// nettement plus lourd qu'un `Skip` constaté au chronomètre, et déclenche
    /// en plus une mise en sourdine de l'auteur (voir `track_handler`).
    Interested,
    NotInterested,
}

impl InteractionType {
    pub fn weight(self) -> f64 {
        match self {
            InteractionType::Like => 1.0,
            InteractionType::Unlike => -1.0,
            InteractionType::Comment => 3.5,
            InteractionType::Retweet => 5.0,
            InteractionType::Unretweet => -2.0,
            InteractionType::Share => 4.0,
            InteractionType::Bookmark => 2.5,
            InteractionType::View => 0.2,
            InteractionType::ProfileView => 1.5,
            InteractionType::Skip => -0.5,
            InteractionType::Report => -12.0,
            InteractionType::Block => -20.0,
            // Déclaré à la main, en réponse à une question directe : ça vaut
            // plus qu'un geste deviné, moins qu'un signalement (qui vise le
            // contenu lui-même, pas le goût du lecteur).
            InteractionType::Interested => 3.0,
            InteractionType::NotInterested => -8.0,
        }
    }

    /// Label d'entraînement du modèle CTR pour cette interaction.
    ///
    /// `Some(true)`  → engagement positif avéré, exemple positif.
    /// `Some(false)` → rejet explicite, exemple négatif.
    /// `None`        → l'interaction ne tranche pas. Une `View` est
    ///   l'impression elle-même : elle ouvre la fenêtre d'attribution au lieu
    ///   de conclure. Si rien ne suit, le balayage la comptera en négatif.
    ///
    /// Piège corrigé : le label se déduisait de `weight() > 0.0`, or une `View`
    /// pèse 0.2 — toute impression était donc étiquetée « cliquée » et le
    /// modèle ne pouvait apprendre qu'une seule chose, « tout est un clic ».
    pub fn ctr_label(self) -> Option<bool> {
        match self {
            InteractionType::Like
            | InteractionType::Comment
            | InteractionType::Retweet
            | InteractionType::Share
            | InteractionType::Bookmark
            | InteractionType::Interested
            | InteractionType::ProfileView => Some(true),

            InteractionType::Skip
            | InteractionType::Report
            | InteractionType::Block
            | InteractionType::Unlike
            | InteractionType::NotInterested
            | InteractionType::Unretweet => Some(false),

            InteractionType::View => None,
        }
    }
}

// ─── Profil utilisateur ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub follower_count: i64,
    pub following_count: i64,
    pub network_influence: f64,

    pub following_ids: Vec<String>,
    pub mutual_follow_ids: Vec<String>,
    pub second_degree_ids: Vec<String>,

    /// Comptes liés à ce lecteur par un blocage `user_follows.status =
    /// 'blocked'`, dans n'importe quel sens. Union avec le hard-ban admin
    /// avant `collect_candidates` — un compte bloqué ne doit pas plus
    /// apparaître dans le vivier qu'un compte hard-banni.
    #[serde(default)]
    pub blocked_ids: Vec<String>,

    pub liked_tweet_ids: Vec<String>,
    pub retweeted_tweet_ids: Vec<String>,
    pub replied_to_tweet_ids: Vec<String>,
    pub bookmarked_tweet_ids: Vec<String>,

    pub top_authors: Vec<(String, f64)>,

    pub hourly_activity: [f64; 24],
    pub daily_activity: [f64; 7],
    pub most_active_hour: u32,
    pub most_active_day: u32,

    pub avg_content_length: f64,
    pub prefers_media: bool,
    pub avg_hashtag_count: f64,
    pub preferred_content_length: ContentLength,

    pub engagement_velocity: f64,
    pub engagement_trend: f64,
    pub personality_type: PersonalityType,
    pub emotional_positivity: f64,

    pub top_words: Vec<(String, u32)>,

    pub churn_risk: f64,
    pub lifetime_value: f64,

    pub seen_tweet_ids: Vec<String>,

    /// Auteurs que CE lecteur a explicitement refusés → nombre de refus.
    ///
    /// Vient du set Redis `twitninf:damped:<user>`, alimenté par les réponses
    /// « ça ne m'intéresse pas ». Porté par le profil (et non relu à chaque
    /// tweet) pour rester à un aller Redis par recommandation.
    #[serde(default)]
    pub damped_authors: std::collections::HashMap<String, f64>,

    /// Moyenne des embeddings des tweets likés récents — voir
    /// `crate::embeddings`. `None` pour un compte neuf, ou tant que le
    /// rattrapage n'a pas encore embedded ses tweets aimés.
    /// `#[serde(default)]` : les profils déjà en cache avant l'ajout de ce
    /// champ n'en portent pas, la désérialisation ne doit pas échouer dessus.
    #[serde(default)]
    pub taste_vector: Option<Vec<f32>>,

    pub user_type: UserType,
    pub profile_confidence: f64,

    // ── Index d'appartenance ─────────────────────────────────────────────
    //
    // Les mêmes ensembles que `following_ids`, `mutual_follow_ids`,
    // `second_degree_ids` et `seen_tweet_ids`, mais en `HashSet`.
    //
    // Ce ne sont pas des données de plus : ce sont les MÊMES, indexées. Le
    // scoring pose la question « ce lecteur suit-il cet auteur ? » une bonne
    // dizaine de fois par tweet candidat (D3 trois fois, le filtre de surface,
    // le boost d'abonnement deux fois, le bandit une fois), et chacune de ces
    // questions balayait un `Vec<String>` qui monte à 1000 abonnements. À
    // 1700 candidats, ça fait plusieurs millions de comparaisons de chaînes
    // par recommandation, pour une information qu'on peut hacher une seule
    // fois au moment où le profil est construit.
    //
    // `#[serde(skip)]` : jamais sérialisés vers Redis — les recalculer coûte
    // moins cher que de les transporter, et un profil relu du cache passe de
    // toute façon par `rebuild_indexes()`.
    #[serde(skip)]
    pub following_set: std::collections::HashSet<String>,
    #[serde(skip)]
    pub mutual_set: std::collections::HashSet<String>,
    #[serde(skip)]
    pub second_degree_set: std::collections::HashSet<String>,
    #[serde(skip)]
    pub seen_set: std::collections::HashSet<String>,
}

impl UserProfile {
    /// (Re)construit les index d'appartenance depuis les vecteurs.
    ///
    /// À appeler à CHAQUE endroit où un profil devient utilisable : à la
    /// sortie de la base, et à la sortie du cache Redis (où `serde(skip)` les
    /// a laissés vides). Un index vide alors que le vecteur ne l'est pas
    /// donnerait des réponses fausses — c'est le seul risque de ce montage,
    /// et il est cantonné à ces deux points.
    pub fn rebuild_indexes(&mut self) {
        self.following_set = self.following_ids.iter().cloned().collect();
        self.mutual_set = self.mutual_follow_ids.iter().cloned().collect();
        self.second_degree_set = self.second_degree_ids.iter().cloned().collect();
        self.seen_set = self.seen_tweet_ids.iter().cloned().collect();
    }

    /// Appartenance, en préférant l'index quand il existe.
    ///
    /// Le repli sur le vecteur n'est pas une coquetterie défensive : un index
    /// vide face à un vecteur plein est INDISCERNABLE d'un « ce lecteur ne
    /// suit personne ». Sans repli, oublier un `rebuild_indexes()` sur un
    /// chemin (celui du cache, une désérialisation ajoutée plus tard, un test)
    /// ne produirait ni erreur ni log — juste un fil où le boost d'abonnement
    /// et D3 valent zéro pour tout le monde. On paie une comparaison
    /// d'entier pour rendre cette panne impossible ; le chemin nominal, lui,
    /// reste bien en O(1).
    #[inline]
    fn member(set: &std::collections::HashSet<String>, list: &[String], needle: &str) -> bool {
        if set.is_empty() && !list.is_empty() {
            return list.iter().any(|id| id == needle);
        }
        set.contains(needle)
    }

    /// Ce lecteur suit-il ce compte ?
    #[inline]
    pub fn follows(&self, author_id: &str) -> bool {
        Self::member(&self.following_set, &self.following_ids, author_id)
    }

    #[inline]
    pub fn is_mutual(&self, author_id: &str) -> bool {
        Self::member(&self.mutual_set, &self.mutual_follow_ids, author_id)
    }

    #[inline]
    pub fn is_second_degree(&self, author_id: &str) -> bool {
        Self::member(&self.second_degree_set, &self.second_degree_ids, author_id)
    }

    #[inline]
    pub fn has_seen(&self, tweet_id: &str) -> bool {
        Self::member(&self.seen_set, &self.seen_tweet_ids, tweet_id)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum ContentLength {
    Short,
    #[default]
    Medium,
    Long,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PersonalityType {
    Enthusiastic,
    Curious,
    Thoughtful,
    #[default]
    Balanced,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum UserType {
    PowerUser,
    Regular,
    #[default]
    Casual,
}

// ─── Palier d'abonnement de l'auteur ─────────────────────────────────────────

/// Palier payant du compte qui publie.
///
/// Deux colonnes décrivent la même chose en base : `subscription_tier` (l'ENUM
/// courant) et `premium` (l'ancien booléen, encore vrai sur des comptes
/// historiques qui n'ont jamais eu de palier). `resolve` applique la même règle
/// que l'API (`customizationTier` dans `userRoutes.js`) : le palier explicite
/// gagne, `premium` seul vaut Pro.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthorTier {
    #[default]
    Free,
    Plus,
    Pro,
}

impl AuthorTier {
    pub fn resolve(tier: &str, legacy_premium: bool) -> Self {
        match tier {
            "pro" => Self::Pro,
            "plus" => Self::Plus,
            _ if legacy_premium => Self::Pro,
            _ => Self::Free,
        }
    }
}

// ─── Tweet candidat brut ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RawTweet {
    pub id: String,
    pub user_id: String,
    pub content: String,
    pub created_at: DateTime<Utc>,

    pub view_count: i64,
    pub like_count: i64,
    pub comment_count: i64,
    pub retweet_count: i64,
    pub share_count: i64,
    pub bookmark_count: i64,
    pub report_count: i64,

    pub likes_1h: i64,
    pub likes_6h: i64,
    pub comments_1h: i64,
    pub retweets_1h: i64,

    pub has_media: bool,
    pub hashtag_count: i32,
    pub mention_count: i32,
    pub content_length: i32,
    pub emoji_count: i32,
    pub exclamation_count: i32,
    pub question_count: i32,
    pub url_count: i32,
    pub words: Vec<String>,

    pub author_followers: i64,
    pub author_following: i64,
    pub author_is_verified: bool,
    pub author_is_premium: bool,
    /// Palier d'abonnement de l'auteur (`users.subscription_tier`). Distinct de
    /// `author_is_premium`, qui est l'ancien drapeau booléen : les deux
    /// coexistent en base, voir `AuthorTier::resolve`.
    pub author_tier: AuthorTier,
    pub author_account_age_days: i32,
    pub author_tweet_count: i64,

    pub moderation_status: String,
    pub recommendation_group: Option<String>,

    /// Tweet auquel celui-ci répond.
    ///
    /// C'est le seul champ fiable pour savoir si un tweet est une réponse :
    /// en base, 98 tweets typés `'tweet'` ont un `parent_tweet_id` renseigné.
    /// Ne pas se fier à `tweet_type`.
    pub parent_tweet_id: Option<String>,
    /// Tweet d'origine quand celui-ci est un retweet ou une citation.
    ///
    /// Attention : les réponses le renseignent aussi (elles pointent la racine
    /// du fil). Ne l'utiliser comme identité de contenu que si `is_retweet`.
    pub original_tweet_id: Option<String>,
    /// Vrai retweet (réaffiche l'original sans texte propre).
    pub is_retweet: bool,

    /// Nombre de fois que CE lecteur a déjà vu ce tweet passer dans son fil
    /// (`user_behavior_data.tweet_view`). Une exposition répétée sans
    /// engagement est un signal négatif fort : c'est un refus implicite.
    pub viewer_impressions: i64,

    /// `users.algorithmic_visibility_multiplier` — levier de visibilité par
    /// compte déjà présent en base, mais qu'aucun scoring ne lisait.
    pub author_visibility_multiplier: f64,

    pub source: TweetSource,
    pub source_weight: f64,

    pub author_shadowban_level: ShadowbanLevel,

    /// Étiquettes produites par l'annotateur LLM (`tweet_llm_labels`).
    /// `None` = tweet pas encore annoté : D9 reste alors neutre au lieu de
    /// pénaliser un contenu dont on ne sait simplement rien.
    pub llm: Option<LlmLabels>,
}

/// Compréhension du contenu produite hors-ligne par le LLM annotateur.
///
/// Ces signaux n'existent nulle part ailleurs dans la base : le corpus est trop
/// petit pour qu'un modèle les apprenne depuis l'engagement (203 paires
/// impression→like au moment de la conception). Le LLM apporte la connaissance,
/// la base ne fournit que les items à annoter.
#[derive(Debug, Clone)]
pub struct LlmLabels {
    pub theme: String,
    /// [0,1] — agression envers une personne ou un groupe.
    pub toxicity_score: f64,
    pub toxicity_category: String,
    /// [0,1] — apport réel du message (informe, fait rire, lance un échange).
    pub quality_score: f64,
    pub tone: String,
    /// [0,1] — confiance de l'annotation, utilisée pour amortir son effet.
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum TweetSource {
    Trending,
    SocialGraph,
    Personalized,
    Viral,
    Temporal,
    Discovery,
    #[default]
    Influencer,
    Quality,
}

// `RawTweet` est construit dans les tests via `..Default::default()`, mais le
// dérive était impossible : `DateTime<Utc>` n'implémente pas `Default`. Le test
// de D1 ne compilait donc pas. Impl manuelle, avec l'epoch comme date neutre.
impl Default for RawTweet {
    fn default() -> Self {
        Self {
            id: String::new(),
            user_id: String::new(),
            content: String::new(),
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_else(Utc::now),
            view_count: 0,
            like_count: 0,
            comment_count: 0,
            retweet_count: 0,
            share_count: 0,
            bookmark_count: 0,
            report_count: 0,
            likes_1h: 0,
            likes_6h: 0,
            comments_1h: 0,
            retweets_1h: 0,
            has_media: false,
            hashtag_count: 0,
            mention_count: 0,
            content_length: 0,
            emoji_count: 0,
            exclamation_count: 0,
            question_count: 0,
            url_count: 0,
            words: Vec::new(),
            author_followers: 0,
            author_following: 0,
            author_is_verified: false,
            author_is_premium: false,
            author_tier: AuthorTier::Free,
            author_account_age_days: 0,
            author_tweet_count: 0,
            moderation_status: "approved".to_string(),
            recommendation_group: None,
            parent_tweet_id: None,
            original_tweet_id: None,
            is_retweet: false,
            viewer_impressions: 0,
            author_visibility_multiplier: 1.0,
            source: TweetSource::default(),
            source_weight: 1.0,
            author_shadowban_level: ShadowbanLevel::default(),
            llm: None,
        }
    }
}

// ─── Score ───────────────────────────────────────────────────────────────────

// `Default` est requis par les helpers de test du bandit contextuel, qui
// construisaient un breakdown vide via `Default::default()` sans que le dérive
// existe — le crate ne compilait donc pas en mode test.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ScoreBreakdown {
    pub engagement_velocity: f64,
    pub content_intelligence: f64,
    pub social_graph_dynamics: f64,
    pub temporal_dynamics: f64,
    pub behavioral_prediction: f64,
    pub content_diversity: f64,
    pub viral_prediction: f64,
    pub personalization_depth: f64,

    pub engagement_velocity_raw: f64,
    pub engagement_acceleration: f64,
    pub viral_velocity: f64,

    pub direct_follow_boost: f64,
    pub mutual_follow_boost: f64,
    pub second_degree_boost: f64,

    pub diversity_multiplier: f64,
    pub moderation_penalty: f64,
    pub source_weight: f64,
    pub shadowban_multiplier: f64,
    pub garbage_penalty: f64,
    /// Mod I — coup de pouce d'abonné (1.0 pour un compte gratuit).
    pub subscription_boost: f64,

    /// Métadonnée portée pour le calcul de diversité de format du feed (D6).
    pub has_media: bool,

    pub final_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredTweet {
    pub tweet_id: String,
    pub score: f64,
    pub breakdown: ScoreBreakdown,
    /// Vecteur de features réellement utilisé pour classer ce tweet. Mémorisé
    /// à l'affichage puis rejoué à l'interaction : c'est la seule façon
    /// d'entraîner le modèle CTR sur ce qu'il a effectivement vu.
    /// Interne au scoring, jamais exposé dans la réponse API.
    #[serde(skip)]
    pub ctr_features: Option<Vec<f64>>,
}

// ─── Réponses API ─────────────────────────────────────────────────────────────

/// Une place dans le fil : un tweet, et le tweet auquel il répond quand ce
/// dernier est servi JUSTE au-dessus de lui.
///
/// C'est la forme mise en cache, et non plus une simple liste d'identifiants :
/// le lien de fil est calculé au moment de la mise en forme, à l'unique endroit
/// où l'on connaît encore les tweets complets. Le recalculer à la lecture du
/// cache aurait exigé de recharger les parents depuis la base à chaque page
/// servie — pour retrouver une information qu'on avait déjà.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

/// Lien de conversation exposé au client : « ce tweet répond à celui-là, et le
/// fil part de cette racine ».
///
/// Redondant avec l'ordre de `tweet_ids` — l'adjacence dit déjà la même chose —
/// et c'est voulu : l'ordre est une convention fragile qu'un intermédiaire peut
/// casser sans s'en rendre compte, ce champ est une affirmation explicite. Un
/// client qui reçoit les deux peut vérifier qu'ils concordent.
#[derive(Debug, Clone, Serialize)]
pub struct ThreadLink {
    pub tweet_id: String,
    pub parent_id: String,
    pub root_id: String,
    /// 1 pour une réponse au tweet racine, 2 pour une réponse à cette réponse…
    pub depth: usize,
}

#[derive(Debug, Serialize)]
pub struct RecommendResponse {
    pub success: bool,
    pub user_id: String,
    pub tweet_ids: Vec<String>,
    /// Liens de fil entre les `tweet_ids` de CETTE page. Vide quand la page ne
    /// contient aucune réponse.
    pub threads: Vec<ThreadLink>,
    pub count: usize,
    pub algorithm: &'static str,
    pub algorithm_version: &'static str,
    pub mode: String,
    pub latency_ms: u64,
    pub cache_hit: bool,
    pub experiments: Vec<crate::experiments::ExperimentAssignment>,
    /// Publicités ciblées à insérer dans CETTE page, avec leur position.
    ///
    /// Vide dans l'écrasante majorité des cas (aucune campagne active, aucune
    /// qui corresponde, plafond de fréquence atteint) — c'est un résultat
    /// normal, pas une panne. L'API hydrate ces identifiants et facture les
    /// impressions ; le moteur ne décide que du QUI voit QUOI.
    #[serde(default)]
    pub ads: Vec<crate::ads::AdPlacement>,
    pub metadata: RecommendMetadata,
}

#[derive(Debug, Serialize)]
pub struct RecommendMetadata {
    pub candidates_collected: usize,
    pub sources: SourceStats,
    pub user_profile: UserProfileSummary,
    pub quality_metrics: QualityMetrics,
    pub pagination: Pagination,
}

#[derive(Debug, Serialize, Default)]
pub struct SourceStats {
    pub trending: usize,
    pub social_graph: usize,
    pub personalized: usize,
    pub viral: usize,
    pub temporal: usize,
    pub discovery: usize,
    pub influencer: usize,
    pub quality: usize,
    pub deduplicated_total: usize,
}

#[derive(Debug, Serialize)]
pub struct UserProfileSummary {
    pub user_type: String,
    pub confidence: f64,
    pub personality: String,
    pub engagement_velocity: f64,
    pub engagement_trend: String,
    pub network_influence: f64,
    pub most_active_hour: u32,
    pub churn_risk: f64,
}

#[derive(Debug, Serialize)]
pub struct QualityMetrics {
    pub diversity_score: f64,
    pub freshness_score: f64,
    pub relevance_score: f64,
    pub viral_potential: f64,
    pub novelty_score: f64,
}

#[derive(Debug, Serialize)]
pub struct Pagination {
    pub limit: i32,
    pub offset: i32,
    pub has_more: bool,
    pub total_available: i64,
}

#[derive(Debug, Serialize)]
pub struct TrackResponse {
    pub success: bool,
    pub tweet_id: String,
    pub user_id: String,
    pub weight_applied: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: &'static str,
    pub db: String,
    pub redis: String,
    pub uptime_secs: u64,
    pub algorithm: &'static str,
}

#[cfg(test)]
mod profile_index_tests {
    use super::*;

    fn profile() -> UserProfile {
        let mut p = UserProfile {
            following_ids: vec!["a".into(), "b".into()],
            mutual_follow_ids: vec!["b".into()],
            second_degree_ids: vec!["c".into()],
            seen_tweet_ids: vec!["t1".into()],
            ..Default::default()
        };
        p.rebuild_indexes();
        p
    }

    #[test]
    fn les_index_repondent_comme_les_vecteurs() {
        let p = profile();
        assert!(p.follows("a") && p.follows("b") && !p.follows("z"));
        assert!(p.is_mutual("b") && !p.is_mutual("a"));
        assert!(p.is_second_degree("c") && !p.is_second_degree("a"));
        assert!(p.has_seen("t1") && !p.has_seen("t2"));
    }

    /// Le seul vrai risque de l'indexation : les ensembles ne sont pas
    /// sérialisés, donc un profil relu du cache Redis arrive avec des index
    /// VIDES. Sans reconstruction, `follows()` répondrait « non » pour tout le
    /// monde et le boost d'abonnement disparaîtrait silencieusement — sans
    /// erreur, sans log, pour tous les lecteurs dont le profil est en cache.
    #[test]
    fn un_profil_relu_du_cache_doit_etre_reindexe() {
        let json = serde_json::to_string(&profile()).unwrap();
        let mut relu: UserProfile = serde_json::from_str(&json).unwrap();

        // État tel qu'il sort du cache : les vecteurs sont là, pas les index.
        assert_eq!(relu.following_ids.len(), 2);
        assert!(relu.following_set.is_empty(), "l'index ne doit pas être sérialisé");

        // Le repli doit déjà donner la BONNE réponse, index absent — c'est ce
        // qui rend un `rebuild_indexes()` oublié inoffensif plutôt que
        // silencieusement destructeur.
        assert!(relu.follows("a") && relu.is_mutual("b") && relu.has_seen("t1"));
        assert!(!relu.follows("z"));

        relu.rebuild_indexes();
        assert!(!relu.following_set.is_empty());
        assert!(relu.follows("a") && relu.is_mutual("b") && relu.has_seen("t1"));
        assert!(!relu.follows("z"));
    }
}
