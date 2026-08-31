use std::sync::Arc;

use aho_corasick::AhoCorasick;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::content::{analyze_content, ContentFeatures};
use crate::shadowban::ShadowbanLevel;
use crate::utils::{FxHashMap, FxHashSet};

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
    /// Ouverture du tweet en pleine page.
    ///
    /// C'est le clic le PLUS net qu'un fil produise — le lecteur quitte le
    /// défilement pour aller lire — et il n'existait pas. L'app émettait bien
    /// `profile_view` quand on tape l'AUTEUR, mais rien quand on tape le tweet
    /// lui-même : le geste le plus informatif de l'écran ne remontait nulle
    /// part.
    Open,
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
            // Entre la visite de profil (1,5) et le commentaire (3,5) :
            // ouvrir demande un geste délibéré et un changement d'écran, mais
            // ne produit rien de public.
            InteractionType::Open => 2.0,
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

    /// Poids de mise en sourdine de l'AUTEUR déclenché par ce geste.
    ///
    /// `None` = ce geste ne dit rien du compte, seulement du tweet.
    ///
    /// Tous les refus ne se valent pas, et c'est exactement ce que le moteur
    /// ignorait : « ça ne m'intéresse pas » était le SEUL geste à porter sur
    /// l'auteur, alors que c'est le plus faible des trois. Un signalement dit
    /// que le contenu n'aurait pas dû être montré ; un blocage dit qu'on ne
    /// veut plus rien voir de ce compte.
    ///
    /// L'échelle est celle de `author_damping` (0,32^n) : 1 point divise la
    /// visibilité par ~3, 2 points par ~10, 5 points la posent au plancher.
    ///
    /// `Skip` reste volontairement à `None` : ignorer UN tweet ne dit rien de
    /// son auteur.
    pub fn refusal_strikes(self) -> Option<f64> {
        match self {
            InteractionType::NotInterested => Some(1.0),
            InteractionType::Report => Some(2.0),
            InteractionType::Block => Some(5.0),
            _ => None,
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
            | InteractionType::Open
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
    pub following_set: FxHashSet<String>,
    #[serde(skip)]
    pub mutual_set: FxHashSet<String>,
    #[serde(skip)]
    pub second_degree_set: FxHashSet<String>,
    #[serde(skip)]
    pub seen_set: FxHashSet<String>,

    /// `top_authors` indexé.
    ///
    /// D3 (« ce lecteur a-t-il déjà interagi avec l'auteur ? ») et D8 (« à quel
    /// point ? ») posent la MÊME question, et la résolvaient chacune par un
    /// balayage linéaire de la liste. Deux balayages par candidat, 3400 par
    /// recommandation, pour une valeur qu'on peut hacher une seule fois.
    #[serde(skip)]
    pub author_affinity_index: FxHashMap<String, f64>,

    /// Recherche simultanée des centres d'intérêt dans le texte d'un candidat.
    ///
    /// D2 et D8 posent là aussi la même question — « lesquels des mots-clés de
    /// ce lecteur apparaissent dans ce tweet ? » — et y répondaient chacune par
    /// une boucle de `str::contains` : une cinquantaine de recherches de
    /// sous-chaîne PAR CANDIDAT, 85 000 par recommandation, pour une réponse
    /// qui tient en un seul passage.
    ///
    /// L'automate donne EXACTEMENT le même verdict que `contains` (« ce motif
    /// est-il une sous-chaîne ? ») : ni frontière de mot, ni casse, ni ordre
    /// n'entrent en jeu. Le classement ne bouge pas, et c'est vérifié par
    /// `l_automate_et_le_balayage_lineaire_disent_la_meme_chose`.
    ///
    /// `Arc` parce que `UserProfile` est cloné et que l'automate, lui, ne doit
    /// pas l'être. `None` quand il n'y a pas de mots-clés, ou qu'il y en a plus
    /// que le masque de bits n'en porte : le repli linéaire prend le relais.
    #[serde(skip)]
    pub word_matcher: Option<Arc<AhoCorasick>>,
}

/// Nombre maximum de centres d'intérêt que l'automate peut suivre.
///
/// Le passage de l'automate rend les motifs trouvés dans l'ordre du TEXTE ; D8
/// les somme, lui, dans l'ordre des MOTS-CLÉS. Pour que la somme flottante soit
/// au bit près celle du balayage linéaire, on mémorise les motifs vus dans un
/// masque de bits, puis on parcourt les mots-clés dans LEUR ordre. D'où la
/// borne : un `u64` porte 64 motifs. `top_words` en contient 30 au plus (voir
/// `extract_content_prefs`), elle n'est donc jamais atteinte en production —
/// elle existe pour que le cas contraire retombe sur le chemin linéaire au lieu
/// de perdre des mots-clés en silence.
pub const MAX_INDEXED_KEYWORDS: usize = 64;

/// Nombre de centres d'intérêt que D8 pondère (les plus fréquents d'abord).
pub const D8_KEYWORD_DEPTH: usize = 20;

/// Ce que D2 et D8 tirent des centres d'intérêt du lecteur, en UN passage.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KeywordHits {
    /// Combien de mots-clés du lecteur apparaissent dans ce tweet — ce que D2
    /// compte.
    pub matches: usize,
    /// Somme pondérée sur les `D8_KEYWORD_DEPTH` premiers mots-clés, AVANT
    /// plafonnement — ce que D8 additionne.
    pub weighted: f64,
    /// Combien de ces `D8_KEYWORD_DEPTH` premiers ont été trouvés (trace D8).
    pub top_matches: usize,
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

        // `or_insert` et non `insert` : le balayage linéaire qu'on remplace
        // s'arrêtait au PREMIER auteur trouvé. Une collecte naïve garderait le
        // dernier, et deux lignes pour le même auteur — impossible avec le
        // GROUP BY qui alimente ce champ, mais rien ne l'interdit au type —
        // donneraient alors une affinité différente d'avant.
        self.author_affinity_index =
            FxHashMap::with_capacity_and_hasher(self.top_authors.len(), Default::default());
        for (author_id, affinity) in &self.top_authors {
            self.author_affinity_index
                .entry(author_id.clone())
                .or_insert(*affinity);
        }

        // Un motif vide serait trouvé dans n'importe quel texte : `contains("")`
        // vaut `true`. Les mots-clés viennent de mots de plus de trois lettres,
        // mais le champ est public et un profil forgé à la main peut en porter.
        // Dans ce cas on ne construit pas d'automate plutôt que d'en construire
        // un qui mentirait sur QUEL mot a été trouvé.
        let has_empty = self.top_words.iter().any(|(word, _)| word.is_empty());
        self.word_matcher = if self.top_words.is_empty()
            || has_empty
            || self.top_words.len() > MAX_INDEXED_KEYWORDS
        {
            None
        } else {
            // Le constructeur par defaut, et non un DFA force : mesure faite,
            // le DFA n est pas plus rapide ici (0,297 ms contre 0,270 sur
            // 1700 candidats) parce qu il perd le PREFILTRE. Sur des mots-cles
            // reels, dont les premieres lettres varient et qui sont absents de
            // la plupart des tweets, ce prefiltre repond  non  en un balayage
            // vectorise sans jamais entrer dans l automate.
            AhoCorasick::new(self.top_words.iter().map(|(word, _)| word.as_str()))
                .ok()
                .map(Arc::new)
        };
    }

    /// Affinité déjà mesurée entre ce lecteur et cet auteur, ou 0.
    ///
    /// Même repli que `member` : un index vide face à une liste pleine est
    /// indiscernable d'un « aucune affinité », donc on rebalaie plutôt que de
    /// rendre un zéro faux.
    #[inline]
    pub fn author_affinity(&self, author_id: &str) -> f64 {
        if self.author_affinity_index.is_empty() && !self.top_authors.is_empty() {
            return self
                .top_authors
                .iter()
                .find(|(uid, _)| uid == author_id)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
        }
        self.author_affinity_index
            .get(author_id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Ce lecteur a-t-il déjà une histoire avec cet auteur ?
    ///
    /// Distinct de `author_affinity(id) > 0.0` : un auteur peut figurer dans
    /// `top_authors` avec une affinité nulle, et le bandit a besoin de la
    /// différence — « jamais vu » est ce qui rend un tweet candidat à
    /// l'exploration, pas « vu et laissé indifférent ».
    #[inline]
    pub fn knows_author(&self, author_id: &str) -> bool {
        if self.author_affinity_index.is_empty() && !self.top_authors.is_empty() {
            return self.top_authors.iter().any(|(uid, _)| uid == author_id);
        }
        self.author_affinity_index.contains_key(author_id)
    }

    /// Centres d'intérêt présents dans ce texte — la question de D2 et de D8,
    /// posée une seule fois.
    ///
    /// `content_lower` doit déjà être en minuscules : `top_words` l'est par
    /// construction, et l'automate ne replie pas la casse.
    pub fn keyword_hits(&self, content_lower: &str) -> KeywordHits {
        let Some(matcher) = self.word_matcher.as_ref() else {
            return self.keyword_hits_linear(content_lower);
        };

        // `find_overlapping_iter` et non `find_iter` : un mot-clé peut être
        // contenu dans un autre (« chat » dans « chateau »). Avec les matchs
        // non chevauchants, le plus long masquerait le plus court et D2
        // compterait une correspondance de moins que `contains`.
        let mut seen: u64 = 0;
        for m in matcher.find_overlapping_iter(content_lower) {
            seen |= 1u64 << m.pattern().as_usize();
        }

        let mut hits = KeywordHits::default();
        if seen == 0 {
            return hits;
        }
        for (i, (_, count)) in self.top_words.iter().enumerate() {
            if seen & (1u64 << i) == 0 {
                continue;
            }
            hits.matches += 1;
            if i < D8_KEYWORD_DEPTH {
                hits.top_matches += 1;
                hits.weighted += (*count as f64).ln() / 5.0;
            }
        }
        hits
    }

    /// Le chemin d'avant, gardé comme repli ET comme référence de justesse : un
    /// test compare les deux sur les mêmes entrées.
    pub fn keyword_hits_linear(&self, content_lower: &str) -> KeywordHits {
        let mut hits = KeywordHits::default();
        for (i, (word, count)) in self.top_words.iter().enumerate() {
            if !content_lower.contains(word.as_str()) {
                continue;
            }
            hits.matches += 1;
            if i < D8_KEYWORD_DEPTH {
                hits.top_matches += 1;
                hits.weighted += (*count as f64).ln() / 5.0;
            }
        }
        hits
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
    fn member(set: &FxHashSet<String>, list: &[String], needle: &str) -> bool {
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
    Ultra,
}

impl AuthorTier {
    pub fn resolve(tier: &str, legacy_premium: bool) -> Self {
        match tier {
            "ultra" => Self::Ultra,
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
    /// Texte analysé une fois pour toutes à l'entrée du pipeline — voir
    /// [`crate::content`]. Remplace le `words: Vec<String>` d'avant, qui
    /// recopiait jusqu'à cinquante mots par candidat pour un détecteur qui ne
    /// fait que les compter et les dédupliquer.
    ///
    /// ⚠ Dérivé de `content` : construit par `analyze_content`, jamais à la
    /// main. Un `RawTweet` monté par `..Default::default()` avec un `content`
    /// mais sans `text` retombe sur le calcul à la volée (voir
    /// [`RawTweet::content_lower`]) plutôt que de faire comme si le tweet
    /// était vide.
    pub text: ContentFeatures,

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
impl RawTweet {
    /// Le texte du tweet en minuscules.
    ///
    /// Rendu depuis l'analyse faite à l'entrée du pipeline quand elle existe,
    /// recalculé à la volée sinon. Ce repli est le même parti que
    /// `UserProfile::member` : un champ dérivé qu'on a oublié de remplir doit
    /// coûter du temps, pas fausser un classement. Sans lui, un `RawTweet`
    /// monté à la main avec un `content` mais sans `text` — c'est le cas de la
    /// centaine de tweets de test construits par `..Default::default()` — se
    /// comporterait comme un tweet au texte VIDE : D2 et D8 ne trouveraient
    /// jamais un seul centre d'intérêt, sans la moindre erreur.
    #[inline]
    pub fn content_lower(&self) -> std::borrow::Cow<'_, str> {
        if self.text.lower.is_empty() && !self.content.is_empty() {
            return std::borrow::Cow::Owned(self.content.to_lowercase());
        }
        std::borrow::Cow::Borrowed(&self.text.lower)
    }

    /// Remplit `text` depuis `content`. À appeler sur tout tweet construit
    /// autrement que par `map_rows`.
    pub fn analyze(&mut self) {
        self.text = analyze_content(&self.content);
    }
}

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
            text: ContentFeatures::default(),
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
    /// Score final de classement, et confiance du moteur dans cette décision
    /// (voir `algorithm::scoring::ranking_confidence`).
    ///
    /// Mis en cache avec le reste de l'entrée, pour la même raison que
    /// `parent_id` : ce sont les seuls instants où l'on connaît encore le
    /// tweet complet. Les recalculer à la lecture du cache exigerait de tout
    /// recharger depuis la base à chaque page servie.
    ///
    /// `#[serde(default)]` : les clés déjà écrites par la version précédente
    /// n'en portent pas et doivent rester lisibles pendant leur TTL — sans ça,
    /// tout lecteur ayant un fil en cache le verrait recalculé intégralement
    /// au déploiement.
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub confidence: f64,
}

/// Lien de conversation exposé au client : « ce tweet répond à celui-là, et le
/// fil part de cette racine ».
///
/// Redondant avec l'ordre de `tweet_ids` — l'adjacence dit déjà la même chose —
/// et c'est voulu : l'ordre est une convention fragile qu'un intermédiaire peut
/// casser sans s'en rendre compte, ce champ est une affirmation explicite. Un
/// client qui reçoit les deux peut vérifier qu'ils concordent.
/// Ce que le moteur a pensé d'un tweet de CETTE page.
///
/// Exposé parce que le client en a besoin pour décider quand DEMANDER : la
/// question explicite (« ça t'intéresse ? ») n'a de sens que là où le moteur
/// hésite, et jusqu'ici l'app n'avait aucun moyen de le savoir — elle
/// retombait sur une heuristique de silence.
#[derive(Debug, Clone, Serialize)]
pub struct TweetScore {
    pub tweet_id: String,
    /// Score final de classement, dans [0,1].
    pub score: f64,
    /// Sur quoi cette décision s'appuie, dans [0,1]. BAS = le moteur devine.
    /// Ce n'est pas « ce tweet est mauvais » — voir `ranking_confidence`.
    pub confidence: f64,
}

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
    /// Score et confiance de chaque tweet de CETTE page, dans le même ordre
    /// que `tweet_ids`.
    pub scores: Vec<TweetScore>,
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

    // ─── Affinité d'auteur indexée ───────────────────────────────────────────

    fn profil_avec_auteurs(auteurs: Vec<(String, f64)>) -> UserProfile {
        let mut p = UserProfile {
            top_authors: auteurs,
            ..Default::default()
        };
        p.rebuild_indexes();
        p
    }

    #[test]
    fn l_affinite_indexee_repond_comme_le_balayage() {
        let auteurs: Vec<(String, f64)> = (0..40)
            .map(|i| (format!("auteur-{i}"), 1.0 - i as f64 * 0.02))
            .collect();
        let p = profil_avec_auteurs(auteurs.clone());
        for (id, attendu) in &auteurs {
            assert_eq!(p.author_affinity(id), *attendu, "auteur {id}");
        }
        assert_eq!(p.author_affinity("inconnu"), 0.0);
    }

    /// Le balayage remplacé s'arrêtait au PREMIER auteur trouvé. Une collecte
    /// naïve dans une table garderait le dernier : deux lignes pour le même
    /// auteur suffiraient alors à changer un score, en silence.
    #[test]
    fn sur_un_auteur_en_double_c_est_la_premiere_ligne_qui_compte() {
        let p = profil_avec_auteurs(vec![
            ("double".into(), 0.90),
            ("autre".into(), 0.10),
            ("double".into(), 0.20),
        ]);
        assert_eq!(p.author_affinity("double"), 0.90);
    }

    /// « Connu » et « apprécié » ne sont pas la même question, et le bandit a
    /// besoin de la première : c'est « jamais vu » qui rend un tweet candidat à
    /// l'exploration, pas « vu et laissé indifférent ». Un auteur inscrit avec
    /// une affinité nulle doit donc être connu.
    #[test]
    fn un_auteur_d_affinite_nulle_reste_un_auteur_connu() {
        let p = profil_avec_auteurs(vec![("froid".into(), 0.0), ("chaud".into(), 0.8)]);
        assert!(p.knows_author("froid"));
        assert_eq!(p.author_affinity("froid"), 0.0);
        assert!(p.knows_author("chaud"));
        assert!(!p.knows_author("inconnu"));
    }

    #[test]
    fn l_auteur_connu_se_reconnait_aussi_sans_index() {
        let sans_index = UserProfile {
            top_authors: vec![("froid".into(), 0.0)],
            ..Default::default()
        };
        assert!(sans_index.author_affinity_index.is_empty());
        assert!(sans_index.knows_author("froid"));
        assert!(!sans_index.knows_author("inconnu"));
    }

    #[test]
    fn l_affinite_repond_juste_meme_sans_index() {
        // Même garde-fou que `follows` : un `rebuild_indexes()` oublié doit
        // coûter du temps, pas fausser un score.
        let sans_index = UserProfile {
            top_authors: vec![("a".into(), 0.7)],
            ..Default::default()
        };
        assert!(sans_index.author_affinity_index.is_empty());
        assert_eq!(sans_index.author_affinity("a"), 0.7);
        assert_eq!(sans_index.author_affinity("b"), 0.0);
    }

    // ─── Recherche des centres d'intérêt ─────────────────────────────────────

    fn profil_avec_mots(mots: &[(&str, u32)]) -> UserProfile {
        let mut p = UserProfile {
            top_words: mots
                .iter()
                .map(|(w, n)| ((*w).to_string(), *n))
                .collect(),
            ..Default::default()
        };
        p.rebuild_indexes();
        p
    }

    /// L'automate remplace une cinquantaine de `str::contains` par candidat.
    /// C'est le remplacement le plus risqué de tout le lot : s'il ne trouve pas
    /// exactement les mêmes motifs, D2 et D8 changent de valeur sans que rien
    /// ne le signale. On compare donc les deux chemins terme à terme, sur des
    /// cas choisis pour casser les implémentations naïves.
    #[test]
    fn l_automate_et_le_balayage_lineaire_disent_la_meme_chose() {
        let profils = [
            // Un motif contenu dans un autre : c'est ce qui distingue une
            // recherche chevauchante d'une recherche gloutonne. Avec des matchs
            // non chevauchants, « chat » disparaîtrait dans « chateau ».
            profil_avec_mots(&[("chat", 9), ("chateau", 4), ("eau", 7)]),
            // Motifs qui se recouvrent partiellement.
            profil_avec_mots(&[("abab", 3), ("baba", 5), ("aba", 2)]),
            // Au-delà des 20 premiers : seuls ceux-là comptent pour D8.
            profil_avec_mots(
                &(0..30)
                    .map(|i| (["alpha", "beta", "gamma"][i % 3], 2 + i as u32))
                    .collect::<Vec<_>>()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
            profil_avec_mots(&[("accentué", 6), ("œuf", 3), ("straße", 5)]),
            profil_avec_mots(&[]),
        ];
        let textes = [
            "",
            "chat",
            "un chateau au bord de l'eau avec un chat",
            "abababa",
            "alpha beta gamma alpha",
            "rien de tout cela ici",
            "accentué œuf straße",
            "CHAT en majuscules ne compte pas, la recherche est deja minuscule",
            "chatchatchat",
        ];
        for (i, p) in profils.iter().enumerate() {
            for texte in textes {
                assert_eq!(
                    p.keyword_hits(texte),
                    p.keyword_hits_linear(texte),
                    "profil {i}, texte {texte:?}"
                );
            }
        }
    }

    /// Le repli doit être exact, pas seulement présent : c'est lui qui sert
    /// pour un profil relu du cache avant reconstruction.
    #[test]
    fn sans_automate_la_recherche_reste_juste() {
        let sans_index = UserProfile {
            top_words: vec![("chat".into(), 9), ("chien".into(), 4)],
            ..Default::default()
        };
        assert!(sans_index.word_matcher.is_none());
        let hits = sans_index.keyword_hits("un chat et un chien");
        assert_eq!(hits.matches, 2);
        assert_eq!(hits.top_matches, 2);
    }

    /// Un mot-clé vide est trouvé dans n'importe quel texte (`"".contains("")`
    /// vaut `true`) et décale les index de motifs. On refuse alors de
    /// construire l'automate plutôt que d'en construire un qui se trompe de
    /// mot.
    #[test]
    fn un_mot_cle_vide_desactive_l_automate_sans_changer_le_resultat() {
        let p = profil_avec_mots(&[("chat", 9), ("", 4)]);
        assert!(p.word_matcher.is_none());
        assert_eq!(
            p.keyword_hits("un chat"),
            p.keyword_hits_linear("un chat")
        );
    }

    #[test]
    fn au_dela_du_masque_de_bits_on_retombe_sur_le_balayage() {
        let mots: Vec<(String, u32)> = (0..MAX_INDEXED_KEYWORDS + 1)
            .map(|i| (format!("mot{i}"), 2 + i as u32))
            .collect();
        let mut p = UserProfile {
            top_words: mots,
            ..Default::default()
        };
        p.rebuild_indexes();
        assert!(p.word_matcher.is_none(), "trop de motifs pour le masque");
        let texte = "mot0 mot7 mot64 quelque chose";
        assert_eq!(p.keyword_hits(texte), p.keyword_hits_linear(texte));
    }

    /// D8 additionne des flottants : l'ORDRE des termes fait le résultat.
    /// L'automate rend ses motifs dans l'ordre du texte, la somme doit rester
    /// dans l'ordre des mots-clés.
    #[test]
    fn la_somme_de_d8_suit_l_ordre_des_mots_cles_pas_du_texte() {
        let p = profil_avec_mots(&[("premier", 7), ("second", 13), ("tiers", 3)]);
        let a = p.keyword_hits("tiers second premier");
        let b = p.keyword_hits("premier second tiers");
        assert_eq!(a.weighted.to_bits(), b.weighted.to_bits());
        assert_eq!(a.weighted.to_bits(), p.keyword_hits_linear("tiers second premier").weighted.to_bits());
    }

    // ─── Minuscules du tweet ─────────────────────────────────────────────────

    #[test]
    fn content_lower_retombe_sur_le_calcul_quand_l_analyse_manque() {
        // Le cas de tous les tweets montés par `..Default::default()`.
        let brut = RawTweet {
            content: "Du TEXTE en Majuscules".into(),
            ..Default::default()
        };
        assert!(brut.text.lower.is_empty());
        assert_eq!(&*brut.content_lower(), "du texte en majuscules");

        let mut analyse = brut.clone();
        analyse.analyze();
        assert_eq!(&*analyse.content_lower(), &*brut.content_lower());
    }

    #[test]
    fn un_tweet_sans_texte_ne_declenche_pas_le_repli() {
        let vide = RawTweet::default();
        assert_eq!(&*vide.content_lower(), "");
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;

    /// Le classement des trois refus doit être strict et dans cet ordre :
    /// bloquer > signaler > ne pas être intéressé. C'est l'inversion de cet
    /// ordre qui était le bug — seul le plus faible des trois agissait.
    #[test]
    fn les_refus_sont_ordonnes_du_plus_faible_au_plus_fort() {
        let pas_interesse = InteractionType::NotInterested.refusal_strikes().unwrap();
        let signalement = InteractionType::Report.refusal_strikes().unwrap();
        let blocage = InteractionType::Block.refusal_strikes().unwrap();
        assert!(pas_interesse < signalement, "{pas_interesse} < {signalement}");
        assert!(signalement < blocage, "{signalement} < {blocage}");
    }

    /// Ignorer un tweet ne dit rien de son auteur, ni un geste positif.
    #[test]
    fn seuls_les_refus_explicites_mettent_un_auteur_en_sourdine() {
        for muet in [
            InteractionType::Skip,
            InteractionType::Like,
            InteractionType::View,
            InteractionType::Unlike,
            InteractionType::Interested,
        ] {
            assert!(
                muet.refusal_strikes().is_none(),
                "{muet:?} ne doit pas viser l'auteur"
            );
        }
    }

    /// Les trois gestes qui mettent un auteur en sourdine doivent aussi être
    /// étiquetés négatifs pour le modèle CTR : sourdine et apprentissage ne
    /// doivent pas se contredire.
    #[test]
    fn un_refus_est_toujours_un_negatif_pour_le_modele() {
        for refus in [
            InteractionType::NotInterested,
            InteractionType::Report,
            InteractionType::Block,
        ] {
            assert!(refus.refusal_strikes().is_some());
            assert_eq!(refus.ctr_label(), Some(false), "{refus:?}");
            assert!(refus.weight() < 0.0, "{refus:?}");
        }
    }
}
