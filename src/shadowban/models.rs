use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::models::RecommendMode;

// ─── Surfaces de distribution ────────────────────────────────────────────────

/// Où un tweet peut être servi. Une restriction porte sur une surface précise,
/// pas sur « la visibilité » en général.
///
/// C'est la distinction que TikTok a formalisée en 2023 en remplaçant son
/// ancien système, jugé opaque, par des restrictions nommées surface par
/// surface : un contenu écarté du fil « For You » reste sur le profil et
/// atteint toujours les abonnés — il n'est pas supprimé, il n'est plus
/// recommandé. Le compte peut être sain et le post écarté, ou l'inverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Fil d'abonnement — ce que le lecteur a explicitement demandé à voir.
    FollowerFeed,
    /// Recommandation personnalisée (équivalent du « For You » de TikTok).
    ForYou,
    /// Découverte explicite.
    Discover,
    /// Tendances / page Explorer.
    Trending,
}

impl Surface {
    pub fn from_mode(mode: &RecommendMode) -> Self {
        match mode {
            RecommendMode::Feed => Surface::FollowerFeed,
            RecommendMode::ForYou => Surface::ForYou,
            RecommendMode::Discover => Surface::Discover,
            RecommendMode::Trending => Surface::Trending,
        }
    }

    /// Vrai si la surface expose le contenu à des lecteurs qui n'ont rien
    /// demandé — c'est-à-dire tout sauf le fil d'abonnement.
    pub fn is_recommendation(self) -> bool {
        !matches!(self, Surface::FollowerFeed)
    }

    pub fn label(self) -> &'static str {
        match self {
            Surface::FollowerFeed => "follower_feed",
            Surface::ForYou => "for_you",
            Surface::Discover => "discover",
            Surface::Trending => "trending",
        }
    }
}

// ─── Niveaux de suppression gradués ──────────────────────────────────────────

/// Restriction de compte à 4 niveaux : progressive, réversible, bornée dans le
/// temps (voir [`StrikeLedger`]).
///
/// Le compte continue de poster normalement et son contenu reste accessible sur
/// son profil : seule la distribution est réduite, et surface par surface.
///
/// ⚠ Pas de `rename_all` ici, contrairement aux autres énumérations de ce
/// module : le panneau d'administration envoie déjà `"Monitoring"` /
/// `"Suppressed"` / `"Ghosted"` en capitale initiale (voir la construction du
/// corps dans `admin/ui/admin_panel.html`). Passer en `snake_case` ferait
/// échouer la désérialisation de `SetShadowbanRequest` et le bouton
/// « Apply Shadowban » renverrait un 422. Les surfaces récentes utilisent
/// `label()` / `level_label`, en minuscules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ShadowbanLevel {
    #[default]
    Clean, // Visibilité normale — aucune suppression
    Monitoring, // Rétrogradation légère (-15%) — aucune surface fermée
    Suppressed, // Rétrogradation forte (-55%) + exclu de Trending & Discover
    Ghosted,    // Exclu de toute recommandation — reste visible des abonnés
}

impl ShadowbanLevel {
    /// Multiplicateur appliqué au final_score.
    ///
    /// Il ne remplace pas [`Self::restricts`] : un multiplicateur, même à 0,05,
    /// reste franchissable par un contenu à très forte vélocité. Une surface
    /// fermée l'est en dur.
    pub fn score_multiplier(self) -> f64 {
        match self {
            ShadowbanLevel::Clean => 1.00,
            ShadowbanLevel::Monitoring => 0.85,
            ShadowbanLevel::Suppressed => 0.45,
            ShadowbanLevel::Ghosted => 0.05,
        }
    }

    /// Vrai si cette surface est fermée à ce niveau.
    ///
    /// `Monitoring` ne ferme rien : c'est un palier d'alerte, pas une sanction.
    /// Aucun niveau ne ferme le fil d'abonnement — un lecteur qui suit un compte
    /// a fait un choix explicite, et le lui retirer en silence est précisément
    /// ce que « shadowban » désigne péjorativement.
    pub fn restricts(self, surface: Surface) -> bool {
        match self {
            ShadowbanLevel::Clean | ShadowbanLevel::Monitoring => false,
            ShadowbanLevel::Suppressed => {
                matches!(surface, Surface::Trending | Surface::Discover)
            }
            ShadowbanLevel::Ghosted => surface.is_recommendation(),
        }
    }

    pub fn excludes_discovery(self) -> bool {
        self.restricts(Surface::Discover)
    }
    pub fn excludes_trending(self) -> bool {
        self.restricts(Surface::Trending)
    }

    pub fn label(self) -> &'static str {
        match self {
            ShadowbanLevel::Clean => "clean",
            ShadowbanLevel::Monitoring => "monitoring",
            ShadowbanLevel::Suppressed => "suppressed",
            ShadowbanLevel::Ghosted => "ghosted",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "clean" => Some(ShadowbanLevel::Clean),
            "monitoring" => Some(ShadowbanLevel::Monitoring),
            "suppressed" => Some(ShadowbanLevel::Suppressed),
            "ghosted" => Some(ShadowbanLevel::Ghosted),
            _ => None,
        }
    }

    /// Phrase affichable au créateur concerné. Un compte restreint doit pouvoir
    /// lire son état : c'est le seul moyen de le corriger.
    pub fn creator_summary(self) -> &'static str {
        match self {
            ShadowbanLevel::Clean => {
                "Ton compte est en règle. Tes posts sont distribués normalement."
            }
            ShadowbanLevel::Monitoring => {
                "Un avertissement est actif sur ton compte. Tes posts sont encore \
                 recommandés, mais un peu moins mis en avant."
            }
            ShadowbanLevel::Suppressed => {
                "Ton compte est temporairement retiré des Tendances et de la \
                 Découverte. Tes abonnés voient toujours tout ce que tu publies."
            }
            ShadowbanLevel::Ghosted => {
                "Ton compte n'est temporairement plus recommandé. Tes posts \
                 restent en ligne et visibles par tes abonnés et sur ton profil."
            }
        }
    }
}

// ─── Domaines de sanction ────────────────────────────────────────────────────

/// Durée de vie d'un avertissement.
///
/// 90 jours, comme chez TikTok : passé ce délai un avertissement « expire et
/// n'est plus pris en compte ». C'est le point qui manquait le plus ici — les
/// clés `admin:shadowban:*` n'avaient aucune expiration, donc une restriction
/// posée une fois durait indéfiniment jusqu'à intervention manuelle. Un système
/// sans chemin de retour ne corrige aucun comportement : il crée juste des
/// comptes morts.
pub const STRIKE_TTL_DAYS: i64 = 90;

pub fn strike_ttl() -> Duration {
    Duration::days(STRIKE_TTL_DAYS)
}

/// Domaine de règle enfreint. Le seuil de sanction dépend du domaine.
///
/// TikTok le formule ainsi : les seuils « varient selon le potentiel de nuisance
/// d'une infraction » — plus stricts pour la promotion d'idéologies haineuses
/// que pour du spam à faible nuisance. Un compte est sanctionné quand il franchit
/// le seuil **d'un** domaine, pas d'un total mélangé : dix spams ne valent pas
/// un appel à la violence, et l'inverse non plus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrikePolicy {
    /// Spam, contenu automatisé, publication en rafale.
    Spam,
    /// Manipulation de l'engagement : « like pour like », faux concours.
    EngagementBait,
    /// Contenu repris sans apport, republication en boucle.
    Unoriginal,
    /// Information trompeuse sur un sujet vérifiable.
    Misinformation,
    /// Harcèlement, insultes ciblées.
    Harassment,
    /// Contenu sexuel non signalé.
    AdultContent,
    /// Haine visant un groupe protégé.
    HatefulConduct,
    /// Menace ou apologie de violence, contenu à nuisance immédiate.
    ViolentThreat,
}

/// Seuils d'un domaine : nombre d'avertissements actifs à partir duquel chaque
/// niveau s'applique, puis le nombre qui justifie un bannissement définitif.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrikeThresholds {
    pub monitoring: u32,
    pub suppressed: u32,
    pub ghosted: u32,
    /// Franchi → le compte est proposé au bannissement définitif (décision humaine).
    pub permanent: u32,
}

impl StrikePolicy {
    pub fn label(self) -> &'static str {
        match self {
            StrikePolicy::Spam => "spam",
            StrikePolicy::EngagementBait => "engagement_bait",
            StrikePolicy::Unoriginal => "unoriginal",
            StrikePolicy::Misinformation => "misinformation",
            StrikePolicy::Harassment => "harassment",
            StrikePolicy::AdultContent => "adult_content",
            StrikePolicy::HatefulConduct => "hateful_conduct",
            StrikePolicy::ViolentThreat => "violent_threat",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "spam" => StrikePolicy::Spam,
            "engagement_bait" => StrikePolicy::EngagementBait,
            "unoriginal" => StrikePolicy::Unoriginal,
            "misinformation" => StrikePolicy::Misinformation,
            "harassment" => StrikePolicy::Harassment,
            "adult_content" => StrikePolicy::AdultContent,
            "hateful_conduct" => StrikePolicy::HatefulConduct,
            "violent_threat" => StrikePolicy::ViolentThreat,
            _ => return None,
        })
    }

    /// Domaine déduit de la catégorie de toxicité posée par l'annotateur LLM
    /// (vocabulaire fermé, voir `annotator-schema.json` côté API — c'est la
    /// source de vérité).
    ///
    /// Sans ça, tout contenu toxique retombe sur [`Self::Harassment`] via
    /// [`IneligibilityReason::policy`] : une menace et une vulgarité légère
    /// compteraient alors pour le même seuil, alors que [`Self::ViolentThreat`]
    /// existe précisément pour couper dès le premier fait constaté et que
    /// [`Self::HatefulConduct`] a un seuil bien plus bas que le harcèlement
    /// générique.
    pub fn from_toxicity_category(category: &str) -> Self {
        match category {
            "menace" => StrikePolicy::ViolentThreat,
            "haine_identitaire" => StrikePolicy::HatefulConduct,
            "contenu_sexuel" => StrikePolicy::AdultContent,
            // "harcelement", "insulte_ciblee", "vulgarite_legere", "aucune"
            // (ne devrait pas atteindre cet appel — voir le seuil de score
            // dans `eligibility::content_eligibility`) et toute valeur future
            // inconnue de l'annotateur : le domaine générique reste le repli
            // le plus sûr.
            _ => StrikePolicy::Harassment,
        }
    }

    /// Formulation destinée au créateur — pas le nom technique du domaine.
    pub fn creator_reason(self) -> &'static str {
        match self {
            StrikePolicy::Spam => "Publication répétitive ou automatisée",
            StrikePolicy::EngagementBait => "Incitation artificielle à l'engagement",
            StrikePolicy::Unoriginal => "Contenu repris sans apport",
            StrikePolicy::Misinformation => "Information trompeuse",
            StrikePolicy::Harassment => "Harcèlement ou insultes ciblées",
            StrikePolicy::AdultContent => "Contenu à caractère sexuel",
            StrikePolicy::HatefulConduct => "Propos haineux",
            StrikePolicy::ViolentThreat => "Menace ou apologie de violence",
        }
    }

    /// Infraction à nuisance immédiate : un seul avertissement suffit à couper
    /// la recommandation, sans passer par les paliers.
    pub fn is_severe(self) -> bool {
        matches!(self, StrikePolicy::ViolentThreat)
    }

    pub fn thresholds(self) -> StrikeThresholds {
        match self {
            // Faible nuisance : large marge avant la moindre fermeture de surface.
            StrikePolicy::Spam => StrikeThresholds {
                monitoring: 2,
                suppressed: 4,
                ghosted: 7,
                permanent: 12,
            },
            StrikePolicy::EngagementBait => StrikeThresholds {
                monitoring: 2,
                suppressed: 4,
                ghosted: 7,
                permanent: 12,
            },
            StrikePolicy::Unoriginal => StrikeThresholds {
                monitoring: 3,
                suppressed: 6,
                ghosted: 10,
                permanent: 16,
            },
            // Nuisance moyenne.
            StrikePolicy::Misinformation => StrikeThresholds {
                monitoring: 1,
                suppressed: 3,
                ghosted: 5,
                permanent: 8,
            },
            StrikePolicy::AdultContent => StrikeThresholds {
                monitoring: 1,
                suppressed: 3,
                ghosted: 5,
                permanent: 8,
            },
            // Nuisance forte : visant une personne ou un groupe.
            StrikePolicy::Harassment => StrikeThresholds {
                monitoring: 1,
                suppressed: 2,
                ghosted: 4,
                permanent: 6,
            },
            StrikePolicy::HatefulConduct => StrikeThresholds {
                monitoring: 1,
                suppressed: 2,
                ghosted: 3,
                permanent: 4,
            },
            // Nuisance immédiate : plus aucune recommandation dès le premier fait.
            StrikePolicy::ViolentThreat => StrikeThresholds {
                monitoring: 1,
                suppressed: 1,
                ghosted: 1,
                permanent: 2,
            },
        }
    }

    /// Niveau impliqué par `count` avertissements actifs dans ce domaine.
    pub fn level_for(self, count: u32) -> ShadowbanLevel {
        let t = self.thresholds();
        if count == 0 {
            ShadowbanLevel::Clean
        } else if count >= t.ghosted {
            ShadowbanLevel::Ghosted
        } else if count >= t.suppressed {
            ShadowbanLevel::Suppressed
        } else if count >= t.monitoring {
            ShadowbanLevel::Monitoring
        } else {
            ShadowbanLevel::Clean
        }
    }

    pub const ALL: [StrikePolicy; 8] = [
        StrikePolicy::Spam,
        StrikePolicy::EngagementBait,
        StrikePolicy::Unoriginal,
        StrikePolicy::Misinformation,
        StrikePolicy::Harassment,
        StrikePolicy::AdultContent,
        StrikePolicy::HatefulConduct,
        StrikePolicy::ViolentThreat,
    ];
}

/// Un avertissement daté. Il expire seul, [`STRIKE_TTL_DAYS`] jours après son émission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strike {
    pub policy: StrikePolicy,
    pub issued_at: DateTime<Utc>,
    /// Tweet à l'origine de l'avertissement, quand il y en a un.
    #[serde(default)]
    pub tweet_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

impl Strike {
    pub fn expires_at(&self) -> DateTime<Utc> {
        self.issued_at + strike_ttl()
    }
    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        self.expires_at() > now
    }
}

// ─── Inéligibilité au niveau du contenu ──────────────────────────────────────

/// Pourquoi un tweet précis n'est plus recommandé.
///
/// Distinct du compte : le post reste en ligne, sur le profil et dans le fil des
/// abonnés — il n'entre simplement plus dans les surfaces de recommandation.
/// C'est la moitié du modèle TikTok qui manquait entièrement ici : la seule
/// réponse au contenu douteux était `GARBAGE_HARD_DROP`, un retrait pur et
/// simple de tout le pipeline, y compris pour les lecteurs abonnés.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IneligibilityReason {
    /// Signaux de spam agrégés au-delà du seuil.
    SpamSignals,
    /// Texte massivement dupliqué ou template.
    Unoriginal,
    /// Presque rien d'autre que des liens.
    LinkSpam,
    /// Hashtags/mentions en masse pour capter de l'audience.
    EngagementBait,
    /// Taux de signalement anormal — en attente de modération humaine.
    UnderReview,
    /// Agressif sans franchir le seuil de retrait.
    Toxic,
    /// Aucun apport détectable, en dessous du plancher de qualité.
    LowQuality,
}

impl IneligibilityReason {
    pub fn label(self) -> &'static str {
        match self {
            IneligibilityReason::SpamSignals => "spam_signals",
            IneligibilityReason::Unoriginal => "unoriginal",
            IneligibilityReason::LinkSpam => "link_spam",
            IneligibilityReason::EngagementBait => "engagement_bait",
            IneligibilityReason::UnderReview => "under_review",
            IneligibilityReason::Toxic => "toxic",
            IneligibilityReason::LowQuality => "low_quality",
        }
    }

    /// Domaine de sanction correspondant, quand une répétition doit compter au
    /// niveau du compte. `None` = motif qui ne se cumule pas (ex. en attente de
    /// modération : rien n'est encore établi).
    pub fn policy(self) -> Option<StrikePolicy> {
        match self {
            IneligibilityReason::SpamSignals => Some(StrikePolicy::Spam),
            IneligibilityReason::LinkSpam => Some(StrikePolicy::Spam),
            IneligibilityReason::Unoriginal => Some(StrikePolicy::Unoriginal),
            IneligibilityReason::EngagementBait => Some(StrikePolicy::EngagementBait),
            IneligibilityReason::Toxic => Some(StrikePolicy::Harassment),
            IneligibilityReason::UnderReview => None,
            IneligibilityReason::LowQuality => None,
        }
    }

    /// Comme [`Self::policy`], mais affine `Toxic` avec la catégorie posée par
    /// l'annotateur LLM quand elle est disponible — voir
    /// [`StrikePolicy::from_toxicity_category`]. Les autres motifs n'ont pas
    /// de sous-catégorie : `policy()` fait déjà foi pour eux.
    pub fn policy_for(self, toxicity_category: Option<&str>) -> Option<StrikePolicy> {
        match self {
            IneligibilityReason::Toxic => Some(
                toxicity_category
                    .map(StrikePolicy::from_toxicity_category)
                    .unwrap_or(StrikePolicy::Harassment),
            ),
            other => other.policy(),
        }
    }

    pub fn creator_reason(self) -> &'static str {
        match self {
            IneligibilityReason::SpamSignals => "Ce post présente des signaux de spam",
            IneligibilityReason::Unoriginal => "Ce post reprend un contenu sans apport",
            IneligibilityReason::LinkSpam => "Ce post ne contient presque que des liens",
            IneligibilityReason::EngagementBait => {
                "Ce post cherche à capter l'engagement artificiellement"
            }
            IneligibilityReason::UnderReview => "Ce post est en cours de vérification",
            IneligibilityReason::Toxic => "Ce post a été jugé agressif",
            IneligibilityReason::LowQuality => "Ce post n'a pas atteint le seuil de qualité",
        }
    }
}

/// État de distribution d'un tweet donné.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentEligibility {
    /// Recommandable partout.
    #[default]
    Eligible,
    /// Reste en ligne et atteint les abonnés, mais n'est plus recommandé.
    NotRecommended(IneligibilityReason),
}

impl ContentEligibility {
    pub fn is_eligible(self) -> bool {
        matches!(self, ContentEligibility::Eligible)
    }
    pub fn reason(self) -> Option<IneligibilityReason> {
        match self {
            ContentEligibility::Eligible => None,
            ContentEligibility::NotRecommended(r) => Some(r),
        }
    }
}

// ─── Signaux de contenu poubelle (par tweet) ──────────────────────────────────

/// Signaux extraits d'un tweet individuel.
#[derive(Debug, Clone, Default)]
pub struct GarbageSignals {
    pub spam_hashtag_density: bool, // hashtags > 30% des mots
    pub spam_mentions: bool,        // > 5 @mentions
    pub zero_engagement: bool,      // > 200 vues, zéro réaction
    pub high_report_rate: bool,     // rapport/vues > 3%
    pub pure_link_spam: bool,       // ≥ 2 URLs + texte < 50 chars
    pub emoji_overload: bool,       // > 8 émojis + texte < 100 chars
    pub repeat_content: bool,       // < 25% de mots uniques (texte dupliqué)
    /// Compte créé il y a ≤ 2 jours avec déjà ≥ 50 tweets publiés — la
    /// signature d'un compte créé pour publier en masse, pas pour échanger.
    pub new_account_burst: bool,
}

impl GarbageSignals {
    /// Score de poubelle 0–1 pondéré par la gravité du signal.
    pub fn score(&self) -> f64 {
        let mut s = 0.0_f64;
        if self.high_report_rate {
            s += 0.35;
        } // signal le plus fort
        if self.zero_engagement {
            s += 0.25;
        }
        if self.pure_link_spam {
            s += 0.15;
        }
        if self.spam_hashtag_density {
            s += 0.12;
        }
        if self.repeat_content {
            s += 0.10;
        }
        if self.new_account_burst {
            s += 0.10;
        }
        if self.spam_mentions {
            s += 0.08;
        }
        if self.emoji_overload {
            s += 0.05;
        }
        s.clamp(0.0, 1.0)
    }

    pub fn is_garbage(&self) -> bool {
        self.score() >= 0.40
    }

    /// Motif d'inéligibilité dominant, s'il y en a un.
    ///
    /// Ordonné par spécificité : un motif nommé vaut mieux qu'un « signaux de
    /// spam » générique, parce qu'il est affichable au créateur et actionnable
    /// par lui.
    pub fn dominant_reason(&self) -> Option<IneligibilityReason> {
        if self.high_report_rate {
            return Some(IneligibilityReason::UnderReview);
        }
        if self.pure_link_spam {
            return Some(IneligibilityReason::LinkSpam);
        }
        if self.repeat_content {
            return Some(IneligibilityReason::Unoriginal);
        }
        if self.spam_hashtag_density || self.spam_mentions {
            return Some(IneligibilityReason::EngagementBait);
        }
        if self.emoji_overload || self.zero_engagement || self.new_account_burst {
            return Some(IneligibilityReason::SpamSignals);
        }
        None
    }

    /// Résumé des signaux actifs pour les logs.
    pub fn active_signals(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.high_report_rate {
            v.push("high_report_rate");
        }
        if self.zero_engagement {
            v.push("zero_engagement");
        }
        if self.pure_link_spam {
            v.push("pure_link_spam");
        }
        if self.spam_hashtag_density {
            v.push("spam_hashtag_density");
        }
        if self.repeat_content {
            v.push("repeat_content");
        }
        if self.spam_mentions {
            v.push("spam_mentions");
        }
        if self.emoji_overload {
            v.push("emoji_overload");
        }
        if self.new_account_burst {
            v.push("new_account_burst");
        }
        v
    }
}
