use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─── Niveaux de suppression gradués ──────────────────────────────────────────

/// Shadowban à 4 niveaux : progressif, silencieux, réversible.
/// Le compte continue de poster normalement — seule la distribution est affectée.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ShadowbanLevel {
    #[default]
    Clean,       // Visibilité normale — aucune suppression
    Monitoring,  // Légère suppression (-15%) — alerte qualité
    Suppressed,  // Suppression forte (-55%) — exclu de Discovery & Trending
    Ghosted,     // Quasi-invisible (-95%) — visible uniquement par les abonnés directs
}

impl ShadowbanLevel {
    /// Multiplicateur appliqué au final_score.
    pub fn score_multiplier(self) -> f64 {
        match self {
            ShadowbanLevel::Clean      => 1.00,
            ShadowbanLevel::Monitoring => 0.85,
            ShadowbanLevel::Suppressed => 0.45,
            ShadowbanLevel::Ghosted    => 0.05,
        }
    }

    pub fn excludes_discovery(self) -> bool {
        matches!(self, ShadowbanLevel::Suppressed | ShadowbanLevel::Ghosted)
    }

    pub fn excludes_trending(self) -> bool {
        matches!(self, ShadowbanLevel::Suppressed | ShadowbanLevel::Ghosted)
    }

    pub fn label(self) -> &'static str {
        match self {
            ShadowbanLevel::Clean      => "clean",
            ShadowbanLevel::Monitoring => "monitoring",
            ShadowbanLevel::Suppressed => "suppressed",
            ShadowbanLevel::Ghosted    => "ghosted",
        }
    }
}

// ─── Signaux de contenu poubelle (par tweet) ──────────────────────────────────

/// Signaux extraits d'un tweet individuel.
#[derive(Debug, Clone, Default)]
pub struct GarbageSignals {
    pub spam_hashtag_density: bool,  // hashtags > 30% des mots
    pub spam_mentions: bool,         // > 5 @mentions
    pub zero_engagement: bool,       // > 200 vues, zéro réaction
    pub high_report_rate: bool,      // rapport/vues > 3%
    pub pure_link_spam: bool,        // ≥ 2 URLs + texte < 50 chars
    pub emoji_overload: bool,        // > 8 émojis + texte < 100 chars
    pub repeat_content: bool,        // < 25% de mots uniques (texte dupliqué)
}

impl GarbageSignals {
    /// Score de poubelle 0–1 pondéré par la gravité du signal.
    pub fn score(&self) -> f64 {
        let mut s = 0.0_f64;
        if self.high_report_rate     { s += 0.35; } // signal le plus fort
        if self.zero_engagement      { s += 0.25; }
        if self.pure_link_spam       { s += 0.15; }
        if self.spam_hashtag_density { s += 0.12; }
        if self.repeat_content       { s += 0.10; }
        if self.spam_mentions        { s += 0.08; }
        if self.emoji_overload       { s += 0.05; }
        s.clamp(0.0, 1.0)
    }

    pub fn is_garbage(&self) -> bool { self.score() >= 0.40 }

    /// Résumé des signaux actifs pour les logs.
    pub fn active_signals(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.high_report_rate     { v.push("high_report_rate"); }
        if self.zero_engagement      { v.push("zero_engagement"); }
        if self.pure_link_spam       { v.push("pure_link_spam"); }
        if self.spam_hashtag_density { v.push("spam_hashtag_density"); }
        if self.repeat_content       { v.push("repeat_content"); }
        if self.spam_mentions        { v.push("spam_mentions"); }
        if self.emoji_overload       { v.push("emoji_overload"); }
        v
    }
}

// ─── Score de qualité du compte (agrégé sur 7 jours) ─────────────────────────

/// Score calculé périodiquement (ex: toutes les heures via job background).
/// Stocké en Redis + PostgreSQL, chargé avec le profil candidat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountQualityScore {
    pub user_id: String,
    pub level: ShadowbanLevel,

    pub garbage_ratio_7d: f64,    // % des posts des 7 derniers jours = poubelle
    pub avg_report_rate_7d: f64,  // taux moyen de signalements sur 7 jours
    pub spam_strikes: u32,        // posts garbage consécutifs les plus récents
    pub engagement_deficit: f64,  // écart vs baseline plateforme (négatif = mauvais)

    pub computed_at: DateTime<Utc>,
    pub manual_override: Option<ShadowbanLevel>, // surcharge équipe modération
}

impl AccountQualityScore {
    /// Niveau effectif : la surcharge manuelle prime toujours sur le calculé.
    pub fn effective_level(&self) -> ShadowbanLevel {
        self.manual_override.unwrap_or(self.level)
    }

    /// Compte propre avec score par défaut.
    pub fn clean(user_id: String) -> Self {
        Self {
            user_id,
            level: ShadowbanLevel::Clean,
            garbage_ratio_7d: 0.0,
            avg_report_rate_7d: 0.0,
            spam_strikes: 0,
            engagement_deficit: 0.0,
            computed_at: Utc::now(),
            manual_override: None,
        }
    }
}
