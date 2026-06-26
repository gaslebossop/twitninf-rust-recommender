use tracing::debug;

use crate::models::{RawTweet, RecommendMode, TweetSource};

use super::models::ShadowbanLevel;

// ─── ShadowbanEnforcer ────────────────────────────────────────────────────────

/// Applique les effets du shadowban au scoring et à la composition du feed.
///
/// Deux points d'intégration dans le pipeline :
/// 1. `apply_to_score()` — appelé dans `score_tweet()` comme Mod D, après Mod C (source)
/// 2. `should_include()` — filtre dur avant insertion dans le feed (notamment Ghosted)
pub struct ShadowbanEnforcer;

impl ShadowbanEnforcer {
    pub fn new() -> Self { Self }

    /// Mod D — Applique le multiplicateur shadowban au score final.
    /// Le tweet reste dans le pipeline pour les comptes Monitoring/Suppressed ;
    /// seul Ghosted approche zéro (mais n'est pas filtré ici — voir `should_include`).
    pub fn apply_to_score(&self, score: f64, level: ShadowbanLevel) -> f64 {
        let mult = level.score_multiplier();
        let adjusted = (score * mult).clamp(0.0, 1.0);
        if level != ShadowbanLevel::Clean {
            debug!(
                original_score = score,
                multiplier = mult,
                adjusted_score = adjusted,
                level = level.label(),
                "Mod D: shadowban multiplier applied"
            );
        }
        adjusted
    }

    /// Filtre dur : détermine si le tweet doit apparaître dans ce contexte de feed.
    ///
    /// - `Ghosted`    → uniquement si l'utilisateur connecté suit l'auteur
    /// - `Suppressed` → exclu de Discovery et Trending (sources de découverte)
    /// - `Monitoring` / `Clean` → toujours inclus
    pub fn should_include(
        &self,
        tweet: &RawTweet,
        level: ShadowbanLevel,
        mode: &RecommendMode,
        viewer_follows_author: bool,
    ) -> bool {
        match level {
            ShadowbanLevel::Clean | ShadowbanLevel::Monitoring => true,

            ShadowbanLevel::Suppressed => {
                // Pas dans les flux de découverte publics
                let excluded_source = matches!(
                    tweet.source,
                    TweetSource::Discovery | TweetSource::Trending
                );
                let excluded_mode = matches!(mode, RecommendMode::Trending | RecommendMode::Discover);
                !(excluded_source || excluded_mode)
            }

            ShadowbanLevel::Ghosted => {
                // Visible seulement pour les abonnés directs, jamais en mode découverte
                viewer_follows_author
                    && !matches!(mode, RecommendMode::Trending | RecommendMode::Discover)
            }
        }
    }

    /// Retourne le multiplicateur pour intégration dans `ScoreBreakdown`.
    pub fn multiplier(&self, level: ShadowbanLevel) -> f64 {
        level.score_multiplier()
    }
}

impl Default for ShadowbanEnforcer {
    fn default() -> Self { Self::new() }
}
