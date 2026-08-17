use tracing::debug;

use crate::models::{RawTweet, RecommendMode, TweetSource};

use super::models::{ContentEligibility, ShadowbanLevel, Surface};

// ─── ShadowbanEnforcer ────────────────────────────────────────────────────────

/// Verdict d'admission d'un tweet sur une surface donnée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    /// Faux → le tweet ne doit pas apparaître sur cette surface.
    pub allowed: bool,
    /// Multiplicateur de score à appliquer quand il est admis.
    pub multiplier: f64,
    /// Ce qui a fermé la porte, pour les logs et l'état de compte.
    pub blocked_by: Option<&'static str>,
}

impl Verdict {
    fn allow(multiplier: f64) -> Self {
        Verdict {
            allowed: true,
            multiplier,
            blocked_by: None,
        }
    }
    fn block(reason: &'static str) -> Self {
        Verdict {
            allowed: false,
            multiplier: 0.0,
            blocked_by: Some(reason),
        }
    }
}

/// Applique les restrictions au scoring et à la composition du feed.
///
/// Deux points d'intégration dans le pipeline :
/// 1. `apply_to_score()` — Mod D, dans `score_tweet()`, après Mod C (source)
/// 2. `admit()` — filtre dur, surface par surface, avant insertion dans le feed
///
/// La séparation compte pour de bon : un multiplicateur, même à 0,05, reste
/// franchissable par un contenu à très forte vélocité, et c'est exactement ce
/// qui se passait — `should_include()` existait mais **n'était appelé nulle
/// part**, donc `excludes_trending()` et `excludes_discovery()` ne fermaient
/// rien. Un compte `Ghosted` était rétrogradé, jamais exclu.
pub struct ShadowbanEnforcer;

impl ShadowbanEnforcer {
    pub fn new() -> Self {
        Self
    }

    /// Mod D — Applique le multiplicateur de compte au score final.
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

    /// Filtre dur — le tweet a-t-il sa place sur cette surface ?
    ///
    /// Un invariant traverse toute la fonction : **un lecteur ne perd jamais un
    /// compte qu'il suit.** Le fil d'abonnement n'est fermé à aucun niveau, et un
    /// auteur suivi n'est jamais retiré de « Pour toi ». Retirer en silence à
    /// quelqu'un un compte qu'il a explicitement demandé, c'est précisément le
    /// reproche que le mot « shadowban » désigne — et ça n'apporte rien : le
    /// lecteur ira le chercher sur le profil, où le contenu est de toute façon
    /// resté.
    ///
    /// Ce qui est fermé, ce sont les surfaces où l'on **pousse** du contenu vers
    /// des gens qui n'ont rien demandé.
    pub fn admit(
        &self,
        level: ShadowbanLevel,
        eligibility: ContentEligibility,
        surface: Surface,
        viewer_follows_author: bool,
    ) -> Verdict {
        // Le fil d'abonnement n'est jamais filtré, à aucun niveau.
        if surface == Surface::FollowerFeed {
            return Verdict::allow(level.score_multiplier());
        }

        // « Pour toi » mélange comptes suivis et découverte : seule la moitié
        // découverte est concernée. Sans cette exception, un compte restreint
        // disparaîtrait pour ses propres abonnés, puisque l'application sert son
        // fil principal en mode `for_you` et non en mode `feed`.
        let is_discovery = !viewer_follows_author || surface != Surface::ForYou;

        if is_discovery {
            if level.restricts(surface) {
                return Verdict::block(match level {
                    ShadowbanLevel::Ghosted => "account_ghosted",
                    _ => "account_suppressed",
                });
            }
            if let ContentEligibility::NotRecommended(reason) = eligibility {
                return Verdict::block(reason.label());
            }
        }

        Verdict::allow(level.score_multiplier())
    }

    /// Surface effective d'un tweet : la source qui l'a fait entrer dans le pool
    /// peut être plus « poussée » que le mode demandé.
    ///
    /// Un tweet remonté par `Trending` à l'intérieur d'un fil « Pour toi » reste
    /// une mise en avant : le fermer au mode seul laisserait passer par la porte
    /// de derrière ce qu'on ferme par la grande.
    ///
    /// Sauf pour un compte suivi : quelle que soit la source qui l'a récupéré,
    /// le lecteur l'a demandé, donc ce n'est pas une mise en avant. Sans cette
    /// exception l'invariant d'`admit()` fuyait — il suffisait que la
    /// déduplication retienne la source `Trending` pour qu'un abonné perde un
    /// compte restreint dans son fil principal.
    pub fn effective_surface(
        tweet: &RawTweet,
        mode: &RecommendMode,
        viewer_follows_author: bool,
    ) -> Surface {
        let mode_surface = Surface::from_mode(mode);
        if viewer_follows_author || mode_surface == Surface::FollowerFeed {
            return mode_surface;
        }
        match tweet.source {
            TweetSource::Trending => Surface::Trending,
            TweetSource::Discovery => Surface::Discover,
            _ => mode_surface,
        }
    }

    /// Retourne le multiplicateur pour intégration dans `ScoreBreakdown`.
    pub fn multiplier(&self, level: ShadowbanLevel) -> f64 {
        level.score_multiplier()
    }
}

impl Default for ShadowbanEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadowban::models::IneligibilityReason;

    const OK: ContentEligibility = ContentEligibility::Eligible;
    const SPAM: ContentEligibility =
        ContentEligibility::NotRecommended(IneligibilityReason::SpamSignals);

    fn e() -> ShadowbanEnforcer {
        ShadowbanEnforcer::new()
    }

    #[test]
    fn compte_propre_passe_partout() {
        for surface in [
            Surface::FollowerFeed,
            Surface::ForYou,
            Surface::Discover,
            Surface::Trending,
        ] {
            assert!(e().admit(ShadowbanLevel::Clean, OK, surface, false).allowed);
        }
    }

    #[test]
    fn surveillance_ne_ferme_aucune_surface() {
        for surface in [
            Surface::FollowerFeed,
            Surface::ForYou,
            Surface::Discover,
            Surface::Trending,
        ] {
            assert!(
                e().admit(ShadowbanLevel::Monitoring, OK, surface, false)
                    .allowed,
                "Monitoring doit rester un palier d'alerte, pas une sanction ({surface:?})"
            );
        }
    }

    #[test]
    fn suppressed_ferme_la_decouverte_pas_pour_toi() {
        let v = e().admit(ShadowbanLevel::Suppressed, OK, Surface::Trending, false);
        assert!(!v.allowed);
        assert_eq!(v.blocked_by, Some("account_suppressed"));
        assert!(
            !e().admit(ShadowbanLevel::Suppressed, OK, Surface::Discover, false)
                .allowed
        );
        // Pour toi reste ouvert : la sanction est graduée, pas binaire.
        assert!(
            e().admit(ShadowbanLevel::Suppressed, OK, Surface::ForYou, false)
                .allowed
        );
    }

    #[test]
    fn ghosted_ferme_toute_recommandation() {
        for surface in [Surface::ForYou, Surface::Discover, Surface::Trending] {
            let v = e().admit(ShadowbanLevel::Ghosted, OK, surface, false);
            assert!(!v.allowed, "{surface:?} doit être fermée");
            assert_eq!(v.blocked_by, Some("account_ghosted"));
        }
    }

    #[test]
    fn un_abonne_ne_perd_jamais_le_compte_qu_il_suit() {
        // Ni via le fil d'abonnement…
        assert!(
            e().admit(ShadowbanLevel::Ghosted, SPAM, Surface::FollowerFeed, true)
                .allowed
        );
        assert!(
            e().admit(ShadowbanLevel::Ghosted, SPAM, Surface::FollowerFeed, false)
                .allowed
        );
        // …ni via « Pour toi », que l'application utilise comme fil principal.
        assert!(
            e().admit(ShadowbanLevel::Ghosted, SPAM, Surface::ForYou, true)
                .allowed
        );
        // Mais la mise en avant publique reste fermée, même pour un abonné.
        assert!(
            !e().admit(ShadowbanLevel::Ghosted, OK, Surface::Trending, true)
                .allowed
        );
    }

    #[test]
    fn contenu_non_recommande_sort_de_la_decouverte_seulement() {
        assert!(
            !e().admit(ShadowbanLevel::Clean, SPAM, Surface::Trending, false)
                .allowed
        );
        assert!(
            !e().admit(ShadowbanLevel::Clean, SPAM, Surface::Discover, false)
                .allowed
        );
        assert!(
            !e().admit(ShadowbanLevel::Clean, SPAM, Surface::ForYou, false)
                .allowed
        );
        // Le post reste en ligne pour qui a demandé ce compte.
        assert!(
            e().admit(ShadowbanLevel::Clean, SPAM, Surface::ForYou, true)
                .allowed
        );
        assert!(
            e().admit(ShadowbanLevel::Clean, SPAM, Surface::FollowerFeed, false)
                .allowed
        );
    }

    #[test]
    fn le_motif_du_blocage_est_nomme() {
        let v = e().admit(ShadowbanLevel::Clean, SPAM, Surface::Trending, false);
        assert_eq!(v.blocked_by, Some("spam_signals"));
    }

    #[test]
    fn une_source_tendance_reste_une_mise_en_avant_dans_pour_toi() {
        let promu = RawTweet {
            source: TweetSource::Trending,
            ..Default::default()
        };
        let normal = RawTweet {
            source: TweetSource::SocialGraph,
            ..Default::default()
        };

        assert_eq!(
            ShadowbanEnforcer::effective_surface(&promu, &RecommendMode::ForYou, false),
            Surface::Trending
        );
        assert_eq!(
            ShadowbanEnforcer::effective_surface(&normal, &RecommendMode::ForYou, false),
            Surface::ForYou
        );
        // Le fil d'abonnement ne se laisse pas requalifier par la source.
        assert_eq!(
            ShadowbanEnforcer::effective_surface(&promu, &RecommendMode::Feed, false),
            Surface::FollowerFeed
        );
    }

    #[test]
    fn la_source_ne_requalifie_pas_un_compte_suivi() {
        // Le piège : la déduplication garde la source de plus fort poids. Si
        // c'est `Trending`, un abonné se retrouvait à consulter une surface
        // « mise en avant » pour un compte qu'il suit — et le perdait.
        let promu = RawTweet {
            source: TweetSource::Trending,
            ..Default::default()
        };
        let surface = ShadowbanEnforcer::effective_surface(&promu, &RecommendMode::ForYou, true);
        assert_eq!(surface, Surface::ForYou);
        assert!(
            e().admit(ShadowbanLevel::Ghosted, SPAM, surface, true)
                .allowed
        );
    }

    #[test]
    fn multiplicateur_inchange_pour_le_scoring() {
        assert_eq!(e().multiplier(ShadowbanLevel::Clean), 1.00);
        assert_eq!(e().multiplier(ShadowbanLevel::Monitoring), 0.85);
        assert_eq!(e().multiplier(ShadowbanLevel::Suppressed), 0.45);
        assert_eq!(e().multiplier(ShadowbanLevel::Ghosted), 0.05);
        assert!((e().apply_to_score(0.8, ShadowbanLevel::Suppressed) - 0.36).abs() < 1e-9);
    }
}
