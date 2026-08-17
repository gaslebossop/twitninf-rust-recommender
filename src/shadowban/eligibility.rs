//! Éligibilité à la recommandation, au niveau du **contenu**.
//!
//! Un post peut cesser d'être recommandé sans que son auteur soit sanctionné, et
//! un compte peut être restreint sans que ce post précis pose problème. Les deux
//! plans sont séparés — c'est la distinction que faisait défaut ici.
//!
//! Avant : un seul verdict, `GARBAGE_HARD_DROP` à 0,60, qui retirait le tweet de
//! **tout** le pipeline. Deux défauts en découlaient :
//!
//! - au-dessus du seuil, le tweet disparaissait aussi du fil des abonnés — des
//!   lecteurs qui avaient explicitement demandé ce compte ne le voyaient plus,
//!   sans que rien ne l'indique nulle part ;
//! - en dessous, il gardait un accès plein aux Tendances avec une simple
//!   rétrogradation multiplicative, franchissable par n'importe quel pic
//!   d'engagement.
//!
//! Maintenant : sous le seuil, distribution normale ; au-dessus, le post reste en
//! ligne, sur le profil et dans le fil des abonnés, mais n'entre plus dans les
//! surfaces de recommandation — avec un motif nommé, affichable à l'auteur.

use tracing::trace;

use crate::models::RawTweet;

use super::models::{ContentEligibility, GarbageSignals, IneligibilityReason};

/// Score de signaux à partir duquel le post quitte les surfaces de découverte.
///
/// Aligné sur `GarbageSignals::is_garbage()` (0,40), et volontairement plus bas
/// que l'ancien retrait total à 0,60 : couper la recommandation est réversible et
/// sans perte pour le lecteur, retirer un post du fil de ses abonnés ne l'est pas.
/// On peut donc être plus strict côté découverte parce qu'on est moins brutal
/// côté abonnés.
pub const NOT_RECOMMENDED_SCORE: f64 = 0.40;

/// Toxicité à partir de laquelle un post n'est plus recommandé.
/// En dessous, `toxicity_penalty` (Mod H) suffit à le rétrograder.
const TOXIC_NOT_RECOMMENDED: f64 = 0.60;

/// Confiance minimale d'annotation pour agir sur une étiquette LLM.
/// Sous ce seuil, l'annotation ne fonde aucune décision binaire.
const LABEL_MIN_CONFIDENCE: f64 = 0.60;

/// Plancher de qualité, appliqué seulement à une annotation confiante.
const QUALITY_NOT_RECOMMENDED: f64 = 0.12;

/// Nombre de vues à partir duquel un post sans la moindre réaction devient un
/// signal négatif exploitable. En dessous, l'absence de réaction ne dit rien.
const REPORT_REVIEW_MIN_VIEWS: i64 = 100;

/// Verdict d'éligibilité d'un tweet, signaux de spam déjà calculés.
///
/// `signals` est passé en paramètre plutôt que recalculé : le détecteur tourne
/// déjà une fois par tweet dans le scoring, et le refaire ici doublerait le coût
/// sur tout le pool de candidats.
pub fn content_eligibility(tweet: &RawTweet, signals: &GarbageSignals) -> ContentEligibility {
    // 1. Signalements — c'est le seul motif qui vient d'humains, pas d'heuristiques.
    if tweet.view_count >= REPORT_REVIEW_MIN_VIEWS && signals.high_report_rate {
        return not_recommended(tweet, IneligibilityReason::UnderReview);
    }

    // 2. Toxicité établie par l'annotateur, avec une confiance suffisante.
    if let Some(labels) = tweet.llm.as_ref() {
        if labels.confidence >= LABEL_MIN_CONFIDENCE {
            if labels.toxicity_score >= TOXIC_NOT_RECOMMENDED {
                return not_recommended(tweet, IneligibilityReason::Toxic);
            }
            if labels.quality_score <= QUALITY_NOT_RECOMMENDED {
                return not_recommended(tweet, IneligibilityReason::LowQuality);
            }
        }
    }

    // 3. Faisceau de signaux de spam.
    if signals.score() >= NOT_RECOMMENDED_SCORE {
        let reason = signals
            .dominant_reason()
            .unwrap_or(IneligibilityReason::SpamSignals);
        return not_recommended(tweet, reason);
    }

    ContentEligibility::Eligible
}

fn not_recommended(tweet: &RawTweet, reason: IneligibilityReason) -> ContentEligibility {
    trace!(
        tweet_id = %tweet.id,
        reason = reason.label(),
        "Contenu retiré des surfaces de recommandation (reste visible des abonnés)"
    );
    ContentEligibility::NotRecommended(reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LlmLabels;

    fn labels(quality: f64, toxicity: f64, confidence: f64) -> LlmLabels {
        LlmLabels {
            theme: "general".into(),
            toxicity_score: toxicity,
            toxicity_category: "aucune".into(),
            quality_score: quality,
            tone: "neutre".into(),
            confidence,
        }
    }

    fn tweet() -> RawTweet {
        RawTweet {
            id: "t1".into(),
            view_count: 500,
            ..Default::default()
        }
    }

    #[test]
    fn tweet_ordinaire_reste_eligible() {
        let t = tweet();
        let s = GarbageSignals::default();
        assert_eq!(content_eligibility(&t, &s), ContentEligibility::Eligible);
    }

    #[test]
    fn signalements_mettent_en_verification() {
        let t = tweet();
        let s = GarbageSignals {
            high_report_rate: true,
            ..Default::default()
        };
        assert_eq!(
            content_eligibility(&t, &s),
            ContentEligibility::NotRecommended(IneligibilityReason::UnderReview)
        );
    }

    #[test]
    fn signalements_ignores_sous_le_plancher_de_vues() {
        // 12 vues et un signalement ne prouvent rien.
        let t = RawTweet {
            view_count: 12,
            ..tweet()
        };
        let s = GarbageSignals {
            high_report_rate: true,
            ..Default::default()
        };
        // Le score des signaux (0,35) reste sous le seuil de spam : éligible.
        assert_eq!(content_eligibility(&t, &s), ContentEligibility::Eligible);
    }

    #[test]
    fn toxicite_confiante_coupe_la_recommandation() {
        let t = RawTweet {
            llm: Some(labels(0.5, 0.75, 0.9)),
            ..tweet()
        };
        assert_eq!(
            content_eligibility(&t, &GarbageSignals::default()),
            ContentEligibility::NotRecommended(IneligibilityReason::Toxic)
        );
    }

    #[test]
    fn toxicite_incertaine_ne_coupe_rien() {
        // Même toxicité, annotation peu sûre : on rétrograde (Mod H) mais on ne
        // ferme aucune surface sur une supposition.
        let t = RawTweet {
            llm: Some(labels(0.5, 0.75, 0.3)),
            ..tweet()
        };
        assert_eq!(
            content_eligibility(&t, &GarbageSignals::default()),
            ContentEligibility::Eligible
        );
    }

    #[test]
    fn qualite_au_plancher_coupe_la_recommandation() {
        let t = RawTweet {
            llm: Some(labels(0.05, 0.0, 0.8)),
            ..tweet()
        };
        assert_eq!(
            content_eligibility(&t, &GarbageSignals::default()),
            ContentEligibility::NotRecommended(IneligibilityReason::LowQuality)
        );
    }

    #[test]
    fn spam_de_liens_est_nomme_precisement() {
        let t = tweet();
        // 0,15 + 0,25 = 0,40 → au seuil.
        let s = GarbageSignals {
            pure_link_spam: true,
            zero_engagement: true,
            ..Default::default()
        };
        assert_eq!(
            content_eligibility(&t, &s),
            ContentEligibility::NotRecommended(IneligibilityReason::LinkSpam)
        );
    }

    #[test]
    fn motif_d_inegibilite_se_rattache_au_bon_domaine() {
        use super::super::models::StrikePolicy;
        assert_eq!(
            IneligibilityReason::LinkSpam.policy(),
            Some(StrikePolicy::Spam)
        );
        assert_eq!(
            IneligibilityReason::Unoriginal.policy(),
            Some(StrikePolicy::Unoriginal)
        );
        assert_eq!(
            IneligibilityReason::Toxic.policy(),
            Some(StrikePolicy::Harassment)
        );
        // En vérification : rien n'est établi, donc rien ne se cumule.
        assert_eq!(IneligibilityReason::UnderReview.policy(), None);
        assert_eq!(IneligibilityReason::LowQuality.policy(), None);
    }

    #[test]
    fn la_categorie_de_toxicite_affine_le_domaine_du_strike() {
        use super::super::models::StrikePolicy;

        // Une menace ne doit pas se noyer dans le harcèlement générique : son
        // seuil de bannissement est bien plus court.
        assert_eq!(
            IneligibilityReason::Toxic.policy_for(Some("menace")),
            Some(StrikePolicy::ViolentThreat)
        );
        assert_eq!(
            IneligibilityReason::Toxic.policy_for(Some("haine_identitaire")),
            Some(StrikePolicy::HatefulConduct)
        );
        assert_eq!(
            IneligibilityReason::Toxic.policy_for(Some("contenu_sexuel")),
            Some(StrikePolicy::AdultContent)
        );
        // Catégorie générique ou absente : repli sur le domaine générique.
        assert_eq!(
            IneligibilityReason::Toxic.policy_for(Some("insulte_ciblee")),
            Some(StrikePolicy::Harassment)
        );
        assert_eq!(
            IneligibilityReason::Toxic.policy_for(None),
            Some(StrikePolicy::Harassment)
        );
        // Les motifs non toxiques ne changent pas de comportement : `policy_for`
        // ignore la catégorie et retombe sur `policy()`.
        assert_eq!(
            IneligibilityReason::LinkSpam.policy_for(Some("menace")),
            Some(StrikePolicy::Spam)
        );
    }
}
