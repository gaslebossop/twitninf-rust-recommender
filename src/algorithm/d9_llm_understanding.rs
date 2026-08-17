/// D9 — LLM Content Understanding
///
/// Les huit premières dimensions déduisent la valeur d'un tweet de son
/// engagement et de son contexte social. Sur un corpus jeune, ces signaux sont
/// creux : un bon tweet d'un compte sans audience reste invisible faute de
/// likes, et le spam bien formaté passe pour du contenu.
///
/// D9 juge le texte lui-même, à partir des étiquettes produites hors-ligne par
/// l'annotateur LLM (`tweet_llm_labels`) : qualité réelle, ton, thème. Elle est
/// donc la seule dimension qui n'a besoin d'aucun engagement préalable pour se
/// prononcer — c'est ce qui la rend utile au démarrage à froid.
///
/// La toxicité n'entre PAS dans ce score : elle agit en multiplicateur de
/// pénalité séparé (voir `toxicity_penalty`), pour qu'un contenu agressif soit
/// rétrogradé quelle que soit sa qualité rédactionnelle par ailleurs.
use crate::models::{LlmLabels, RawTweet};

/// Score neutre appliqué aux tweets non annotés : ni bonus ni malus.
const NEUTRAL: f64 = 0.5;

/// En dessous de cette confiance, l'annotation est ramenée vers le neutre
/// proportionnellement — une étiquette hésitante ne doit pas trancher le
/// classement.
const CONFIDENCE_FLOOR: f64 = 0.35;

/// Plancher du multiplicateur de toxicité. Un tweet toxique est fortement
/// rétrogradé mais jamais annulé : le retrait est une décision de modération,
/// pas de scoring.
const MIN_TOXICITY_MULTIPLIER: f64 = 0.15;

/// Seuil de toxicité à partir duquel le tweet part en file de modération.
pub const MODERATION_QUEUE_THRESHOLD: f64 = 0.60;

/// Qualité en dessous de laquelle un tweet n'est plus recommandé du tout.
///
/// L'annotateur travaille en [0,1] (`clamp01` côté worker) : 0.10 correspond au
/// « 10 » de l'échelle affichée dans le panel. Contrairement à la toxicité, qui
/// se contente de rétrograder, un contenu sans aucun apport n'a pas de place
/// dans un fil de recommandation — le rétrograder le ferait juste remonter les
/// jours creux.
pub const MIN_QUALITY: f64 = 0.10;

/// Qualité à partir de laquelle le tweet est jugé « bon » et mérite un appui.
const QUALITY_BOOST_FLOOR: f64 = 0.70;

/// Plafond du coup de pouce de qualité. Volontairement petit : D9 récompense
/// déjà la qualité dans le score global, ce multiplicateur ne fait que départager
/// deux tweets par ailleurs comparables. Au-delà, il écraserait la fraîcheur et
/// le graphe social, et le fil se figerait sur un même petit lot de tweets.
const MAX_QUALITY_BOOST: f64 = 0.06;

/// Le tweet doit-il être écarté des recommandations pour qualité insuffisante ?
///
/// Deux garde-fous, parce que l'exclusion est totale et invisible :
///
/// - un tweet NON annoté n'est jamais écarté. L'annotateur passe en différé :
///   exclure sur une étiquette absente viderait le fil de tout ce qui vient
///   d'être publié, ce qui est exactement l'inverse du but.
/// - une annotation hésitante non plus. Sous `CONFIDENCE_FLOOR`, D9 considère
///   déjà l'étiquette comme sans valeur et la ramène au neutre ; elle ne peut
///   pas dans le même temps fonder un retrait définitif.
///
/// Comparaison stricte : une qualité pile au plancher reste servable.
pub fn below_quality_floor(tweet: &RawTweet) -> bool {
    let Some(labels) = tweet.llm.as_ref() else {
        return false;
    };
    labels.confidence.clamp(0.0, 1.0) >= CONFIDENCE_FLOOR
        && labels.quality_score.clamp(0.0, 1.0) < MIN_QUALITY
}

/// Multiplicateur d'appui pour les tweets de bonne qualité, dans
/// [1.0, 1.0 + MAX_QUALITY_BOOST].
///
/// Séparé de D9 pour la même raison que `toxicity_penalty` : mettre le poids de
/// D9 à zéro dans le panel admin ne doit pas emporter au passage la préférence
/// pour le contenu qui apporte quelque chose.
///
/// La montée est progressive à partir de `QUALITY_BOOST_FLOOR` — un seuil sec
/// créerait une falaise où deux tweets à 0.69 et 0.71 seraient traités très
/// différemment — et amortie par la confiance, comme partout ailleurs ici.
pub fn quality_boost(tweet: &RawTweet) -> f64 {
    let Some(labels) = tweet.llm.as_ref() else {
        return 1.0;
    };
    let quality = labels.quality_score.clamp(0.0, 1.0);
    if quality <= QUALITY_BOOST_FLOOR {
        return 1.0;
    }
    let c = labels.confidence.clamp(0.0, 1.0);
    let weight = ((c - CONFIDENCE_FLOOR) / (1.0 - CONFIDENCE_FLOOR)).clamp(0.0, 1.0);
    let ramp = (quality - QUALITY_BOOST_FLOOR) / (1.0 - QUALITY_BOOST_FLOOR);
    1.0 + MAX_QUALITY_BOOST * ramp * weight
}

/// Score D9 dans [0,1].
pub fn d9_llm_understanding(tweet: &RawTweet) -> f64 {
    let Some(labels) = tweet.llm.as_ref() else {
        return NEUTRAL;
    };

    // La qualité porte l'essentiel du signal : c'est elle qui distingue un
    // message qui apporte quelque chose d'un remplissage bien orthographié.
    let quality = labels.quality_score.clamp(0.0, 1.0);

    // Le ton module à la marge. On ne récompense pas la positivité en soi —
    // l'ironie et la critique font partie d'un fil vivant — mais l'agressivité
    // gratuite dégrade l'expérience de lecture.
    let tone_adj = match labels.tone.as_str() {
        "enthousiaste" | "positif" => 0.05,
        "ironique" | "neutre" => 0.0,
        "negatif" => -0.05,
        "agressif" => -0.15,
        _ => 0.0,
    };

    // Le thème sert surtout à la diversité (D6). Ici on se contente de retenir
    // les catégories qui n'apportent structurellement rien au lecteur.
    let theme_adj = match labels.theme.as_str() {
        "spam_vide" => -0.20,
        "clash_insulte" => -0.10,
        _ => 0.0,
    };

    let raw = (quality + tone_adj + theme_adj).clamp(0.0, 1.0);

    // Amortissement par la confiance : une annotation peu sûre glisse vers le
    // neutre au lieu de s'imposer.
    let c = labels.confidence.clamp(0.0, 1.0);
    let weight = ((c - CONFIDENCE_FLOOR) / (1.0 - CONFIDENCE_FLOOR)).clamp(0.0, 1.0);

    NEUTRAL + (raw - NEUTRAL) * weight
}

/// Multiplicateur de pénalité pour toxicité, dans [MIN_TOXICITY_MULTIPLIER, 1.0].
///
/// Séparé du score D9 pour rester lisible et débranchable : mettre le poids de
/// D9 à zéro dans le panel admin ne doit pas désactiver au passage la
/// protection contre le contenu agressif.
pub fn toxicity_penalty(tweet: &RawTweet) -> f64 {
    let Some(labels) = tweet.llm.as_ref() else {
        return 1.0;
    };
    let tox = labels.toxicity_score.clamp(0.0, 1.0);
    let c = labels.confidence.clamp(0.0, 1.0);

    // Une toxicité incertaine pénalise moins : on multiplie l'intensité par la
    // confiance plutôt que d'appliquer un seuil binaire.
    let effective = tox * c;
    (1.0 - effective).max(MIN_TOXICITY_MULTIPLIER)
}

/// Le tweet doit-il être soumis à une revue humaine ?
///
/// Volontairement distinct de la pénalité de score : le feed rétrograde tout
/// seul, mais aucun contenu n'est retiré sans qu'un humain ait tranché.
pub fn needs_moderation_review(labels: &LlmLabels) -> bool {
    labels.toxicity_score >= MODERATION_QUEUE_THRESHOLD && labels.confidence >= 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(quality: f64, tox: f64, tone: &str, theme: &str, conf: f64) -> LlmLabels {
        LlmLabels {
            theme: theme.to_string(),
            toxicity_score: tox,
            toxicity_category: "aucune".to_string(),
            quality_score: quality,
            tone: tone.to_string(),
            confidence: conf,
        }
    }

    fn tweet_with(l: Option<LlmLabels>) -> RawTweet {
        let mut t = crate::models::RawTweet::default();
        t.llm = l;
        t
    }

    #[test]
    fn tweet_non_annote_reste_neutre() {
        assert_eq!(d9_llm_understanding(&tweet_with(None)), NEUTRAL);
        assert_eq!(toxicity_penalty(&tweet_with(None)), 1.0);
    }

    #[test]
    fn qualite_haute_bat_qualite_basse() {
        let bon =
            d9_llm_understanding(&tweet_with(Some(labels(0.9, 0.0, "neutre", "humour", 1.0))));
        let mauvais =
            d9_llm_understanding(&tweet_with(Some(labels(0.1, 0.0, "neutre", "humour", 1.0))));
        assert!(bon > mauvais, "bon={bon} mauvais={mauvais}");
    }

    #[test]
    fn confiance_faible_ramene_vers_neutre() {
        let sur =
            d9_llm_understanding(&tweet_with(Some(labels(1.0, 0.0, "neutre", "humour", 1.0))));
        let hesitant =
            d9_llm_understanding(&tweet_with(Some(labels(1.0, 0.0, "neutre", "humour", 0.4))));
        assert!(
            (hesitant - NEUTRAL).abs() < (sur - NEUTRAL).abs(),
            "hesitant={hesitant} sur={sur}"
        );
    }

    #[test]
    fn toxicite_penalise_sans_annuler() {
        let p = toxicity_penalty(&tweet_with(Some(labels(
            0.5,
            1.0,
            "agressif",
            "clash_insulte",
            1.0,
        ))));
        assert!(p >= MIN_TOXICITY_MULTIPLIER, "penalite={p}");
        assert!(
            p < 0.5,
            "un tweet tres toxique doit etre fortement retrograde, penalite={p}"
        );
    }

    #[test]
    fn qualite_sous_le_plancher_est_ecartee() {
        let nul = tweet_with(Some(labels(0.05, 0.0, "neutre", "humour", 0.9)));
        assert!(below_quality_floor(&nul));
    }

    #[test]
    fn plancher_inclusif_et_qualite_correcte_conservee() {
        // Pile au plancher : servable (comparaison stricte).
        let pile = tweet_with(Some(labels(MIN_QUALITY, 0.0, "neutre", "humour", 0.9)));
        assert!(!below_quality_floor(&pile));
        let bon = tweet_with(Some(labels(0.4, 0.0, "neutre", "humour", 0.9)));
        assert!(!below_quality_floor(&bon));
    }

    #[test]
    fn tweet_non_annote_nest_jamais_ecarte() {
        // Garde-fou central : l'annotateur passe en différé, exclure sur une
        // étiquette absente viderait le fil de tout ce qui vient d'être publié.
        assert!(!below_quality_floor(&tweet_with(None)));
        assert_eq!(quality_boost(&tweet_with(None)), 1.0);
    }

    #[test]
    fn annotation_hesitante_nexclut_pas() {
        // Même qualité nulle, mais confiance sous le plancher : D9 la ramène
        // déjà au neutre, elle ne peut pas fonder un retrait définitif.
        let hesitant = tweet_with(Some(labels(0.01, 0.0, "neutre", "humour", 0.2)));
        assert!(!below_quality_floor(&hesitant));
    }

    #[test]
    fn bonne_qualite_recoit_un_petit_boost() {
        let excellent = quality_boost(&tweet_with(Some(labels(1.0, 0.0, "neutre", "humour", 1.0))));
        assert!(excellent > 1.0, "boost={excellent}");
        assert!(
            excellent <= 1.0 + MAX_QUALITY_BOOST + f64::EPSILON,
            "le boost doit rester petit, boost={excellent}"
        );

        // Sous le seuil de « bonne qualité », aucun appui.
        let moyen = quality_boost(&tweet_with(Some(labels(0.5, 0.0, "neutre", "humour", 1.0))));
        assert_eq!(moyen, 1.0);
    }

    #[test]
    fn le_boost_monte_progressivement() {
        // Pas de falaise autour du seuil : deux qualités voisines restent
        // traitées de façon voisine.
        let juste_au_dessus = quality_boost(&tweet_with(Some(labels(
            0.71, 0.0, "neutre", "humour", 1.0,
        ))));
        let bien_au_dessus = quality_boost(&tweet_with(Some(labels(
            0.95, 0.0, "neutre", "humour", 1.0,
        ))));
        assert!(juste_au_dessus > 1.0);
        assert!(bien_au_dessus > juste_au_dessus);
        assert!(
            juste_au_dessus < 1.0 + MAX_QUALITY_BOOST * 0.2,
            "montée trop brutale: {juste_au_dessus}"
        );
    }

    #[test]
    fn boost_amorti_par_la_confiance() {
        let sur = quality_boost(&tweet_with(Some(labels(1.0, 0.0, "neutre", "humour", 1.0))));
        let hesitant = quality_boost(&tweet_with(Some(labels(1.0, 0.0, "neutre", "humour", 0.4))));
        assert!(hesitant < sur, "hesitant={hesitant} sur={sur}");
    }

    #[test]
    fn score_toujours_dans_l_intervalle() {
        for q in [0.0, 0.5, 1.0] {
            for tone in ["agressif", "positif", "neutre"] {
                for theme in ["spam_vide", "humour"] {
                    let s =
                        d9_llm_understanding(&tweet_with(Some(labels(q, 0.0, tone, theme, 1.0))));
                    assert!((0.0..=1.0).contains(&s), "score hors intervalle: {s}");
                }
            }
        }
    }
}
