use crate::algorithm::scoring::age_hours;
use crate::models::RawTweet;
use tracing::{debug, trace};

/// Score de tendance d'un tweet basé sur l'accélération d'engagement.
pub fn trending_score(t: &RawTweet) -> f64 {
    let age_h = age_hours(t).max(0.1);
    let total_eng = (t.like_count + t.comment_count + t.retweet_count + t.share_count) as f64;

    if total_eng == 0.0 {
        trace!(tweet_id = %t.id, "Trending: no engagement yet");
        return 0.0;
    }

    let eng_per_hour = total_eng / age_h;
    trace!(
        age_h,
        total_eng,
        eng_per_hour,
        "Trending: engagement velocity"
    );

    // Bonus pour tweets très récents avec déjà de l'engagement
    let recency_mult = if age_h < 1.0 {
        3.0
    } else if age_h < 3.0 {
        2.0
    } else if age_h < 6.0 {
        1.5
    } else {
        1.0
    };
    trace!(age_h, recency_mult, "Trending: recency multiplier");

    // Calibré : 50 eng/h ≈ score 0.7
    let raw = eng_per_hour * recency_mult;
    let score = 1.0 / (1.0 + (-raw / 30.0).exp());
    debug!(tweet_id = %t.id, score, eng_per_hour, recency_mult, "Trending score calculated");
    score
}

pub fn top_trending(tweets: &[RawTweet], top_n: usize) -> Vec<(String, f64)> {
    debug!(
        "Calculating trending scores for {} tweets, top {}",
        tweets.len(),
        top_n
    );
    let mut scores: Vec<(String, f64)> = tweets
        .iter()
        .map(|t| {
            let ts = trending_score(t);
            trace!(tweet_id = %t.id, trending_score = ts, "Trending score");
            (t.id.clone(), ts)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    debug!(
        "Top trending tweets: {:?}",
        scores
            .iter()
            .map(|(id, score)| (id.clone(), format!("{:.4}", score)))
            .collect::<Vec<_>>()
    );
    scores
}

/// Convertit des similarités de goût brutes en FACTEURS de renfort, pour le
/// seul mode Trending.
///
/// ── Pourquoi une normalisation relative au vivier ──────────────────────────
/// L'échelle absolue d'une similarité cosinus dépend du modèle d'embedding :
/// sur beaucoup de modèles de texte, deux phrases sans rapport tournent déjà
/// autour de 0,6. Un seuil écrit en dur (« au-dessus de 0,7, on renforce »)
/// serait donc à réécrire à chaque changement de modèle, et silencieusement
/// faux entre-temps. On se cale sur la MÉDIANE du vivier du moment : le renfort
/// va à ce qui est proche du goût du lecteur *comparé aux autres candidats de
/// cette page*, ce qui reste vrai quel que soit le modèle.
///
/// ── Pourquoi aucune pénalité ──────────────────────────────────────────────
/// Tout ce qui est sous la médiane reçoit `1,0`, jamais moins. Renforcer ce qui
/// ressemble aux goûts est une aide à la découverte ; déprécier le reste
/// refermerait la page sur le déjà-aimé, ce qu'une grille d'exploration ne doit
/// pas faire. Les tweets sans embedding, absents de l'entrée, sont dans le même
/// cas : ils ne sont pas mesurables, donc ils ne sont pas touchés.
///
/// `max_boost` est le facteur atteint par le candidat le PLUS proche du goût
/// (voir `TRENDING_TASTE_BOOST_MAX`).
pub fn taste_boost_factors(
    similarities: &std::collections::HashMap<String, f64>,
    max_boost: f64,
) -> std::collections::HashMap<String, f64> {
    if similarities.is_empty() {
        return std::collections::HashMap::new();
    }

    let mut sorted: Vec<f64> = similarities.values().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];

    // Vivier plat (une seule valeur, ou toutes identiques) : aucun candidat ne
    // se distingue, donc personne ne mérite un renfort. Garde aussi contre la
    // division par zéro juste en dessous.
    let span = max - median;
    if span <= f64::EPSILON {
        return std::collections::HashMap::new();
    }

    similarities
        .iter()
        .filter(|(_, sim)| **sim > median)
        .map(|(id, sim)| {
            let position = ((sim - median) / span).clamp(0.0, 1.0);
            (id.clone(), 1.0 + (max_boost - 1.0) * position)
        })
        .collect()
}

#[cfg(test)]
mod taste_tests {
    use super::taste_boost_factors;
    use std::collections::HashMap;

    fn sims(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs
            .iter()
            .map(|(id, s)| ((*id).to_string(), *s))
            .collect()
    }

    #[test]
    fn le_plus_proche_recoit_le_plafond() {
        let factors = taste_boost_factors(&sims(&[("a", 0.1), ("b", 0.5), ("c", 0.9)]), 1.18);
        assert!((factors["c"] - 1.18).abs() < 1e-9);
    }

    #[test]
    fn sous_la_mediane_aucun_renfort() {
        // Absent de la map = multiplicateur neutre côté appelant. Surtout pas
        // une valeur < 1,0 : on ne pénalise jamais.
        let factors = taste_boost_factors(&sims(&[("a", 0.1), ("b", 0.5), ("c", 0.9)]), 1.18);
        assert!(!factors.contains_key("a"));
        assert!(!factors.contains_key("b"), "la médiane elle-même n'est pas renforcée");
        for f in factors.values() {
            assert!(*f >= 1.0, "aucun facteur ne doit pénaliser");
        }
    }

    #[test]
    fn le_renfort_est_monotone() {
        // Sept valeurs pour que TROIS soient strictement au-dessus de la
        // médiane (0,4) : avec cinq, la valeur centrale est la médiane
        // elle-même, donc exclue, et il ne resterait que deux points à
        // comparer — pas de quoi vérifier une monotonie.
        let factors = taste_boost_factors(
            &sims(&[
                ("a", 0.1),
                ("b", 0.2),
                ("c", 0.3),
                ("mediane", 0.4),
                ("d", 0.6),
                ("e", 0.8),
                ("f", 1.0),
            ]),
            1.18,
        );
        assert!(factors["f"] > factors["e"]);
        assert!(factors["e"] > factors["d"]);
        assert!(factors["d"] > 1.0);
    }

    #[test]
    fn echelle_absolue_sans_effet() {
        // Le cœur du choix : deux viviers aux échelles très différentes mais au
        // même classement doivent produire les mêmes facteurs. C'est ce qui
        // rend le renfort insensible au modèle d'embedding.
        let bas = taste_boost_factors(&sims(&[("a", 0.01), ("b", 0.02), ("c", 0.03)]), 1.18);
        let haut = taste_boost_factors(&sims(&[("a", 0.71), ("b", 0.72), ("c", 0.73)]), 1.18);
        assert!((bas["c"] - haut["c"]).abs() < 1e-9);
    }

    #[test]
    fn vivier_plat_ne_renforce_personne() {
        let factors = taste_boost_factors(&sims(&[("a", 0.5), ("b", 0.5), ("c", 0.5)]), 1.18);
        assert!(factors.is_empty());
    }

    #[test]
    fn vivier_vide_ne_panique_pas() {
        assert!(taste_boost_factors(&HashMap::new(), 1.18).is_empty());
    }

    #[test]
    fn un_seul_candidat_ne_recoit_rien() {
        // Médiane == max : il n'y a personne à départager.
        assert!(taste_boost_factors(&sims(&[("a", 0.9)]), 1.18).is_empty());
    }
}
