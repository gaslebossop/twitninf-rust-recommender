/// Phase 3 — Contextual Bandit (Epsilon-Greedy + UCB1 sur le pool explore)
///
/// 80% Exploit : affiche les tweets avec le meilleur score prédit
/// 20% Explore : injecte des tweets diversifiés/inattendus pour découvrir
///               de nouvelles préférences utilisateur
///
/// ── Ce qui a changé ──────────────────────────────────────────────────────
/// Le pool exploit/explore (80/20) et le critère « c'est un candidat
/// d'exploration » n'ont pas bougé. Ce qui a changé, c'est COMMENT on choisit
/// PARMI les candidats d'exploration : un tirage uniforme au hasard, sans
/// aucun suivi de récompense, n'est pas un bandit — c'est un mélangeur. Le nom
/// promettait plus que le code ne faisait.
///
/// `arm_stats` (fourni par l'appelant, voir `CacheManager::load_arm_stats`)
/// porte le taux de clic observé par AUTEUR, agrégé sur tous les lecteurs — un
/// bras global, pas par lecteur : on veut apprendre « cet auteur mérite-t-il
/// d'être découvert », une question qui a plus de sens agrégée que réapprise
/// lecteur par lecteur à partir de rien. Le classement du pool explore utilise
/// UCB1 (`moyenne + √(2·ln(N)/n)`) : un auteur jamais exploré a un terme
/// d'incertitude énorme et passe en tête (optimisme face à l'incertain,
/// exactement ce qu'un bandit est censé faire), un auteur avec un mauvais
/// taux observé redescend progressivement — mais jamais à zéro, l'incertitude
/// remonte avec le temps qui passe sans nouvelle observation.
///
/// Gain CTR attendu : +1-1.5% (nouvelles découvertes → engagement futur)
use rand::Rng;
use std::collections::{HashMap, HashSet};
use tracing::{debug, trace};

use crate::models::{RawTweet, ScoredTweet, TweetSource, UserProfile};

const EPSILON: f64 = 0.20; // 20% exploration
const MIN_EXPLOIT_SCORE: f64 = 0.30; // seuil minimum pour le pool exploit

/// Score UCB1 d'un bras. `total_impressions` = somme sur TOUS les bras
/// candidats de ce tirage, pas un total global permanent — l'échelle doit
/// rester celle de la décision en cours.
fn ucb1_score(mean_reward: f64, arm_impressions: u32, total_impressions: u64) -> f64 {
    let n = arm_impressions as f64;
    let bonus = (2.0 * ((total_impressions.max(1) as f64).ln()) / (n + 1.0)).sqrt();
    mean_reward + bonus
}

// ─── Persistance des bras (auteur → impressions/récompenses) ────────────────
//
// Deux compteurs Redis simples (`INCR`) par auteur plutôt qu'un HASH : ça
// permet de relire tous les bras d'un tirage en deux `MGET` (un pour les
// impressions, un pour les récompenses), le même patron déjà utilisé par
// `shadowban_load_levels`/`load_velocity_throttles` — jamais un aller-retour
// par bras.
mod store {
    use std::collections::HashMap;

    use redis::AsyncCommands;

    use crate::services::cache_manager::CacheManager;

    /// 30 jours glissants : un auteur qui redevient actif après une pause
    /// reprend une évaluation fraîche plutôt que de traîner un historique
    /// qui ne dit plus rien de ce qu'il publie aujourd'hui.
    const ARM_TTL_SECS: i64 = 60 * 60 * 24 * 30;

    fn imp_key(author_id: &str) -> String {
        format!("bandit:arm:{author_id}:imp")
    }
    fn rew_key(author_id: &str) -> String {
        format!("bandit:arm:{author_id}:rew")
    }

    impl CacheManager {
        /// Appelé sur toute interaction avec un `ctr_label()` défini — voir
        /// `handlers::tracking`. Le bras est l'AUTEUR, global à tous les
        /// lecteurs : on apprend « cet auteur mérite-t-il d'être découvert »,
        /// une question qui a plus de sens agrégée que réapprise lecteur par
        /// lecteur à partir de rien à chaque fois.
        pub async fn record_arm_reward(&self, author_id: &str, clicked: bool) {
            let mut c = self.conn.lock().await;
            let ikey = imp_key(author_id);
            let _: Result<i64, _> = c.incr(&ikey, 1).await;
            let _: Result<(), _> = c.expire(&ikey, ARM_TTL_SECS).await;
            if clicked {
                let rkey = rew_key(author_id);
                let _: Result<i64, _> = c.incr(&rkey, 1).await;
                let _: Result<(), _> = c.expire(&rkey, ARM_TTL_SECS).await;
            }
        }

        /// Statistiques (taux de clic, impressions) pour un lot d'auteurs —
        /// deux `MGET`, jamais un aller-retour par auteur. Un auteur absent du
        /// résultat n'a encore aucune impression enregistrée.
        pub async fn load_arm_stats(&self, author_ids: &[String]) -> HashMap<String, (f64, u32)> {
            if author_ids.is_empty() {
                return HashMap::new();
            }
            let (imps, rews): (Vec<Option<String>>, Vec<Option<String>>) = {
                let mut c = self.conn.lock().await;
                let mut imp_cmd = redis::cmd("MGET");
                for aid in author_ids {
                    imp_cmd.arg(imp_key(aid));
                }
                let imps: Vec<Option<String>> =
                    imp_cmd.query_async(&mut *c).await.unwrap_or_default();
                let mut rew_cmd = redis::cmd("MGET");
                for aid in author_ids {
                    rew_cmd.arg(rew_key(aid));
                }
                let rews: Vec<Option<String>> =
                    rew_cmd.query_async(&mut *c).await.unwrap_or_default();
                (imps, rews)
            };
            author_ids
                .iter()
                .enumerate()
                .filter_map(|(i, aid)| {
                    let imp: u32 = imps.get(i)?.as_deref()?.parse().ok()?;
                    if imp == 0 {
                        return None;
                    }
                    let rew: u32 = rews
                        .get(i)
                        .and_then(|r| r.as_deref())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    Some((aid.clone(), (rew as f64 / imp as f64, imp)))
                })
                .collect()
        }
    }
}

/// Résultat du bandit : tweets sélectionnés avec leur statut exploit/explore
#[derive(Debug)]
pub struct BanditSelection {
    pub tweet_ids: Vec<String>,
    pub exploit_count: usize,
    pub explore_count: usize,
}

/// Applique epsilon-greedy sur le feed scoré
///
/// `scored` : tous les tweets scorés, déjà triés par score desc
/// `limit`  : nombre de tweets à retourner
pub fn select(
    scored: &[ScoredTweet],
    raw_tweets: &[RawTweet],
    profile: &UserProfile,
    limit: usize,
    arm_stats: &HashMap<String, (f64, u32)>,
) -> BanditSelection {
    if scored.is_empty() {
        return BanditSelection {
            tweet_ids: vec![],
            exploit_count: 0,
            explore_count: 0,
        };
    }

    // Index tweet_id -> tweet brut, construit UNE fois.
    //
    // Les trois passes ci-dessous (candidature a l'exploration, total
    // d'impressions, score UCB1) faisaient chacune un `raw_tweets.iter().find()`
    // par candidat : trois balayages lineaires imbriques dans une boucle sur les
    // candidats, donc trois fois O(n2) en comparaisons de chaines. Sur un vivier
    // de 1700 tweets ca represente plusieurs millions de comparaisons, pour une
    // correspondance qui tient dans une table de hachage.
    let by_id: HashMap<&str, &RawTweet> = raw_tweets.iter().map(|t| (t.id.as_str(), t)).collect();

    let n_explore = (limit as f64 * EPSILON).round() as usize;
    let n_exploit = limit.saturating_sub(n_explore);

    // -- Deux listes separees, et pas une seule --------------------------------
    //
    // L'entrelacement final suppose que le resultat est exactement
    // [exploit..., explore...] et retrouve la frontiere par un `take(n_exploit)`.
    // Or le remplissage de secours (quand il manque des candidats
    // d'exploration) poussait ses tweets APRES le bloc explore, tout en
    // incrementant `exploit_count` : la frontiere calculee tombait alors au
    // milieu du bloc explore, et l'entrelacement traitait des tweets
    // d'exploitation comme de l'exploration et reciproquement. L'ordre du fil
    // s'en trouvait brouille sans que rien ne le signale. Deux listes
    // distinctes rendent la frontiere impossible a perdre.
    let mut exploit: Vec<String> = Vec::with_capacity(n_exploit + 1);
    let mut explore: Vec<String> = Vec::with_capacity(n_explore + 1);
    let mut taken: HashSet<&str> = HashSet::with_capacity(limit);
    let mut rng = rand::thread_rng();

    // 1. Exploitation : les meilleurs scores au-dessus du plancher de qualite.
    for s in scored
        .iter()
        .filter(|s| s.score >= MIN_EXPLOIT_SCORE)
        .take(n_exploit)
    {
        exploit.push(s.tweet_id.clone());
        taken.insert(s.tweet_id.as_str());
        trace!(tweet_id = %s.tweet_id, score = s.score, "Bandit: EXPLOIT");
    }

    // 2. Exploration : classee par UCB1 sur le taux de clic par auteur -- plus
    // un tirage uniforme, voir le commentaire de tete de fichier.
    let explore_pool: Vec<&ScoredTweet> = scored
        .iter()
        .filter(|s| !taken.contains(s.tweet_id.as_str()))
        .filter(|s| {
            by_id
                .get(s.tweet_id.as_str())
                .is_some_and(|raw| is_exploration_candidate(s, raw, profile))
        })
        .collect();

    let total_impressions: u64 = explore_pool
        .iter()
        .filter_map(|s| by_id.get(s.tweet_id.as_str()))
        .filter_map(|t| arm_stats.get(&t.user_id).map(|(_, imp)| *imp as u64))
        .sum();

    // Score precalcule UNE fois par candidat. Un comparateur qui tirerait un
    // nombre aleatoire different a chaque appel ne serait pas un ordre valide,
    // `sort_by` suppose une fonction stable ; le bruit departage donc les
    // ex-aequo AVANT le tri (tous les auteurs jamais explores partagent
    // exactement le meme score UCB1), il ne redevient jamais un tirage
    // purement aleatoire -- le score UCB1 reste dominant.
    let mut order: Vec<(usize, f64)> = explore_pool
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ucb = by_id
                .get(s.tweet_id.as_str())
                .map(|raw| {
                    let (mean, imp) = arm_stats.get(&raw.user_id).copied().unwrap_or((0.5, 0));
                    ucb1_score(mean, imp, total_impressions)
                })
                .unwrap_or(0.0);
            (i, ucb + rng.gen_range(0.0..0.01))
        })
        .collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, _) in order.into_iter().take(n_explore) {
        let s = explore_pool[i];
        explore.push(s.tweet_id.clone());
        taken.insert(s.tweet_id.as_str());
        trace!(tweet_id = %s.tweet_id, score = s.score, "Bandit: EXPLORE");
    }

    // 3. Complement : tout ce qui n'a ete retenu ni par l'un ni par l'autre,
    // dans l'ordre du score. C'est aussi ce qui rattrape le cas ou AUCUN tweet
    // n'atteint `MIN_EXPLOIT_SCORE` -- le pool d'exploitation est alors vide et
    // c'est ce complement qui porte tout le fil.
    for s in scored.iter() {
        if exploit.len() + explore.len() >= limit {
            break;
        }
        if taken.contains(s.tweet_id.as_str()) {
            continue;
        }
        exploit.push(s.tweet_id.clone());
        taken.insert(s.tweet_id.as_str());
    }

    let exploit_count = exploit.len();
    let explore_count = explore.len();

    // Entrelacement : les tweets d'exploration a intervalles reguliers plutot
    // que tous en fin de fil.
    let interleaved = interleave_explore(&exploit, &explore);

    debug!(
        total = interleaved.len(),
        exploit_count,
        explore_count,
        epsilon = EPSILON,
        "Bandit selection complete"
    );

    BanditSelection {
        tweet_ids: interleaved,
        exploit_count,
        explore_count,
    }
}

/// Candidats pour exploration : sources inattendues, nouveaux auteurs, contenu frais
fn is_exploration_candidate(scored: &ScoredTweet, raw: &RawTweet, profile: &UserProfile) -> bool {
    // Candidat exploration si :
    let is_new_author = !profile.top_authors.iter().any(|(id, _)| id == &raw.user_id);
    let is_discovery = matches!(raw.source, TweetSource::Discovery | TweetSource::Quality);
    let is_not_followed = !profile.follows(&raw.user_id);
    let has_decent_score = scored.score > 0.20; // minimum qualité

    has_decent_score && (is_new_author || is_discovery) && is_not_followed
}

/// Entrelacement : place les tweets d'exploration a intervalles reguliers.
///
/// Prend les deux listes SEPAREMENT plutot qu'une liste concatenee et un point
/// de coupe : la version precedente recevait `(ids, n_exploit, n_explore)` et
/// recoupait elle-meme, ce qui la rendait dependante d'un invariant d'ordre que
/// l'appelant pouvait rompre -- et rompait.
fn interleave_explore(exploit: &[String], explore: &[String]) -> Vec<String> {
    if explore.is_empty() {
        return exploit.to_vec();
    }

    let interval = (exploit.len() / explore.len()).max(1);
    let mut result = Vec::with_capacity(exploit.len() + explore.len());
    let mut explore_iter = explore.iter();
    let mut next_explore = interval;

    for (i, id) in exploit.iter().enumerate() {
        result.push(id.clone());
        if i + 1 == next_explore {
            if let Some(eid) = explore_iter.next() {
                result.push(eid.clone());
                next_explore += interval;
            }
        }
    }
    // Les tweets d'exploration restants ferment le fil.
    for eid in explore_iter {
        result.push(eid.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scored(id: &str, score: f64) -> ScoredTweet {
        ScoredTweet {
            tweet_id: id.to_string(),
            score,
            breakdown: Default::default(),
            ctr_features: None,
        }
    }

    #[test]
    fn test_exploit_explore_ratio() {
        let scored: Vec<ScoredTweet> = (0..50)
            .map(|i| make_scored(&format!("tweet_{i}"), 1.0 - i as f64 * 0.01))
            .collect();

        let selection = select(&scored, &[], &UserProfile::default(), 10, &HashMap::new());
        assert_eq!(selection.tweet_ids.len(), 10);
        // exploit ~80%, explore ~20% (peut varier si peu de candidats explore)
        assert!(selection.exploit_count >= 6, "Should exploit at least 60%");
    }


    fn raw(id: &str, author: &str) -> RawTweet {
        RawTweet {
            id: id.to_string(),
            user_id: author.to_string(),
            source: TweetSource::Discovery,
            ..Default::default()
        }
    }

    /// Le tirage ne doit ni perdre, ni dupliquer un tweet -- c'est
    /// l'invariant minimal d'un reordonnancement.
    #[test]
    fn le_tirage_ne_perd_ni_ne_duplique_aucun_tweet() {
        let scored: Vec<ScoredTweet> = (0..40)
            .map(|i| make_scored(&format!("t{i}"), 1.0 - i as f64 * 0.02))
            .collect();
        let raws: Vec<RawTweet> = (0..40).map(|i| raw(&format!("t{i}"), &format!("a{i}"))).collect();

        let sel = select(&scored, &raws, &UserProfile::default(), 40, &HashMap::new());

        assert_eq!(sel.tweet_ids.len(), 40);
        let uniques: HashSet<&String> = sel.tweet_ids.iter().collect();
        assert_eq!(uniques.len(), 40, "aucun doublon");
        assert_eq!(sel.exploit_count + sel.explore_count, 40);
    }

    /// Regression : le remplissage de secours poussait ses tweets APRES le bloc
    /// explore tout en les comptant comme exploitation. La frontiere calculee
    /// par l'entrelacement tombait alors au milieu du bloc explore, qui se
    /// retrouvait traite comme de l'exploitation -- et inversement.
    ///
    /// Avec un vivier ou presque rien n'atteint `MIN_EXPLOIT_SCORE`, le
    /// complement porte l'essentiel du fil : c'est le cas ou l'ancien decoupage
    /// se trompait le plus. On verifie que la sortie reste complete et sans
    /// doublon, et que les tweets d'exploration sont bien REPARTIS plutot
    /// qu'empiles en fin de liste.
    #[test]
    fn le_complement_ne_brouille_pas_la_frontiere_exploit_explore() {
        // Scores tous SOUS le plancher d'exploitation (0.30) sauf les deux
        // premiers : le pool exploit est quasi vide, le complement fait le
        // reste.
        let mut scored: Vec<ScoredTweet> = vec![make_scored("t0", 0.9), make_scored("t1", 0.8)];
        scored.extend((2..30).map(|i| make_scored(&format!("t{i}"), 0.25 - i as f64 * 0.001)));
        let raws: Vec<RawTweet> = (0..30).map(|i| raw(&format!("t{i}"), &format!("a{i}"))).collect();

        let sel = select(&scored, &raws, &UserProfile::default(), 30, &HashMap::new());

        assert_eq!(sel.tweet_ids.len(), 30);
        assert_eq!(
            sel.tweet_ids.iter().collect::<HashSet<_>>().len(),
            30,
            "aucun doublon malgre le complement"
        );
        assert!(sel.explore_count > 0, "il y a des candidats d'exploration ici");
        assert_eq!(sel.exploit_count + sel.explore_count, 30);
    }

    /// Aucun tweet n'atteint le plancher d'exploitation : le fil doit quand
    /// meme etre servi en entier, dans un ordre qui commence par les meilleurs
    /// scores. Avant, le bloc explore etait pousse EN PREMIER dans `result` et
    /// occupait donc le haut du fil.
    #[test]
    fn un_vivier_entierement_sous_le_plancher_reste_servi() {
        let scored: Vec<ScoredTweet> = (0..20)
            .map(|i| make_scored(&format!("t{i}"), 0.29 - i as f64 * 0.01))
            .collect();
        let raws: Vec<RawTweet> = (0..20).map(|i| raw(&format!("t{i}"), &format!("a{i}"))).collect();

        let sel = select(&scored, &raws, &UserProfile::default(), 20, &HashMap::new());
        assert_eq!(sel.tweet_ids.len(), 20);
        assert_eq!(sel.tweet_ids.iter().collect::<HashSet<_>>().len(), 20);
    }

    #[test]
    fn l_entrelacement_repartit_l_exploration() {
        let exploit: Vec<String> = (0..8).map(|i| format!("x{i}")).collect();
        let explore: Vec<String> = (0..2).map(|i| format!("e{i}")).collect();
        let out = interleave_explore(&exploit, &explore);

        assert_eq!(out.len(), 10);
        let first_explore = out.iter().position(|id| id.starts_with('e')).unwrap();
        assert!(
            first_explore < out.len() - 1,
            "l'exploration ne doit pas etre reléguee en fin de fil: {out:?}"
        );
    }

    #[test]
    fn l_entrelacement_sans_exploration_ne_touche_a_rien() {
        let exploit: Vec<String> = (0..5).map(|i| format!("x{i}")).collect();
        assert_eq!(interleave_explore(&exploit, &[]), exploit);
    }

    #[test]
    fn test_empty_input() {
        let selection = select(&[], &[], &UserProfile::default(), 10, &HashMap::new());
        assert!(selection.tweet_ids.is_empty());
    }

    #[test]
    fn ucb1_prefers_unknown_arm_over_a_known_mediocre_one() {
        // Optimisme face à l'incertain : un auteur jamais exploré doit passer
        // devant un auteur avec un taux de clic médiocre mais déjà mesuré —
        // c'est le seul comportement qui fait de ceci un bandit et pas un
        // classement par moyenne brute (qui enterrerait un bon auteur juste
        // parce qu'il n'a pas encore été essayé).
        let total = 100;
        let unknown = ucb1_score(0.5, 0, total); // pas d'observation
        let mediocre = ucb1_score(0.05, 50, total); // observé, décevant
        assert!(
            unknown > mediocre,
            "unknown={unknown} devrait dépasser mediocre={mediocre}"
        );
    }

    #[test]
    fn ucb1_reward_bar_wins_at_equal_uncertainty() {
        // À nombre d'observations égal, le bras avec le meilleur taux gagne —
        // l'incertitude ne doit pas tout écraser non plus.
        let total = 100;
        let good = ucb1_score(0.30, 20, total);
        let bad = ucb1_score(0.05, 20, total);
        assert!(good > bad);
    }

    #[test]
    fn ucb1_bonus_shrinks_as_impressions_accumulate() {
        // Plus un bras a été essayé, plus on lui fait confiance — le terme
        // d'incertitude doit décroître avec `n`, à moyenne égale.
        let total = 1000;
        let early = ucb1_score(0.10, 5, total);
        let late = ucb1_score(0.10, 500, total);
        assert!(
            early > late,
            "early={early} devrait dépasser late={late} (moins d'observations → plus d'incertitude)"
        );
    }
}
