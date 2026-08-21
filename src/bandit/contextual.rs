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
use std::collections::HashMap;
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

    let n_explore = (limit as f64 * EPSILON).round() as usize;
    let n_exploit = limit.saturating_sub(n_explore);

    // Pool exploit : top tweets par score (déjà triés)
    let exploit_pool: Vec<&ScoredTweet> = scored
        .iter()
        .filter(|s| s.score >= MIN_EXPLOIT_SCORE)
        .take(n_exploit * 3) // oversample pour pouvoir filtrer
        .collect();

    // Pool explore : tweets diversifiés (sources inattendues, auteurs nouveaux)
    let explore_pool: Vec<&ScoredTweet> = scored
        .iter()
        .filter(|s| is_exploration_candidate(s, raw_tweets, profile))
        .collect();

    let mut result: Vec<String> = Vec::with_capacity(limit);
    let mut exploit_count = 0;
    let mut explore_count = 0;
    let mut rng = rand::thread_rng();

    // 1. Remplir exploit : prendre les meilleurs scores
    for s in exploit_pool.iter().take(n_exploit) {
        result.push(s.tweet_id.clone());
        exploit_count += 1;
        trace!(tweet_id = %s.tweet_id, score = s.score, "Bandit: EXPLOIT");
    }

    // 2. Remplir explore : classé par UCB1 sur le taux de clic par auteur —
    // plus un tirage uniforme, voir le commentaire de tête de fichier.
    let mut explore_candidates: Vec<_> = explore_pool
        .iter()
        .filter(|s| !result.contains(&s.tweet_id))
        .collect();

    let total_impressions: u64 = explore_candidates
        .iter()
        .filter_map(|s| raw_tweets.iter().find(|t| t.id == s.tweet_id))
        .filter_map(|t| arm_stats.get(&t.user_id).map(|(_, imp)| *imp as u64))
        .sum();

    // Score précalculé UNE fois par candidat (index dans `explore_candidates`,
    // pas la référence elle-même — plus simple que de manier des piles de
    // références imbriquées). Un comparateur qui tirerait un nombre aléatoire
    // différent à chaque appel ne serait pas un ordre valide, `sort_by`
    // suppose une fonction stable ; le bruit départage donc les ex-æquo AVANT
    // le tri (tous les auteurs jamais explorés partagent exactement le même
    // score UCB1), il ne redevient jamais un tirage purement aléatoire — le
    // score UCB1 reste dominant, le bruit ne fait que casser les égalités.
    let mut order: Vec<(usize, f64)> = explore_candidates
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ucb = raw_tweets
                .iter()
                .find(|t| t.id == s.tweet_id)
                .map(|raw| {
                    let (mean, imp) = arm_stats.get(&raw.user_id).copied().unwrap_or((0.5, 0));
                    ucb1_score(mean, imp, total_impressions)
                })
                .unwrap_or(0.0);
            (i, ucb + rng.gen_range(0.0..0.01))
        })
        .collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let explore_candidates: Vec<_> = order
        .into_iter()
        .map(|(i, _)| explore_candidates[i])
        .collect();

    for s in explore_candidates.iter().take(n_explore) {
        result.push(s.tweet_id.clone());
        explore_count += 1;
        trace!(tweet_id = %s.tweet_id, score = s.score, "Bandit: EXPLORE");
    }

    // Compléter si manque de candidats explore
    if result.len() < limit {
        for s in scored.iter() {
            if result.len() >= limit {
                break;
            }
            if !result.contains(&s.tweet_id) {
                result.push(s.tweet_id.clone());
                exploit_count += 1;
            }
        }
    }

    // Interleave exploit + explore pour meilleure UX (pas tout l'explore à la fin)
    let interleaved = interleave_explore(&result, exploit_count, explore_count);

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
fn is_exploration_candidate(
    scored: &ScoredTweet,
    raw_tweets: &[RawTweet],
    profile: &UserProfile,
) -> bool {
    let Some(raw) = raw_tweets.iter().find(|t| t.id == scored.tweet_id) else {
        return false;
    };

    // Candidat exploration si :
    let is_new_author = !profile.top_authors.iter().any(|(id, _)| id == &raw.user_id);
    let is_discovery = matches!(raw.source, TweetSource::Discovery | TweetSource::Quality);
    let is_not_followed = !profile.follows(&raw.user_id);
    let has_decent_score = scored.score > 0.20; // minimum qualité

    has_decent_score && (is_new_author || is_discovery) && is_not_followed
}

/// Interleave : place les tweets explore à intervalles réguliers dans le feed
fn interleave_explore(ids: &[String], n_exploit: usize, n_explore: usize) -> Vec<String> {
    if n_explore == 0 {
        return ids.to_vec();
    }

    let interval = if n_explore > 0 {
        (n_exploit / n_explore).max(1)
    } else {
        usize::MAX
    };
    let exploit_ids: Vec<_> = ids.iter().take(n_exploit).collect();
    let explore_ids: Vec<_> = ids.iter().skip(n_exploit).collect();

    let mut result = Vec::with_capacity(ids.len());
    let mut explore_iter = explore_ids.iter();
    let mut next_explore = interval;

    for (i, id) in exploit_ids.iter().enumerate() {
        result.push((*id).clone());
        if i + 1 == next_explore {
            if let Some(eid) = explore_iter.next() {
                result.push((*eid).clone());
                next_explore += interval;
            }
        }
    }
    // Ajouter les explore restants à la fin
    for eid in explore_iter {
        result.push((*eid).clone());
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
