/// Phase 3 — Contextual Bandit (Epsilon-Greedy)
///
/// 80% Exploit : affiche les tweets avec le meilleur score prédit
/// 20% Explore : injecte des tweets diversifiés/inattendus pour découvrir
///               de nouvelles préférences utilisateur
///
/// Gain CTR attendu : +1-1.5% (nouvelles découvertes → engagement futur)
use rand::Rng;
use tracing::{debug, trace};

use crate::models::{RawTweet, ScoredTweet, TweetSource, UserProfile};

const EPSILON: f64 = 0.20; // 20% exploration
const MIN_EXPLOIT_SCORE: f64 = 0.30; // seuil minimum pour le pool exploit

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
) -> BanditSelection {
    if scored.is_empty() {
        return BanditSelection { tweet_ids: vec![], exploit_count: 0, explore_count: 0 };
    }

    let n_explore = (limit as f64 * EPSILON).round() as usize;
    let n_exploit = limit.saturating_sub(n_explore);

    // Pool exploit : top tweets par score (déjà triés)
    let exploit_pool: Vec<&ScoredTweet> = scored.iter()
        .filter(|s| s.score >= MIN_EXPLOIT_SCORE)
        .take(n_exploit * 3) // oversample pour pouvoir filtrer
        .collect();

    // Pool explore : tweets diversifiés (sources inattendues, auteurs nouveaux)
    let explore_pool: Vec<&ScoredTweet> = scored.iter()
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

    // 2. Remplir explore : sélection aléatoire parmi candidats exploration
    let mut explore_candidates: Vec<_> = explore_pool.iter()
        .filter(|s| !result.contains(&s.tweet_id))
        .collect();

    // Shuffle pour exploration vraiment aléatoire
    let n = explore_candidates.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        explore_candidates.swap(i, j);
    }

    for s in explore_candidates.iter().take(n_explore) {
        result.push(s.tweet_id.clone());
        explore_count += 1;
        trace!(tweet_id = %s.tweet_id, score = s.score, "Bandit: EXPLORE");
    }

    // Compléter si manque de candidats explore
    if result.len() < limit {
        for s in scored.iter() {
            if result.len() >= limit { break; }
            if !result.contains(&s.tweet_id) {
                result.push(s.tweet_id.clone());
                exploit_count += 1;
            }
        }
    }

    // Interleave exploit + explore pour meilleure UX (pas tout l'explore à la fin)
    let interleaved = interleave_explore(&result, exploit_count, explore_count);

    debug!(
        total = interleaved.len(), exploit_count, explore_count,
        epsilon = EPSILON, "Bandit selection complete"
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
    let is_not_followed = !profile.following_ids.contains(&raw.user_id);
    let has_decent_score = scored.score > 0.20; // minimum qualité

    has_decent_score && (is_new_author || is_discovery) && is_not_followed
}

/// Interleave : place les tweets explore à intervalles réguliers dans le feed
fn interleave_explore(ids: &[String], n_exploit: usize, n_explore: usize) -> Vec<String> {
    if n_explore == 0 { return ids.to_vec(); }

    let interval = if n_explore > 0 { (n_exploit / n_explore).max(1) } else { usize::MAX };
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
        }
    }

    #[test]
    fn test_exploit_explore_ratio() {
        let scored: Vec<ScoredTweet> = (0..50)
            .map(|i| make_scored(&format!("tweet_{i}"), 1.0 - i as f64 * 0.01))
            .collect();

        let selection = select(&scored, &[], &UserProfile::default(), 10);
        assert_eq!(selection.tweet_ids.len(), 10);
        // exploit ~80%, explore ~20% (peut varier si peu de candidats explore)
        assert!(selection.exploit_count >= 6, "Should exploit at least 60%");
    }

    #[test]
    fn test_empty_input() {
        let selection = select(&[], &[], &UserProfile::default(), 10);
        assert!(selection.tweet_ids.is_empty());
    }
}
