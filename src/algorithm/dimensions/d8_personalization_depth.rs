use crate::constants::*;
use crate::models::{RawTweet, UserProfile};
use tracing::{debug, trace};

pub fn calculate(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0;

    let author_affinity = profile
        .top_authors
        .iter()
        .find(|(uid, _)| uid == &t.user_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    score += author_affinity * D8_AUTHOR_AFFINITY_WEIGHT;
    trace!(author_affinity, "D8 Author affinity");

    let content_lower = t.content.to_lowercase();
    let interest_score: f64 = profile
        .top_words
        .iter()
        .take(20)
        .map(|(word, count)| {
            if content_lower.contains(word.as_str()) {
                (*count as f64).ln() / 5.0
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .min(D8_INTEREST_SCORE_WEIGHT);
    score += interest_score;

    if profile.emotional_positivity > D8_EMOTIONAL_POSITIVITY_THRESHOLD && t.emoji_count > 0 {
        score += D8_EMOTIONAL_BONUS;
    }

    let pub_hour = t.created_at.format("%H").to_string().parse::<u32>().unwrap_or(12);
    if pub_hour == profile.most_active_hour {
        score += D8_PEAK_HOUR_BONUS;
    }

    let d8 = score.clamp(0.0, 1.0);
    debug!(d8, "D8 Final");
    d8
}
