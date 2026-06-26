use crate::constants::*;
use crate::models::{RawTweet, UserProfile, ScoredTweet};
use tracing::{debug, trace};

pub fn calculate(t: &RawTweet, profile: &UserProfile, feed: &[ScoredTweet]) -> f64 {
    let feed_size = feed.len();
    if feed_size == 0 {
        return D6_FIRST_TWEET_SCORE;
    }

    let mut score = D6_BASE_SCORE;
    let media_ratio_in_feed = feed.iter().filter(|_| true).count() as f64 / feed_size.max(1) as f64;

    if t.has_media && media_ratio_in_feed < D6_MEDIA_DIVERSITY_THRESHOLD {
        score += D6_MEDIA_BONUS;
    }

    if t.hashtag_count > 0 {
        score += D6_HASHTAG_NOVELTY_BONUS;
    }

    if !profile.seen_tweet_ids.contains(&t.id) {
        score += D6_CONTENT_NOVELTY_BONUS;
    }

    let d6 = score.clamp(0.0, 1.0);
    debug!(d6, "D6 Final");
    d6
}
