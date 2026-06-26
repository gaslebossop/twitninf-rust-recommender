use crate::constants::*;
use crate::models::{RawTweet, UserProfile, UserType, ContentLength};
use crate::utils::math::sigmoid;
use tracing::{debug, trace};

pub fn calculate(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0;

    let activity_bonus = match profile.user_type {
        UserType::PowerUser => D5_POWERUSER_BONUS,
        UserType::Regular => D5_REGULAR_BONUS,
        UserType::Casual => D5_CASUAL_BONUS,
    };
    score += activity_bonus;
    trace!(activity_bonus, "D5 User activity bonus");

    if profile.prefers_media && t.has_media {
        score += D5_MEDIA_PREFERENCE_BONUS;
    }

    let length_match = match (&profile.preferred_content_length, t.content_length) {
        (ContentLength::Short, l) if l < 100 => D5_LENGTH_MATCH_BONUS,
        (ContentLength::Medium, l) if (80..=200).contains(&l) => D5_LENGTH_MATCH_BONUS,
        (ContentLength::Long, l) if l > 180 => D5_LENGTH_MATCH_BONUS,
        _ => 0.05,
    };
    score += length_match;

    let profile_retweet_rate = if !profile.liked_tweet_ids.is_empty() {
        profile.retweeted_tweet_ids.len() as f64 / profile.liked_tweet_ids.len() as f64
    } else {
        0.1
    };
    let retweet_pred =
        sigmoid((t.retweet_count as f64 / t.view_count.max(1) as f64) * 100.0) * profile_retweet_rate;
    score += (retweet_pred * D5_RETWEET_PREDICTION_WEIGHT).min(0.20);

    if profile.engagement_trend > D5_ENGAGEMENT_TREND_THRESHOLD {
        score += D5_ENGAGEMENT_TREND_BONUS;
    }

    score += (1.0 - profile.churn_risk) * D5_LOYALTY_WEIGHT;

    let d5 = score.clamp(0.0, 1.0);
    debug!(d5, "D5 Final");
    d5
}
