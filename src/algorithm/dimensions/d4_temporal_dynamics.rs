use crate::constants::*;
use crate::models::{RawTweet, UserProfile};
use crate::utils::math::sigmoid;
use tracing::{debug, trace};

pub fn calculate(t: &RawTweet, profile: &UserProfile, age_h: f64) -> f64 {
    let recency = (-RECENCY_DECAY_RATE * age_h).exp();
    trace!(recency, "D4 Recency (6h half-life)");

    let pub_hour = t.created_at.format("%H").to_string().parse::<u32>().unwrap_or(12) as usize;
    let hour_match = profile.hourly_activity[pub_hour];
    trace!(pub_hour, hour_match, "D4 Hourly activity match");

    let pub_day = t.created_at.format("%w").to_string().parse::<usize>().unwrap_or(1);
    let day_match = profile.daily_activity[pub_day];
    trace!(pub_day, day_match, "D4 Daily activity match");

    let momentum = if age_h < MOMENTUM_THRESHOLD_2H {
        let eng_total = t.like_count + t.comment_count + t.retweet_count;
        sigmoid(eng_total as f64 / MOMENTUM_ENGAGEMENT_SCALE) * 1.5
    } else if age_h < MOMENTUM_THRESHOLD_6H {
        1.0
    } else {
        0.7
    };
    trace!(momentum, "D4 Momentum");

    let d4 = (recency * D4_RECENCY_WEIGHT
        + hour_match * D4_HOUR_MATCH_WEIGHT
        + day_match * D4_DAY_MATCH_WEIGHT
        + momentum.min(1.0) * D4_MOMENTUM_WEIGHT)
        .clamp(0.0, 1.0);
    debug!(d4, recency, hour_match, day_match, momentum, "D4 Final");
    d4
}
