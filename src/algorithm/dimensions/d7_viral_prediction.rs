use crate::constants::*;
use crate::models::RawTweet;
use crate::utils::math::sigmoid;
use tracing::{debug, trace};

pub fn calculate(t: &RawTweet, age_h: f64) -> f64 {
    let total_eng = t.like_count + t.comment_count + t.retweet_count + t.share_count;
    if total_eng == 0 {
        return 0.05;
    }

    let share_ratio = (t.retweet_count + t.share_count) as f64 / total_eng.max(1) as f64;
    trace!(share_ratio, "D7 Share ratio");

    let cascade = if t.like_count > 0 {
        (t.retweet_count as f64 / t.like_count as f64).min(CASCADE_RATIO_MAX) / CASCADE_RATIO_MAX
    } else if t.retweet_count > 0 {
        1.0
    } else {
        0.0
    };

    let eng_per_hour = total_eng as f64 / age_h;
    let spread_velocity = sigmoid(eng_per_hour / VIRAL_SPREAD_THRESHOLD);

    let mut shareability = if t.has_media {
        D7_MEDIA_SHAREABILITY
    } else {
        D7_NO_MEDIA_SHAREABILITY
    };
    if t.hashtag_count > 0 {
        shareability += D7_HASHTAG_SHAREABILITY;
    }
    if t.url_count > 0 {
        shareability += D7_URL_SHAREABILITY;
    }

    let d7 = (share_ratio * D7_SHARE_RATIO_WEIGHT
        + cascade * D7_CASCADE_WEIGHT
        + spread_velocity * D7_SPREAD_VELOCITY_WEIGHT
        + shareability * D7_SHAREABILITY_WEIGHT)
        .clamp(0.0, 1.0);
    debug!(d7, "D7 Final");
    d7
}
