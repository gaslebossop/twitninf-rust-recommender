use crate::constants::*;
use crate::models::RawTweet;
use crate::utils::math::{sigmoid, exponential_decay};
use tracing::{debug, trace};

/// D1 — ENGAGEMENT VELOCITY (25%)
/// Mesure la vitesse d'accumulation d'engagement sur 3 fenêtres temporelles.
/// Calcule aussi l'accélération (tendance haussière vs stable).
pub fn calculate(t: &RawTweet, age_h: f64) -> (f64, f64, f64, f64) {
    let age_h = age_h.max(0.1);
    trace!(age_h, "D1 Age in hours");

    // Score brut pondéré par valeur d'action
    let eng_total = calculate_total_engagement(t);
    trace!(likes = t.like_count, comments = t.comment_count, retweets = t.retweet_count,
           shares = t.share_count, bookmarks = t.bookmark_count, views = t.view_count,
           eng_total, "D1 Raw engagement total");

    // Vélocité = engagement / âge pondéré logarithmiquement
    let velocity_raw = eng_total / (1.0 + (age_h / 6.0).ln().max(0.0) * 2.0);
    trace!(velocity_raw, "D1 Raw velocity (engagement/age)");

    // Engagement récent (1h) : signal d'accélération
    let recent_eng = calculate_recent_engagement(t);
    trace!(recent_eng, likes_1h = t.likes_1h, comments_1h = t.comments_1h, retweets_1h = t.retweets_1h,
           "D1 Recent engagement (1h)");

    // Accélération : compare 1h à 6h pour détecter montée en puissance
    let acceleration = calculate_acceleration(t, recent_eng);
    trace!(acceleration, "D1 Acceleration (recent/6h ratio)");

    // Vitesse de viralité = engagement / heure, seuil calibré à 50 eng/h
    let viral_vel = sigmoid(velocity_raw / (age_h * VIRAL_THRESHOLD_ENG_PER_HOUR));
    trace!(viral_vel, "D1 Viral velocity sigmoid");

    // Score D1 composite
    let ev_raw = sigmoid(velocity_raw / ENGAGEMENT_SIGMOID_SCALE);
    let d1 = (ev_raw * 0.50 + acceleration * 0.30 + viral_vel * 0.20).clamp(0.0, 1.0);
    debug!(d1, ev_raw, acceleration, viral_vel, "D1 Final (50% raw_velocity + 30% acceleration + 20% viral_vel)");

    (d1, ev_raw, acceleration, viral_vel)
}

#[inline]
fn calculate_total_engagement(t: &RawTweet) -> f64 {
    t.like_count as f64 * ENG_VELOCITY_LIKE_WEIGHT
        + t.comment_count as f64 * ENG_VELOCITY_COMMENT_WEIGHT
        + t.retweet_count as f64 * ENG_VELOCITY_RETWEET_WEIGHT
        + t.share_count as f64 * ENG_VELOCITY_SHARE_WEIGHT
        + t.bookmark_count as f64 * ENG_VELOCITY_BOOKMARK_WEIGHT
        + t.view_count as f64 * ENG_VELOCITY_VIEW_WEIGHT
}

#[inline]
fn calculate_recent_engagement(t: &RawTweet) -> f64 {
    t.likes_1h as f64 * ENG_VELOCITY_LIKE_WEIGHT
        + t.comments_1h as f64 * ENG_VELOCITY_COMMENT_WEIGHT
        + t.retweets_1h as f64 * ENG_VELOCITY_RETWEET_WEIGHT
}

#[inline]
fn calculate_acceleration(t: &RawTweet, recent_eng: f64) -> f64 {
    let eng_6h = t.likes_6h as f64 * ENG_VELOCITY_LIKE_WEIGHT
        + (t.comment_count.min(t.comments_1h * 6 + 1)) as f64 * ENG_VELOCITY_COMMENT_WEIGHT;

    if eng_6h > 0.0 {
        // ratio 1h annualisé vs 6h : si > 1.0, trending en hausse
        ((recent_eng * 6.0) / eng_6h).clamp(0.0, 5.0) / 5.0
    } else if recent_eng > 0.0 {
        1.0 // entièrement récent → accélération maximale
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engagement_calculation() {
        let t = RawTweet {
            like_count: 100,
            comment_count: 10,
            retweet_count: 5,
            share_count: 2,
            bookmark_count: 3,
            view_count: 1000,
            likes_1h: 50,
            comments_1h: 5,
            retweets_1h: 2,
            likes_6h: 80,
            ..Default::default()
        };

        let total = calculate_total_engagement(&t);
        assert!(total > 0.0);
        assert!(total < 1100.0); // view weight is 0.05
    }
}
