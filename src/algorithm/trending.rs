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
