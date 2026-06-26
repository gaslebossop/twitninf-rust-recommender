use crate::constants::*;
use crate::models::{RawTweet, UserProfile};
use crate::utils::math::sigmoid;
use tracing::{debug, trace};

/// D3 — SOCIAL GRAPH DYNAMICS (15%)
/// Proximité sociale multi-degrés : suivi direct, mutuel, amis d'amis
pub fn calculate(t: &RawTweet, profile: &UserProfile) -> (f64, f64, f64, f64) {
    let mut score = 0.0;
    trace!(author_id = %t.user_id, "D3 Social graph analysis");

    let direct = if profile.following_ids.contains(&t.user_id) {
        D3_DIRECT_FOLLOW_BOOST
    } else {
        0.0
    };
    score += direct;
    trace!(direct, is_following = profile.following_ids.contains(&t.user_id), "D3 Degree 1: Direct follow");

    let mutual = if profile.mutual_follow_ids.contains(&t.user_id) {
        D3_MUTUAL_FOLLOW_BOOST
    } else {
        0.0
    };
    score += mutual;
    trace!(mutual, is_mutual = profile.mutual_follow_ids.contains(&t.user_id), "D3 Degree 1.5: Mutual follow");

    let second = if profile.second_degree_ids.contains(&t.user_id) {
        D3_SECOND_DEGREE_BOOST
    } else {
        0.0
    };
    score += second;
    trace!(second, "D3 Degree 2: Second degree");

    let affinity_bonus = profile
        .top_authors
        .iter()
        .find(|(uid, _)| uid == &t.user_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
        * D3_PRIOR_AFFINITY_WEIGHT;
    score += affinity_bonus;
    trace!(affinity_bonus, "D3 Prior author affinity");

    let author_influence = sigmoid((t.author_followers as f64).ln().max(0.0) / 10.0) * D3_AUTHOR_INFLUENCE_WEIGHT;
    score += author_influence;
    trace!(author_followers = t.author_followers, author_influence, "D3 Author network influence");

    let final_d3 = score.clamp(0.0, 1.0);
    debug!(final_d3, direct, mutual, second, "D3 Final");
    (final_d3, direct, mutual, second)
}
