/// Phase 3 — Real-time CTR Feedback Loop
use deadpool_redis::Pool as RedisPool;
use redis::AsyncCommands;
use tracing::{debug, warn};

const CLICK_BOOST: f64  =  0.15;
const SKIP_PENALTY: f64 = -0.08;
const BOOST_TTL_SECS: u64 = 1800; // 30 minutes
const MAX_HASHTAG_KEY_LEN: usize = 64;

/// Sanitise une chaîne pour l'utiliser dans une clé Redis.
/// Garde uniquement les caractères alphanumériques et tirets/underscores.
/// Tronque à MAX_HASHTAG_KEY_LEN pour éviter les clés gigantesques.
fn sanitize_redis_segment(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(MAX_HASHTAG_KEY_LEN)
        .collect::<String>()
        .to_lowercase()
}

/// Valide qu'une chaîne est un UUID bien formé.
fn is_valid_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

pub async fn record_feedback(
    redis: &RedisPool,
    user_id: &str,
    tweet_id: &str,
    clicked: bool,
    author_id: &str,
    hashtags: &[String],
) {
    // Les deux IDs doivent être des UUIDs valides — rejeter sinon
    if !is_valid_uuid(user_id) || !is_valid_uuid(author_id) {
        warn!(user_id, author_id, "feedback: invalid UUID — skipping");
        return;
    }

    let delta = if clicked { CLICK_BOOST } else { SKIP_PENALTY };

    let author_key = format!("feedback:author:{user_id}:{author_id}");

    // Clés hashtags : sanitisées pour éviter l'injection de clé Redis
    let hashtag_keys: Vec<String> = hashtags.iter()
        .map(|h| {
            let safe = sanitize_redis_segment(h);
            format!("feedback:hashtag:{user_id}:{safe}")
        })
        .filter(|k| !k.ends_with(':')) // rejeter les hashtags vides après sanitisation
        .collect();

    let Ok(mut conn) = redis.get().await else {
        warn!("Redis connection failed — feedback not recorded");
        return;
    };

    let _: Result<(), _> = conn.set_ex(&author_key, delta.to_string(), BOOST_TTL_SECS).await;

    for key in &hashtag_keys {
        let _: Result<(), _> = conn.set_ex(key, delta.to_string(), BOOST_TTL_SECS).await;
    }

    let seen_key = format!("seen:{user_id}");
    let _: Result<(), _> = conn.sadd(&seen_key, tweet_id).await;
    let _: Result<(), _> = conn.expire(&seen_key, 3600 * 24 * 7_usize).await;

    debug!(
        user_id, tweet_id, clicked, delta,
        author_key = %author_key,
        hashtag_count = hashtag_keys.len(),
        "Feedback loop updated"
    );
}

pub async fn get_realtime_boost(
    redis: &RedisPool,
    user_id: &str,
    author_id: &str,
    hashtags: &[String],
) -> f64 {
    if !is_valid_uuid(user_id) || !is_valid_uuid(author_id) {
        return 0.0;
    }

    let Ok(mut conn) = redis.get().await else {
        return 0.0;
    };

    let mut total_boost = 0.0_f64;

    let author_key = format!("feedback:author:{user_id}:{author_id}");
    if let Ok(Some(s)) = conn.get::<_, Option<String>>(&author_key).await {
        total_boost += s.parse::<f64>().unwrap_or(0.0);
    }

    let mut hashtag_boosts = 0.0_f64;
    let mut hashtag_count  = 0usize;
    for h in hashtags {
        let safe = sanitize_redis_segment(h);
        if safe.is_empty() { continue; }
        let key = format!("feedback:hashtag:{user_id}:{safe}");
        if let Ok(Some(s)) = conn.get::<_, Option<String>>(&key).await {
            hashtag_boosts += s.parse::<f64>().unwrap_or(0.0);
            hashtag_count  += 1;
        }
    }
    if hashtag_count > 0 {
        total_boost += hashtag_boosts / hashtag_count as f64;
    }

    total_boost.clamp(-0.20, 0.20)
}

pub async fn is_seen(redis: &RedisPool, user_id: &str, tweet_id: &str) -> bool {
    let Ok(mut conn) = redis.get().await else {
        return false;
    };
    let key = format!("seen:{user_id}");
    conn.sismember::<_, _, bool>(&key, tweet_id).await.unwrap_or(false)
}
