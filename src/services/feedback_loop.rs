/// Phase 3 — Real-time CTR Feedback Loop
///
/// Quand un utilisateur clique → boost immédiat des tweets similaires dans Redis
/// Quand il skip → réduction du score pour éviter de re-montrer
///
/// Implémentation : score delta stocké dans Redis avec TTL court (30 min)
/// Gain CTR attendu : +1-1.5% (feed adaptatif en temps réel)
use deadpool_redis::Pool as RedisPool;
use redis::AsyncCommands;
use tracing::{debug, warn};

const CLICK_BOOST: f64 = 0.15;   // boost CTR pour tweets similaires après click
const SKIP_PENALTY: f64 = -0.08; // pénalité après skip
const BOOST_TTL_SECS: u64 = 1800; // 30 minutes

/// Enregistre une interaction et met à jour les scores Redis en temps réel
pub async fn record_feedback(
    redis: &RedisPool,
    user_id: &str,
    tweet_id: &str,
    clicked: bool,
    author_id: &str,
    hashtags: &[String],
) {
    let delta = if clicked { CLICK_BOOST } else { SKIP_PENALTY };

    // Clé de boost auteur : si click → booster autres tweets du même auteur
    let author_key = format!("feedback:author:{user_id}:{author_id}");
    // Clé de boost hashtag : si click → booster tweets avec mêmes hashtags
    let hashtag_keys: Vec<String> = hashtags.iter()
        .map(|h| format!("feedback:hashtag:{user_id}:{}", h.to_lowercase()))
        .collect();

    let Ok(mut conn) = redis.get().await else {
        warn!("Redis connection failed — feedback not recorded");
        return;
    };

    // Stocker le delta auteur
    let _: Result<(), _> = conn.set_ex(&author_key, delta.to_string(), BOOST_TTL_SECS).await;

    // Stocker les deltas hashtags
    for key in &hashtag_keys {
        let _: Result<(), _> = conn.set_ex(key, delta.to_string(), BOOST_TTL_SECS).await;
    }

    // Log du tweet vu (pour éviter re-affichage)
    let seen_key = format!("seen:{user_id}");
    let _: Result<(), _> = conn.sadd(&seen_key, tweet_id).await;
    let _: Result<(), _> = conn.expire(&seen_key, 3600 * 24 * 7_usize).await; // 7 jours

    debug!(
        user_id, tweet_id, clicked, delta,
        author_key = %author_key,
        hashtag_count = hashtag_keys.len(),
        "Feedback loop updated"
    );
}

/// Récupère le delta de boost temps-réel pour un tweet
/// Additionne les boosts auteur + hashtags de l'utilisateur
pub async fn get_realtime_boost(
    redis: &RedisPool,
    user_id: &str,
    author_id: &str,
    hashtags: &[String],
) -> f64 {
    let Ok(mut conn) = redis.get().await else {
        return 0.0;
    };

    let mut total_boost = 0.0_f64;

    // Boost auteur
    let author_key = format!("feedback:author:{user_id}:{author_id}");
    if let Ok(val) = conn.get::<_, Option<String>>(&author_key).await {
        if let Some(s) = val {
            total_boost += s.parse::<f64>().unwrap_or(0.0);
        }
    }

    // Boost hashtags (moyenne des boosts sur les hashtags du tweet)
    let mut hashtag_boosts = 0.0_f64;
    let mut hashtag_count = 0;
    for h in hashtags {
        let key = format!("feedback:hashtag:{user_id}:{}", h.to_lowercase());
        if let Ok(val) = conn.get::<_, Option<String>>(&key).await {
            if let Some(s) = val {
                hashtag_boosts += s.parse::<f64>().unwrap_or(0.0);
                hashtag_count += 1;
            }
        }
    }
    if hashtag_count > 0 {
        total_boost += hashtag_boosts / hashtag_count as f64;
    }

    total_boost.clamp(-0.20, 0.20) // cap le boost pour stabilité
}

/// Vérifie si un tweet a déjà été vu par l'utilisateur (pour exclusion)
pub async fn is_seen(redis: &RedisPool, user_id: &str, tweet_id: &str) -> bool {
    let Ok(mut conn) = redis.get().await else {
        return false;
    };
    let key = format!("seen:{user_id}");
    conn.sismember::<_, _, bool>(&key, tweet_id).await.unwrap_or(false)
}
