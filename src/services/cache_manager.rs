use anyhow::Result;
use redis::{aio::MultiplexedConnection, AsyncCommands, Client};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::warn;

use crate::models::UserProfile;

const TTL_SEEN_TWEETS: u64 = 86400;
const TTL_USER_PROFILE: u64 = 300;
const TWEET_SCORES_KEY: &str = "twitninf:tweet_scores";

#[derive(Clone)]
pub struct CacheManager {
    conn: Arc<Mutex<MultiplexedConnection>>,
}

impl CacheManager {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)?;
        let conn = client.get_multiplexed_tokio_connection().await?;
        Ok(CacheManager { conn: Arc::new(Mutex::new(conn)) })
    }

    // ─── Recommandations ─────────────────────────────────────────────────────

    pub async fn get_recommendations(&self, user_id: &str, mode: &str) -> Option<Vec<String>> {
        let key = format!("twitninf:reco:{}:{}", user_id, mode);
        let mut c = self.conn.lock().await;
        let json: Option<String> = c.get(&key).await.ok()?;
        json.and_then(|j| serde_json::from_str(&j).ok())
    }

    pub async fn set_recommendations_ttl(&self, user_id: &str, mode: &str, ids: &[String], ttl: u64) {
        let key = format!("twitninf:reco:{}:{}", user_id, mode);
        if let Ok(json) = serde_json::to_string(ids) {
            let mut c = self.conn.lock().await;
            let _: Result<(), _> = c.set_ex(&key, json, ttl).await;
        }
    }

    pub async fn invalidate_recommendations(&self, user_id: &str) {
        let mut c = self.conn.lock().await;
        for mode in &["feed", "discover", "trending", "for_you"] {
            let key = format!("twitninf:reco:{}:{}", user_id, mode);
            let _: Result<(), _> = c.del(&key).await;
        }
    }

    // ─── Profil utilisateur ───────────────────────────────────────────────────

    pub async fn get_profile(&self, key: &str) -> Option<UserProfile> {
        let mut c = self.conn.lock().await;
        let json: Option<String> = c.get(key).await.ok()?;
        json.and_then(|j| serde_json::from_str(&j).ok())
    }

    pub async fn set_profile(&self, key: &str, profile: &UserProfile) {
        if let Ok(json) = serde_json::to_string(profile) {
            let mut c = self.conn.lock().await;
            let _: Result<(), _> = c.set_ex(key, json, TTL_USER_PROFILE).await;
        }
    }

    // ─── Scores de tweets ─────────────────────────────────────────────────────

    pub async fn update_tweet_score(&self, tweet_id: &str, weight: f64) -> Result<f64> {
        let mut c = self.conn.lock().await;
        let new_score: f64 = c.zincr(TWEET_SCORES_KEY, tweet_id, weight).await?;
        Ok(new_score)
    }

    // ─── Tweets vus ───────────────────────────────────────────────────────────

    pub async fn mark_tweet_seen(&self, user_id: &str, tweet_id: &str) {
        let key = format!("twitninf:seen:{}", user_id);
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.sadd(&key, tweet_id).await;
        let _: Result<(), _> = c.expire(&key, TTL_SEEN_TWEETS as i64).await;
    }

    pub async fn get_seen_tweet_ids(&self, user_id: &str) -> Vec<String> {
        let key = format!("twitninf:seen:{}", user_id);
        let mut c = self.conn.lock().await;
        c.smembers(&key).await.unwrap_or_default()
    }

    pub async fn health_check(&self) -> bool {
        let mut c = self.conn.lock().await;
        redis::cmd("PING").query_async::<String>(&mut *c).await.is_ok()
    }
}
