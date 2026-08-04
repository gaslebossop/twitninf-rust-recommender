use anyhow::{bail, Result};
use redis::{aio::MultiplexedConnection, AsyncCommands, Client};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::warn;

use crate::models::{FeedEntry, UserProfile};

const TTL_SEEN_TWEETS: u64 = 86400;
const TTL_USER_PROFILE: u64 = 300;
const TWEET_SCORES_KEY: &str = "twitninf:tweet_scores";
const MAX_TWEET_SCORE: f64 = 500.0; // plafond absolu pour éviter l'inflation Redis

/// Impressions en attente d'attribution, triées par timestamp d'exposition.
const CTR_PENDING_KEY: &str = "twitninf:ctr:pending";
/// Au-delà de ce délai, une impression sans engagement devient un exemple négatif.
pub const CTR_ATTRIBUTION_SECS: u64 = 1800;
/// TTL des features stockées : marge au-dessus de la fenêtre d'attribution.
const TTL_CTR_FEATURES: u64 = CTR_ATTRIBUTION_SECS + 600;

#[derive(Clone)]
pub struct CacheManager {
    pub(crate) conn: Arc<Mutex<MultiplexedConnection>>,
}

impl CacheManager {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)?;
        let conn = client.get_multiplexed_tokio_connection().await?;
        Ok(CacheManager { conn: Arc::new(Mutex::new(conn)) })
    }

    // ─── Recommandations ─────────────────────────────────────────────────────

    /// ⚠ Deux formats cohabitent dans Redis pendant la durée d'un TTL.
    ///
    /// La clé contenait une simple liste d'identifiants ; elle contient
    /// désormais des `FeedEntry`, qui portent en plus le lien de fil. Au
    /// déploiement, les clés déjà écrites par la version précédente restent
    /// valides jusqu'à 180 s — les lire en `FeedEntry` échoue, et sans ce repli
    /// chaque utilisateur ayant un feed en cache aurait vu son fil recalculé
    /// intégralement à sa prochaine requête. On les relit donc à l'ancienne,
    /// sans lien de fil : elles ne contenaient de toute façon aucune réponse.
    pub async fn get_recommendations(&self, user_id: &str, mode: &str) -> Option<Vec<FeedEntry>> {
        let key = format!("twitninf:reco:{}:{}", user_id, mode);
        let mut c = self.conn.lock().await;
        let json: Option<String> = c.get(&key).await.ok()?;
        let json = json?;

        if let Ok(entries) = serde_json::from_str::<Vec<FeedEntry>>(&json) {
            return Some(entries);
        }
        serde_json::from_str::<Vec<String>>(&json).ok().map(|ids| {
            ids.into_iter().map(|id| FeedEntry { id, parent_id: None }).collect()
        })
    }

    pub async fn set_recommendations_ttl(&self, user_id: &str, mode: &str, entries: &[FeedEntry], ttl: u64) {
        let key = format!("twitninf:reco:{}:{}", user_id, mode);
        if let Ok(json) = serde_json::to_string(entries) {
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
        // Valider que tweet_id est un UUID avant de l'utiliser comme membre Redis
        if uuid::Uuid::parse_str(tweet_id).is_err() {
            bail!("tweet_id must be a valid UUID");
        }
        let mut c = self.conn.lock().await;
        let new_score: f64 = c.zincr(TWEET_SCORES_KEY, tweet_id, weight).await?;
        // Plafonner le score pour éviter l'inflation illimitée via fausses interactions
        if new_score > MAX_TWEET_SCORE {
            let _: Result<(), _> = c.zadd(TWEET_SCORES_KEY, tweet_id, MAX_TWEET_SCORE).await;
            return Ok(MAX_TWEET_SCORE);
        }
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

    // ─── Impressions CTR ──────────────────────────────────────────────────────
    //
    // Le modèle CTR ne peut apprendre que si on l'entraîne sur le vecteur de
    // features réellement utilisé pour classer le tweet. On mémorise donc ce
    // vecteur au moment où le feed est servi, et on le rejoue au moment où le
    // lecteur réagit (ou pas). Sans ça, l'entraînement se fait sur des features
    // inventées côté tracking et le modèle apprend du bruit.

    fn imp_key(user_id: &str, tweet_id: &str) -> String {
        format!("twitninf:ctr:imp:{}:{}", user_id, tweet_id)
    }

    /// Enregistre les features d'une impression servie et la met en attente
    /// d'attribution. Silencieux en cas d'échec : le feed prime sur le ML.
    pub async fn record_impression(&self, user_id: &str, tweet_id: &str, features: &[f64]) {
        if uuid::Uuid::parse_str(user_id).is_err() || uuid::Uuid::parse_str(tweet_id).is_err() {
            return;
        }
        let Ok(json) = serde_json::to_string(features) else { return };
        let now = now_secs();
        let key = Self::imp_key(user_id, tweet_id);
        let member = format!("{}:{}", user_id, tweet_id);

        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.set_ex(&key, json, TTL_CTR_FEATURES).await;
        let _: Result<(), _> = c.zadd(CTR_PENDING_KEY, member, now as f64).await;
    }

    /// Récupère les features d'une impression et la retire de la file
    /// d'attribution : l'engagement observé tranche, plus besoin du timeout.
    pub async fn take_impression(&self, user_id: &str, tweet_id: &str) -> Option<Vec<f64>> {
        let key = Self::imp_key(user_id, tweet_id);
        let member = format!("{}:{}", user_id, tweet_id);

        let mut c = self.conn.lock().await;
        let json: Option<String> = c.get(&key).await.ok()?;
        let _: Result<(), _> = c.del(&key).await;
        let _: Result<(), _> = c.zrem(CTR_PENDING_KEY, member).await;
        json.and_then(|j| serde_json::from_str::<Vec<f64>>(&j).ok())
    }

    /// Retire et retourne les impressions expirées sans engagement : ce sont les
    /// exemples négatifs du modèle CTR.
    pub async fn drain_expired_impressions(&self, limit: usize) -> Vec<(String, String, Vec<f64>)> {
        let cutoff = now_secs().saturating_sub(CTR_ATTRIBUTION_SECS);

        let members: Vec<String> = {
            let mut c = self.conn.lock().await;
            match c
                .zrangebyscore_limit(CTR_PENDING_KEY, 0f64, cutoff as f64, 0, limit as isize)
                .await
            {
                Ok(m) => m,
                Err(e) => {
                    warn!("CTR sweep: lecture de la file impossible: {e}");
                    return Vec::new();
                }
            }
        };

        let mut out = Vec::with_capacity(members.len());
        for member in members {
            // member = "<user_uuid>:<tweet_uuid>" — les UUID contiennent des
            // tirets mais pas de deux-points, la coupe est donc sans ambiguïté.
            let Some((user_id, tweet_id)) = member.split_once(':') else { continue };
            let key = Self::imp_key(user_id, tweet_id);

            let mut c = self.conn.lock().await;
            let json: Option<String> = c.get(&key).await.unwrap_or(None);
            let _: Result<(), _> = c.del(&key).await;
            let _: Result<(), _> = c.zrem(CTR_PENDING_KEY, &member).await;
            drop(c);

            if let Some(f) = json.and_then(|j| serde_json::from_str::<Vec<f64>>(&j).ok()) {
                out.push((user_id.to_string(), tweet_id.to_string(), f));
            }
        }
        out
    }

    /// Nombre d'impressions actuellement en attente d'attribution.
    pub async fn pending_impressions(&self) -> u64 {
        let mut c = self.conn.lock().await;
        c.zcard(CTR_PENDING_KEY).await.unwrap_or(0)
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
