use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use tokio::join;
use tracing::{info, warn, debug, trace};

use crate::algorithm::scoring::{compute_feed_metrics, score_tweet};
use crate::algorithm::trending::trending_score;
use crate::models::*;
use crate::services::cache_manager::CacheManager;

pub struct RecommenderService {
    pg: PgPool,
    cache: CacheManager,
}

impl RecommenderService {
    pub fn new(pg: PgPool, cache: CacheManager) -> Self {
        Self { pg, cache }
    }

    pub async fn recommend(&self, req: &RecommendRequest) -> Result<RecommendResponse> {
        let start = Instant::now();
        let mode = req.mode.clone().unwrap_or_default();
        let mode_str = mode_label(&mode);
        let limit = req.limit.unwrap_or(50).clamp(1, 200) as usize;
        let offset = req.offset.unwrap_or(0).max(0) as usize;
        let force_refresh = req.force_refresh.unwrap_or(false);

        debug!(user_id = %req.user_id, mode = mode_str, limit, offset, force_refresh, "━━━ RECOMMEND REQUEST ━━━");

        if !force_refresh {
            if let Some(cached) = self.cache.get_recommendations(&req.user_id, mode_str).await {
                let cached_total = cached.len();
                let page: Vec<String> = cached.into_iter().skip(offset).take(limit).collect();
                let count = page.len();
                debug!(cache_hit = true, cached_total, page_size = count, "Cache hit!");
                return Ok(self.build_empty_response(
                    &req.user_id, page, count, mode_str,
                    start.elapsed().as_millis() as u64, true,
                ));
            }
        }

        debug!("Building user profile...");
        let profile = self.build_user_profile(&req.user_id).await?;
        trace!(following_count = profile.following_ids.len(), top_authors = profile.top_authors.len(),
               user_type = ?profile.user_type, "User profile built");

        debug!("Collecting candidates from {} sources...", 8);
        let (sources, source_stats) = self.collect_candidates(&req.user_id, &profile, &mode).await?;
        let total_candidates = sources.len();
        debug!(total_candidates,
               trending = source_stats.trending, social_graph = source_stats.social_graph,
               viral = source_stats.viral, discovery = source_stats.discovery,
               temporal = source_stats.temporal, influencer = source_stats.influencer,
               personalized = source_stats.personalized, quality = source_stats.quality,
               "Candidates collected from 8 sources");

        if sources.is_empty() {
            warn!("No candidates found for user");
            return Ok(self.build_empty_response(
                &req.user_id, vec![], 0, mode_str,
                start.elapsed().as_millis() as u64, false,
            ));
        }

        debug!("Deduplicating {} candidates...", total_candidates);
        let deduped = deduplicate(sources);
        let deduped_count = deduped.len();
        debug!(deduped_count, removed = total_candidates - deduped_count, "Deduplication complete");

        debug!("Scoring {} tweets with 8 dimensions...", deduped_count);
        let scored = self.score_all(&deduped, &profile, &mode);
        debug!(scored_count = scored.len(), "Scoring complete");

        // Show top 5 scores
        let top_5: Vec<_> = scored.iter().take(5).map(|s| (&s.tweet_id, s.score)).collect();
        trace!("Top 5 scores: {:?}", top_5);

        let all_ids: Vec<String> = scored.iter().map(|s| s.tweet_id.clone()).collect();

        let tweet_map: HashMap<&str, &RawTweet> = deduped.iter().map(|t| (t.id.as_str(), t)).collect();
        let pairs: Vec<(&RawTweet, &ScoredTweet)> = scored.iter()
            .filter_map(|s| tweet_map.get(s.tweet_id.as_str()).map(|t| (*t, s)))
            .collect();

        debug!("Computing feed quality metrics...");
        let metrics = compute_feed_metrics(&pairs);
        debug!(diversity_score = metrics.diversity_score, freshness_score = metrics.freshness_score,
               relevance_score = metrics.relevance_score, viral_potential = metrics.viral_potential,
               novelty_score = metrics.novelty_score, "Feed metrics calculated");

        let adaptive_ttl = adaptive_ttl(&profile, &mode);
        debug!(ttl_seconds = adaptive_ttl, "Setting cache TTL");
        self.cache.set_recommendations_ttl(&req.user_id, mode_str, &all_ids, adaptive_ttl).await;

        let total_available = self.count_available(&req.user_id).await.unwrap_or(1000);
        let page_ids: Vec<String> = all_ids.into_iter().skip(offset).take(limit).collect();
        let count = page_ids.len();
        debug!(pagination_offset = offset, pagination_limit = limit, page_size = count, total_available, "Pagination applied");

        info!(
            user_id = %req.user_id, mode = mode_str,
            candidates = total_candidates, deduped = deduped_count,
            returned = count,
            latency_ms = start.elapsed().as_millis(),
            "NeuralRank recommendations computed"
        );

        Ok(RecommendResponse {
            success: true,
            user_id: req.user_id.clone(),
            tweet_ids: page_ids,
            count,
            algorithm: "NeuralRank Fusion",
            algorithm_version: "2.0.0 — 12 dimensions réelles",
            mode: mode_str.to_string(),
            latency_ms: start.elapsed().as_millis() as u64,
            cache_hit: false,
            metadata: RecommendMetadata {
                candidates_collected: total_candidates,
                sources: SourceStats {
                    deduplicated_total: deduped_count,
                    ..source_stats
                },
                user_profile: UserProfileSummary {
                    user_type: format!("{:?}", profile.user_type),
                    confidence: profile.profile_confidence,
                    personality: format!("{:?}", profile.personality_type),
                    engagement_velocity: profile.engagement_velocity,
                    engagement_trend: if profile.engagement_trend > 1.2 { "increasing".into() }
                        else if profile.engagement_trend < 0.8 { "decreasing".into() }
                        else { "stable".into() },
                    network_influence: profile.network_influence,
                    most_active_hour: profile.most_active_hour,
                    churn_risk: profile.churn_risk,
                },
                quality_metrics: QualityMetrics {
                    diversity_score: metrics.diversity_score,
                    freshness_score: metrics.freshness_score,
                    relevance_score: metrics.relevance_score,
                    viral_potential: metrics.viral_potential,
                    novelty_score: metrics.novelty_score,
                },
                pagination: Pagination {
                    limit: limit as i32,
                    offset: offset as i32,
                    has_more: offset + count < total_available as usize,
                    total_available,
                },
            },
        })
    }

    fn score_all(&self, tweets: &[RawTweet], profile: &UserProfile, mode: &RecommendMode) -> Vec<ScoredTweet> {
        let mut author_count: HashMap<String, u32> = HashMap::new();
        let mut scored_feed: Vec<ScoredTweet> = Vec::with_capacity(tweets.len());
        trace!(mode = ?mode, "Scoring all tweets with mode adjustments");

        for (idx, tweet) in tweets.iter().enumerate() {
            let ac = *author_count.get(&tweet.user_id).unwrap_or(&0);
            let mut s = score_tweet(tweet, profile, ac, &scored_feed);
            let base_score = s.score;

            match mode {
                RecommendMode::Trending => {
                    let ts = trending_score(tweet);
                    s.score = (s.score * 0.40 + ts * 0.60).clamp(0.0, 1.0);
                    trace!(tweet_id = %tweet.id, base_score, trending_score = ts, final_score = s.score, "Trending mode: blended 40% base + 60% trending");
                }
                RecommendMode::Discover => {
                    let mut multiplier = 1.0;
                    if profile.following_ids.contains(&tweet.user_id) {
                        multiplier *= 0.65;
                        trace!(tweet_id = %tweet.id, "Discover: user follows author, reducing score by 35%");
                    }
                    if tweet.source == TweetSource::Discovery {
                        multiplier *= 1.30;
                        trace!(tweet_id = %tweet.id, "Discover: from Discovery source, boosting by 30%");
                    }
                    s.score = (s.score * multiplier).min(1.0);
                }
                RecommendMode::Feed => {
                    let mut boost = 1.0;
                    if profile.following_ids.contains(&tweet.user_id) {
                        boost *= 1.30;
                        trace!(tweet_id = %tweet.id, "Feed: user follows author, boosting by 30%");
                    }
                    if profile.mutual_follow_ids.contains(&tweet.user_id) {
                        boost *= 1.15;
                        trace!(tweet_id = %tweet.id, "Feed: mutual follow, boosting by 15%");
                    }
                    s.score = (s.score * boost).min(1.0);
                }
                RecommendMode::ForYou => {
                    trace!(tweet_id = %tweet.id, "ForYou mode: no mode-specific adjustment");
                }
            }

            if idx < 3 {
                trace!(idx, tweet_id = %tweet.id, base_score, final_score = s.score, "Sample scored tweet");
            }

            *author_count.entry(tweet.user_id.clone()).or_insert(0) += 1;
            scored_feed.push(s);
        }

        debug!("Sorting {} scored tweets by final score...", scored_feed.len());
        scored_feed.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_scores: Vec<_> = scored_feed.iter().take(3).map(|s| (&s.tweet_id, s.score)).collect();
        debug!("Top 3 final scores: {:?}", top_scores);

        scored_feed
    }

    async fn build_user_profile(&self, user_id: &str) -> Result<UserProfile> {
        let cache_key = format!("twitninf:profile:{}", user_id);
        if let Some(cached) = self.cache.get_profile(&cache_key).await {
            debug!(user_id, "Profile loaded from cache");
            return Ok(cached);
        }

        debug!(user_id, "Building user profile from database...");
        let client = self.pg.get().await?;
        let uid = user_id;

        // Séquentiel sur une seule connexion (pas de vrai parallélisme possible sur un seul client PG)
        let social_res = client.query(
            &*format!("SELECT following_id::text, EXISTS(SELECT 1 FROM user_follows f2 WHERE f2.follower_id = following_id AND f2.following_id = '{uid}'::uuid) AS is_mutual FROM user_follows WHERE follower_id = '{uid}'::uuid LIMIT 1000"),
            &[]
        ).await;
        let engagement_res = client.query(
            &*format!("SELECT SUM(CASE WHEN created_at > NOW() - INTERVAL '1 day' THEN 1 ELSE 0 END) AS daily, SUM(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END) AS weekly FROM tweet_likes WHERE user_id = '{uid}'::uuid"),
            &[]
        ).await;
        let temporal_res = client.query(
            &*format!("SELECT EXTRACT(HOUR FROM created_at)::int AS h, EXTRACT(DOW FROM created_at)::int AS d, COUNT(*) AS cnt FROM tweet_likes WHERE user_id = '{uid}'::uuid AND created_at > NOW() - INTERVAL '60 days' GROUP BY h, d ORDER BY h, d"),
            &[]
        ).await;
        let content_pref_res = client.query(
            &*format!("SELECT AVG(LENGTH(t.content))::float8 AS avg_len, SUM(CASE WHEN t.media_urls IS NOT NULL AND t.media_urls != '[]'::jsonb THEN 1 ELSE 0 END)::float8 / GREATEST(COUNT(*), 1) AS media_ratio, AVG(COALESCE(jsonb_array_length(t.hashtags), 0))::float8 AS avg_hashtags FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id WHERE tl.user_id = '{uid}'::uuid AND tl.created_at > NOW() - INTERVAL '30 days'"),
            &[]
        ).await;
        let behavior_res = client.query(
            &*format!("SELECT (SELECT COUNT(*) FROM tweet_retweets WHERE user_id = '{uid}'::uuid) AS rt_count, (SELECT COUNT(*) FROM tweet_likes WHERE user_id = '{uid}'::uuid) AS like_count, (SELECT COUNT(*) FROM tweets WHERE user_id = '{uid}'::uuid AND parent_tweet_id IS NOT NULL) AS reply_count, (SELECT COUNT(*) FROM user_follows WHERE following_id = '{uid}'::uuid) AS followers, (SELECT COUNT(*) FROM user_follows WHERE follower_id = '{uid}'::uuid) AS following"),
            &[]
        ).await;
        let author_affinity_res = client.query(
            &*format!("SELECT t.user_id::text AS author_id, COUNT(*)::float8 AS affinity FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id WHERE tl.user_id = '{uid}'::uuid AND tl.created_at > NOW() - INTERVAL '60 days' GROUP BY t.user_id ORDER BY affinity DESC LIMIT 20"),
            &[]
        ).await;
        let seen_ids_res = client.query(
            &*format!("SELECT tweet_id::text FROM tweet_likes WHERE user_id = '{uid}'::uuid ORDER BY created_at DESC LIMIT 500"),
            &[]
        ).await;

        let mut profile = UserProfile::default();
        profile.user_id = user_id.to_string();

        if let Ok(rows) = social_res {
            for row in &rows {
                let fid: String = row.try_get(0).unwrap_or_default();
                let is_mutual: bool = row.try_get(1).unwrap_or(false);
                if !fid.is_empty() {
                    profile.following_ids.push(fid.clone());
                    if is_mutual { profile.mutual_follow_ids.push(fid); }
                }
            }
            trace!(following = profile.following_ids.len(), mutual = profile.mutual_follow_ids.len(), "Social graph loaded");
        }

        if let Ok(rows) = behavior_res {
            if let Some(row) = rows.first() {
                let like_count: i64  = row.try_get(1).unwrap_or(0);
                let follower_count: i64 = row.try_get(3).unwrap_or(0);
                let following_count: i64 = row.try_get(4).unwrap_or(0);

                profile.follower_count  = follower_count;
                profile.following_count = following_count;
                profile.network_influence = ((follower_count as f64).ln().max(0.0) * 10.0).min(100.0);

                profile.user_type = if like_count > 200 { UserType::PowerUser }
                    else if like_count > 30 { UserType::Regular }
                    else { UserType::Casual };

                profile.profile_confidence = (0.3 + (like_count as f64 / 400.0).min(0.7)).min(1.0);
                profile.churn_risk = 0.2;
                trace!(like_count, follower_count, following_count, user_type = ?profile.user_type, "Behavior metrics loaded");
            }
        }

        if let Ok(rows) = engagement_res {
            if let Some(row) = rows.first() {
                let daily: i64  = row.try_get(0).unwrap_or(0);
                let weekly: i64 = row.try_get(1).unwrap_or(0);
                let weekly_per_day = weekly as f64 / 7.0;
                profile.engagement_velocity = daily as f64;
                profile.engagement_trend = if weekly_per_day > 0.0 { daily as f64 / weekly_per_day } else { 1.0 };
                trace!(daily_engagement = daily, engagement_trend = profile.engagement_trend, "Engagement metrics loaded");
            }
        }

        if let Ok(rows) = temporal_res {
            let mut hourly = [0.0_f64; 24];
            let mut daily  = [0.0_f64; 7];
            for row in &rows {
                let h: i32 = row.try_get(0).unwrap_or(0);
                let d: i32 = row.try_get(1).unwrap_or(0);
                let cnt: i64 = row.try_get(2).unwrap_or(0);
                if (0..24).contains(&h) { hourly[h as usize] += cnt as f64; }
                if (0..7).contains(&d)  { daily[d as usize]  += cnt as f64; }
            }
            let h_max = hourly.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
            let d_max = daily.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
            for i in 0..24 { profile.hourly_activity[i] = hourly[i] / h_max; }
            for i in 0..7  { profile.daily_activity[i]  = daily[i]  / d_max; }
            profile.most_active_hour = hourly.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(12) as u32;
            trace!(most_active_hour = profile.most_active_hour, "Temporal activity patterns loaded");
        }

        if let Ok(rows) = content_pref_res {
            if let Some(row) = rows.first() {
                let avg_len: f64     = row.try_get(0).unwrap_or(100.0);
                let media_ratio: f64 = row.try_get(1).unwrap_or(0.3);
                let avg_ht: f64      = row.try_get(2).unwrap_or(1.0);
                profile.avg_content_length = avg_len;
                profile.prefers_media = media_ratio > 0.35;
                profile.avg_hashtag_count = avg_ht;
                profile.preferred_content_length = if avg_len < 80.0 { ContentLength::Short }
                    else if avg_len > 200.0 { ContentLength::Long }
                    else { ContentLength::Medium };
                trace!(avg_content_length = avg_len, media_ratio, content_preference = ?profile.preferred_content_length, "Content preferences loaded");
            }
        }

        if let Ok(rows) = author_affinity_res {
            let max_aff = rows.first()
                .and_then(|r| r.try_get::<_, f64>(1).ok())
                .unwrap_or(1.0).max(1.0);
            profile.top_authors = rows.iter()
                .filter_map(|r| {
                    let uid: String = r.try_get(0).ok()?;
                    let aff: f64   = r.try_get(1).ok()?;
                    Some((uid, aff / max_aff))
                })
                .collect();
            trace!(top_authors_count = profile.top_authors.len(), "Top authors affinity loaded");
        }

        if let Ok(rows) = seen_ids_res {
            profile.liked_tweet_ids = rows.iter()
                .filter_map(|r| r.try_get::<_, String>(0).ok())
                .collect();
            trace!(liked_tweets_count = profile.liked_tweet_ids.len(), "Liked tweet history loaded");
        }
        profile.seen_tweet_ids = self.cache.get_seen_tweet_ids(user_id).await;

        // Amis d'amis
        if let Ok(c2) = self.pg.get().await {
            if let Ok(rows) = c2.query(
                &*format!("SELECT DISTINCT f2.following_id::text FROM user_follows f JOIN user_follows f2 ON f2.follower_id = f.following_id WHERE f.follower_id = '{user_id}'::uuid AND f2.following_id != '{user_id}'::uuid AND f2.following_id NOT IN (SELECT following_id FROM user_follows WHERE follower_id = '{user_id}'::uuid) LIMIT 200"),
                &[]
            ).await {
                profile.second_degree_ids = rows.iter()
                    .filter_map(|r| r.try_get::<_, String>(0).ok())
                    .collect();
                trace!(second_degree_count = profile.second_degree_ids.len(), "Second degree network loaded");
            }
        }

        debug!(profile_confidence = profile.profile_confidence, "User profile built and cached");
        self.cache.set_profile(&cache_key, &profile).await;
        Ok(profile)
    }

    async fn collect_candidates(
        &self,
        user_id: &str,
        profile: &UserProfile,
        mode: &RecommendMode,
    ) -> Result<(Vec<RawTweet>, SourceStats)> {
        let (window_trending, window_social, window_discover, window_viral) = match mode {
            RecommendMode::Trending  => ("6 hours",  "24 hours", "48 hours", "3 hours"),
            RecommendMode::Feed      => ("12 hours", "72 hours", "48 hours", "6 hours"),
            RecommendMode::Discover  => ("24 hours", "48 hours", "96 hours", "12 hours"),
            RecommendMode::ForYou    => ("72 hours", "72 hours", "96 hours", "24 hours"),
        };
        debug!(mode = ?mode, trending_window = window_trending, social_window = window_social,
               discover_window = window_discover, viral_window = window_viral, "Collecting candidates with time windows");

        let following_list = &profile.following_ids;
        let active_hour = profile.most_active_hour as i32;

        // Formater une liste d'UUIDs pour une clause IN SQL
        let format_uuid_list = |ids: &[String]| -> String {
            if ids.is_empty() { return "'00000000-0000-0000-0000-000000000000'".to_string(); }
            ids.iter().map(|id| format!("'{}'", id)).collect::<Vec<_>>().join(",")
        };

        let following_sql = format_uuid_list(following_list);
        let top_authors_sql = format_uuid_list(
            &profile.top_authors.iter().take(10).map(|(id, _)| id.clone()).collect::<Vec<_>>()
        );

        // Source 1 : Trending
        let sql_trending = format!(
            "SELECT {COLS}, likes_1h, likes_6h, comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             LEFT JOIN LATERAL (SELECT COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') AS likes_1h, COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours') AS likes_6h FROM tweet_likes WHERE tweet_id = t.id) lk ON true \
             LEFT JOIN LATERAL (SELECT COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') AS comments_1h FROM tweets WHERE parent_tweet_id = t.id AND deleted_at IS NULL) cm ON true \
             WHERE {WHERE} AND t.created_at > NOW() - INTERVAL '{w}' AND t.user_id != '{uid}'::uuid \
             ORDER BY (COALESCE(t.view_count,0) + lk.likes_1h * 10) DESC LIMIT 400",
            COLS = TWEET_COLS, WHERE = WHERE_BASE, w = window_trending, uid = user_id
        );

        // Source 2 : Social graph
        let sql_social = if following_list.is_empty() {
            "SELECT 1 WHERE false".to_string()
        } else {
            format!(
                "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
                 FROM tweets t JOIN users u ON u.id = t.user_id \
                 WHERE {WHERE} AND t.user_id IN ({ids}) AND t.created_at > NOW() - INTERVAL '{w}' \
                 ORDER BY t.created_at DESC LIMIT 300",
                COLS = TWEET_COLS, WHERE = WHERE_BASE, ids = following_sql, w = window_social
            )
        };

        // Source 3 : Viral
        let sql_viral = format!(
            "SELECT {COLS}, likes_1h, likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             LEFT JOIN LATERAL (SELECT COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') AS likes_1h, COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours') AS likes_6h FROM tweet_likes WHERE tweet_id = t.id) lk ON true \
             WHERE {WHERE} AND t.created_at > NOW() - INTERVAL '{w}' AND t.user_id != '{uid}'::uuid \
             ORDER BY (lk.likes_1h * 10 + lk.likes_6h) DESC LIMIT 250",
            COLS = TWEET_COLS, WHERE = WHERE_BASE, w = window_viral, uid = user_id
        );

        // Source 4 : Discovery
        let exclude_following_sql = if following_list.is_empty() {
            String::new()
        } else {
            format!("AND t.user_id NOT IN ({})", following_sql)
        };
        let sql_discovery = format!(
            "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             WHERE {WHERE} AND t.user_id != '{uid}'::uuid AND t.created_at > NOW() - INTERVAL '{w}' {excl} \
             ORDER BY RANDOM() LIMIT 150",
            COLS = TWEET_COLS, WHERE = WHERE_BASE, w = window_discover, excl = exclude_following_sql, uid = user_id
        );

        // Source 5 : Temporal
        let sql_temporal = format!(
            "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             WHERE {WHERE} AND t.user_id != '{uid}'::uuid AND t.created_at > NOW() - INTERVAL '72 hours' \
             AND EXTRACT(HOUR FROM t.created_at) BETWEEN {h_lo} AND {h_hi} \
             ORDER BY t.created_at DESC LIMIT 150",
            COLS = TWEET_COLS, WHERE = WHERE_BASE,
            h_lo = (active_hour - 1).max(0), h_hi = (active_hour + 1).min(23), uid = user_id
        );

        // Source 6 : Influenceurs (verified or premium)
        let sql_influencer = format!(
            "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             WHERE {WHERE} AND t.user_id != '{uid}'::uuid AND t.created_at > NOW() - INTERVAL '48 hours' \
             AND (u.verified = true OR u.premium = true) \
             ORDER BY t.created_at DESC LIMIT 150",
            COLS = TWEET_COLS, WHERE = WHERE_BASE, uid = user_id
        );

        // Source 7 : Personnalisé
        let sql_personalized = if profile.top_authors.is_empty() {
            "SELECT 1 WHERE false".to_string()
        } else {
            format!(
                "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
                 FROM tweets t JOIN users u ON u.id = t.user_id \
                 WHERE {WHERE} AND t.user_id IN ({ids}) AND t.created_at > NOW() - INTERVAL '7 days' \
                 ORDER BY t.created_at DESC LIMIT 200",
                COLS = TWEET_COLS, WHERE = WHERE_BASE, ids = top_authors_sql
            )
        };

        // Source 8 : Qualité
        let sql_quality = format!(
            "SELECT {COLS}, 0::bigint AS likes_1h, 0::bigint AS likes_6h, 0::bigint AS comments_1h, 0::bigint AS retweets_1h \
             FROM tweets t JOIN users u ON u.id = t.user_id \
             WHERE {WHERE} AND t.user_id != '{uid}'::uuid AND t.created_at > NOW() - INTERVAL '72 hours' \
             AND (u.verified = true OR u.premium = true) \
             ORDER BY t.created_at DESC LIMIT 100",
            COLS = TWEET_COLS, WHERE = WHERE_BASE, uid = user_id
        );

        debug!("Running 8 parallel database queries for candidate sources...");
        let (r1, r2, r3, r4, r5, r6, r7, r8) = join!(
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_trending.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Trending, 0.15),
                        Err(e) => { warn!("Trending error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                if following_list.is_empty() { return vec![]; }
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_social.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::SocialGraph, 0.12),
                        Err(e) => { warn!("Social error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_viral.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Viral, 0.08),
                        Err(e) => { warn!("Viral error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_discovery.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Discovery, 0.05),
                        Err(e) => { warn!("Discovery error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_temporal.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Temporal, 0.06),
                        Err(e) => { warn!("Temporal error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_influencer.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Influencer, 0.04),
                        Err(e) => { warn!("Influencer error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                if profile.top_authors.is_empty() { return vec![]; }
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_personalized.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Personalized, 0.10),
                        Err(e) => { warn!("Personalized error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
            async {
                match self.pg.get().await {
                    Ok(c) => match c.query(sql_quality.as_str(), &[]).await {
                        Ok(rows) => map_rows(rows, TweetSource::Quality, 0.02),
                        Err(e) => { warn!("Quality error: {}", e); vec![] }
                    },
                    Err(_) => vec![]
                }
            },
        );

        let stats = SourceStats {
            trending: r1.len(), social_graph: r2.len(), viral: r3.len(),
            discovery: r4.len(), temporal: r5.len(), influencer: r6.len(),
            personalized: r7.len(), quality: r8.len(), deduplicated_total: 0,
        };

        trace!(
            trending = stats.trending, social_graph = stats.social_graph,
            viral = stats.viral, discovery = stats.discovery,
            temporal = stats.temporal, influencer = stats.influencer,
            personalized = stats.personalized, quality = stats.quality,
            "All 8 sources completed"
        );

        let mut all = Vec::with_capacity(
            stats.trending + stats.social_graph + stats.viral + stats.discovery
            + stats.temporal + stats.influencer + stats.personalized + stats.quality
        );
        all.extend(r1); all.extend(r2); all.extend(r3); all.extend(r4);
        all.extend(r5); all.extend(r6); all.extend(r7); all.extend(r8);

        debug!("Merged {} candidates from 8 sources", all.len());
        Ok((all, stats))
    }

    async fn count_available(&self, user_id: &str) -> Result<i64> {
        let client = self.pg.get().await?;
        let row = client.query_one(
            &*format!("SELECT COUNT(*) FROM tweets t WHERE t.deleted_at IS NULL AND t.moderation_status = 'approved' AND t.user_id != '{user_id}'::uuid AND t.parent_tweet_id IS NULL"),
            &[]
        ).await?;
        Ok(row.get(0))
    }

    fn build_empty_response(
        &self, user_id: &str, tweet_ids: Vec<String>, count: usize,
        mode: &str, latency_ms: u64, cache_hit: bool,
    ) -> RecommendResponse {
        RecommendResponse {
            success: true, user_id: user_id.to_string(), tweet_ids, count,
            algorithm: "NeuralRank Fusion",
            algorithm_version: "2.0.0 — 12 dimensions réelles",
            mode: mode.to_string(), latency_ms, cache_hit,
            metadata: RecommendMetadata {
                candidates_collected: 0,
                sources: SourceStats::default(),
                user_profile: UserProfileSummary {
                    user_type: "cached".into(), confidence: 1.0,
                    personality: "cached".into(), engagement_velocity: 0.0,
                    engagement_trend: "cached".into(), network_influence: 0.0,
                    most_active_hour: 12, churn_risk: 0.0,
                },
                quality_metrics: QualityMetrics {
                    diversity_score: 0.0, freshness_score: 0.0,
                    relevance_score: 0.0, viral_potential: 0.0, novelty_score: 0.0,
                },
                pagination: Pagination { limit: 50, offset: 0, has_more: false, total_available: 0 },
            },
        }
    }
}

// ─── SQL constants ────────────────────────────────────────────────────────────

const TWEET_COLS: &str = r#"
    t.id::text,
    t.user_id::text,
    COALESCE(t.content, '') AS content,
    t.created_at,
    COALESCE(t.view_count, 0) AS view_count,
    (SELECT COUNT(*)::bigint FROM tweet_likes WHERE tweet_id = t.id) AS like_count,
    (SELECT COUNT(*)::bigint FROM tweets rep WHERE rep.parent_tweet_id = t.id AND rep.deleted_at IS NULL) AS comment_count,
    (SELECT COUNT(*)::bigint FROM tweet_retweets WHERE tweet_id = t.id) AS retweet_count,
    0::bigint AS share_count,
    0::bigint AS bookmark_count,
    0::bigint AS report_count,
    (t.media_urls IS NOT NULL AND t.media_urls != '[]'::jsonb) AS has_media,
    COALESCE(jsonb_array_length(t.hashtags), 0)::int AS hashtag_count,
    COALESCE(jsonb_array_length(t.mentions), 0)::int AS mention_count,
    LENGTH(COALESCE(t.content, ''))::int AS content_length,
    0::bigint AS author_followers,
    0::bigint AS author_following,
    COALESCE(u.verified, false) AS author_is_verified,
    COALESCE(u.premium, false) AS author_is_premium,
    COALESCE(t.moderation_status::text, 'pending') AS moderation_status,
    t.recommendation_group::text
"#;

const WHERE_BASE: &str = r#"
    t.deleted_at IS NULL
    AND t.moderation_status = 'approved'
    AND t.parent_tweet_id IS NULL
"#;

// ─── Mapping rows → RawTweet ──────────────────────────────────────────────────

fn map_rows(rows: Vec<tokio_postgres::Row>, source: TweetSource, weight: f64) -> Vec<RawTweet> {
    rows.into_iter().filter_map(|r| {
        let id: String      = r.try_get(0).ok()?;
        let user_id: String = r.try_get(1).ok()?;
        let content: String = r.try_get(2).unwrap_or_default();

        let content_lower = content.to_lowercase();
        let emoji_count       = content.chars().filter(|c| *c as u32 > 0x1F000).count() as i32;
        let exclamation_count = content.matches('!').count() as i32;
        let question_count    = content.matches('?').count() as i32;
        let url_count         = content.matches("http").count() as i32;
        let words: Vec<String> = content_lower.split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .take(50)
            .collect();

        Some(RawTweet {
            id, user_id, content,
            created_at:        r.try_get(3).ok()?,
            view_count:        r.try_get(4).unwrap_or(0),
            like_count:        r.try_get(5).unwrap_or(0),
            comment_count:     r.try_get(6).unwrap_or(0),
            retweet_count:     r.try_get(7).unwrap_or(0),
            share_count:       r.try_get(8).unwrap_or(0),
            bookmark_count:    r.try_get(9).unwrap_or(0),
            report_count:      r.try_get(10).unwrap_or(0),
            has_media:         r.try_get(11).unwrap_or(false),
            hashtag_count:     r.try_get(12).unwrap_or(0),
            mention_count:     r.try_get(13).unwrap_or(0),
            content_length:    r.try_get(14).unwrap_or(0),
            author_followers:  r.try_get(15).unwrap_or(0),
            author_following:  r.try_get(16).unwrap_or(0),
            author_is_verified: r.try_get(17).unwrap_or(false),
            author_is_premium:  r.try_get(18).unwrap_or(false),
            moderation_status:  r.try_get(19).unwrap_or_else(|_| "approved".into()),
            recommendation_group: r.try_get(20).ok().flatten(),
            // Engagement récent (colonnes 21-24 injectées par les LATERAL joins)
            likes_1h:    r.try_get(21).unwrap_or(0),
            likes_6h:    r.try_get(22).unwrap_or(0),
            comments_1h: r.try_get(23).unwrap_or(0),
            retweets_1h: r.try_get(24).unwrap_or(0),
            emoji_count, exclamation_count, question_count, url_count, words,
            author_account_age_days: 365,
            author_tweet_count: 0,
            source, source_weight: weight,
        })
    }).collect()
}

fn deduplicate(mut tweets: Vec<RawTweet>) -> Vec<RawTweet> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut result: Vec<RawTweet> = Vec::new();

    for tweet in tweets.drain(..) {
        match seen.get(&tweet.id) {
            None => {
                seen.insert(tweet.id.clone(), result.len());
                result.push(tweet);
            }
            Some(&idx) => {
                if tweet.source_weight > result[idx].source_weight {
                    result[idx].source_weight = tweet.source_weight;
                    result[idx].source = tweet.source;
                }
            }
        }
    }
    result
}

fn mode_label(mode: &RecommendMode) -> &'static str {
    match mode {
        RecommendMode::Feed     => "feed",
        RecommendMode::Discover => "discover",
        RecommendMode::Trending => "trending",
        RecommendMode::ForYou   => "for_you",
    }
}

fn adaptive_ttl(profile: &UserProfile, mode: &RecommendMode) -> u64 {
    let mut ttl = match profile.user_type {
        UserType::PowerUser => 45_u64,
        UserType::Regular   => 90_u64,
        UserType::Casual    => 180_u64,
    };
    if profile.engagement_trend > 1.5 { ttl = ttl.saturating_sub(20); }
    if *mode == RecommendMode::Trending { ttl = ttl.min(60); }
    if *mode == RecommendMode::Discover { ttl = ttl.min(120); }
    ttl.max(30)
}
