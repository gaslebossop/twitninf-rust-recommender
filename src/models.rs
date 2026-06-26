use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::shadowban::ShadowbanLevel;

// ─── Requêtes entrantes ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RecommendRequest {
    pub user_id: String,   // UUID
    pub limit: Option<i32>,
    pub offset: Option<i32>,
    pub mode: Option<RecommendMode>,
    pub exclude_seen: Option<bool>,
    pub force_refresh: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RecommendMode {
    Feed,
    Discover,
    Trending,
    ForYou,
}

impl Default for RecommendMode {
    fn default() -> Self { RecommendMode::ForYou }
}

#[derive(Debug, Deserialize)]
pub struct TrackInteractionRequest {
    pub user_id: String,   // UUID
    pub tweet_id: String,  // UUID
    pub interaction_type: InteractionType,
    pub dwell_ms: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionType {
    Like, Unlike, Comment, Retweet, Unretweet,
    Share, View, Bookmark, ProfileView, Skip, Report, Block,
}

impl InteractionType {
    pub fn weight(self) -> f64 {
        match self {
            InteractionType::Like        =>  1.0,
            InteractionType::Unlike      => -1.0,
            InteractionType::Comment     =>  3.5,
            InteractionType::Retweet     =>  5.0,
            InteractionType::Unretweet   => -2.0,
            InteractionType::Share       =>  4.0,
            InteractionType::Bookmark    =>  2.5,
            InteractionType::View        =>  0.2,
            InteractionType::ProfileView =>  1.5,
            InteractionType::Skip        => -0.5,
            InteractionType::Report      => -12.0,
            InteractionType::Block       => -20.0,
        }
    }
}

// ─── Profil utilisateur ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub follower_count: i64,
    pub following_count: i64,
    pub network_influence: f64,

    pub following_ids: Vec<String>,
    pub mutual_follow_ids: Vec<String>,
    pub second_degree_ids: Vec<String>,

    pub liked_tweet_ids: Vec<String>,
    pub retweeted_tweet_ids: Vec<String>,
    pub replied_to_tweet_ids: Vec<String>,
    pub bookmarked_tweet_ids: Vec<String>,

    pub top_authors: Vec<(String, f64)>,

    pub hourly_activity: [f64; 24],
    pub daily_activity: [f64; 7],
    pub most_active_hour: u32,
    pub most_active_day: u32,

    pub avg_content_length: f64,
    pub prefers_media: bool,
    pub avg_hashtag_count: f64,
    pub preferred_content_length: ContentLength,

    pub engagement_velocity: f64,
    pub engagement_trend: f64,
    pub personality_type: PersonalityType,
    pub emotional_positivity: f64,

    pub top_words: Vec<(String, u32)>,

    pub churn_risk: f64,
    pub lifetime_value: f64,

    pub seen_tweet_ids: Vec<String>,

    pub user_type: UserType,
    pub profile_confidence: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum ContentLength { Short, #[default] Medium, Long }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PersonalityType {
    Enthusiastic, Curious, Thoughtful, #[default] Balanced,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum UserType { PowerUser, Regular, #[default] Casual }

// ─── Tweet candidat brut ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RawTweet {
    pub id: String,
    pub user_id: String,
    pub content: String,
    pub created_at: DateTime<Utc>,

    pub view_count: i64,
    pub like_count: i64,
    pub comment_count: i64,
    pub retweet_count: i64,
    pub share_count: i64,
    pub bookmark_count: i64,
    pub report_count: i64,

    pub likes_1h: i64,
    pub likes_6h: i64,
    pub comments_1h: i64,
    pub retweets_1h: i64,

    pub has_media: bool,
    pub hashtag_count: i32,
    pub mention_count: i32,
    pub content_length: i32,
    pub emoji_count: i32,
    pub exclamation_count: i32,
    pub question_count: i32,
    pub url_count: i32,
    pub words: Vec<String>,

    pub author_followers: i64,
    pub author_following: i64,
    pub author_is_verified: bool,
    pub author_is_premium: bool,
    pub author_account_age_days: i32,
    pub author_tweet_count: i64,

    pub moderation_status: String,
    pub recommendation_group: Option<String>,

    pub source: TweetSource,
    pub source_weight: f64,

    pub author_shadowban_level: ShadowbanLevel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TweetSource {
    Trending, SocialGraph, Personalized, Viral,
    Temporal, Discovery, Influencer, Quality,
}

// ─── Score ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ScoreBreakdown {
    pub engagement_velocity: f64,
    pub content_intelligence: f64,
    pub social_graph_dynamics: f64,
    pub temporal_dynamics: f64,
    pub behavioral_prediction: f64,
    pub content_diversity: f64,
    pub viral_prediction: f64,
    pub personalization_depth: f64,

    pub engagement_velocity_raw: f64,
    pub engagement_acceleration: f64,
    pub viral_velocity: f64,

    pub direct_follow_boost: f64,
    pub mutual_follow_boost: f64,
    pub second_degree_boost: f64,

    pub diversity_multiplier: f64,
    pub moderation_penalty: f64,
    pub source_weight: f64,
    pub shadowban_multiplier: f64,

    pub final_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredTweet {
    pub tweet_id: String,
    pub score: f64,
    pub breakdown: ScoreBreakdown,
}

// ─── Réponses API ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct RecommendResponse {
    pub success: bool,
    pub user_id: String,
    pub tweet_ids: Vec<String>,
    pub count: usize,
    pub algorithm: &'static str,
    pub algorithm_version: &'static str,
    pub mode: String,
    pub latency_ms: u64,
    pub cache_hit: bool,
    pub metadata: RecommendMetadata,
}

#[derive(Debug, Serialize)]
pub struct RecommendMetadata {
    pub candidates_collected: usize,
    pub sources: SourceStats,
    pub user_profile: UserProfileSummary,
    pub quality_metrics: QualityMetrics,
    pub pagination: Pagination,
}

#[derive(Debug, Serialize, Default)]
pub struct SourceStats {
    pub trending: usize,
    pub social_graph: usize,
    pub personalized: usize,
    pub viral: usize,
    pub temporal: usize,
    pub discovery: usize,
    pub influencer: usize,
    pub quality: usize,
    pub deduplicated_total: usize,
}

#[derive(Debug, Serialize)]
pub struct UserProfileSummary {
    pub user_type: String,
    pub confidence: f64,
    pub personality: String,
    pub engagement_velocity: f64,
    pub engagement_trend: String,
    pub network_influence: f64,
    pub most_active_hour: u32,
    pub churn_risk: f64,
}

#[derive(Debug, Serialize)]
pub struct QualityMetrics {
    pub diversity_score: f64,
    pub freshness_score: f64,
    pub relevance_score: f64,
    pub viral_potential: f64,
    pub novelty_score: f64,
}

#[derive(Debug, Serialize)]
pub struct Pagination {
    pub limit: i32,
    pub offset: i32,
    pub has_more: bool,
    pub total_available: i64,
}

#[derive(Debug, Serialize)]
pub struct TrackResponse {
    pub success: bool,
    pub tweet_id: String,
    pub user_id: String,
    pub new_score: f64,
    pub weight_applied: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: &'static str,
    pub db: String,
    pub redis: String,
    pub uptime_secs: u64,
    pub algorithm: &'static str,
}
