// ═══════════════════════════════════════════════════════════════════════════════
// SCORING WEIGHTS & CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Dimension weights (sum = 1.0) — Phase 1 CTR Optimization
pub const W_D1_ENGAGEMENT_VELOCITY: f64 = 0.32; // +0.07: meilleur prédicteur CTR
pub const W_D2_CONTENT_INTELLIGENCE: f64 = 0.18; // -0.02
pub const W_D3_SOCIAL_GRAPH: f64 = 0.15;
pub const W_D4_TEMPORAL: f64 = 0.10;
pub const W_D5_BEHAVIORAL: f64 = 0.08; // -0.02
pub const W_D6_DIVERSITY: f64 = 0.06; // -0.02
pub const W_D7_VIRAL: f64 = 0.07;
pub const W_D8_PERSONALIZATION: f64 = 0.04; // -0.01

// ─── D1 : Engagement Velocity ───────────────────────────────────────────────

pub const ENG_VELOCITY_LIKE_WEIGHT: f64 = 1.0;
pub const ENG_VELOCITY_COMMENT_WEIGHT: f64 = 3.5;
pub const ENG_VELOCITY_RETWEET_WEIGHT: f64 = 5.0;
pub const ENG_VELOCITY_SHARE_WEIGHT: f64 = 4.0;
pub const ENG_VELOCITY_BOOKMARK_WEIGHT: f64 = 2.5;
pub const ENG_VELOCITY_VIEW_WEIGHT: f64 = 0.05;

pub const ENGAGEMENT_SIGMOID_SCALE: f64 = 200.0;
pub const VIRAL_THRESHOLD_ENG_PER_HOUR: f64 = 50.0;

pub const RECENCY_MULTIPLIER_1H: f64 = 3.0;
pub const RECENCY_MULTIPLIER_3H: f64 = 2.0;
pub const RECENCY_MULTIPLIER_6H: f64 = 1.5;

// ─── D2 : Content Intelligence ─────────────────────────────────────────────

pub const CONTENT_LENGTH_SHORT_THRESHOLD: f64 = 80.0;
pub const CONTENT_LENGTH_LONG_THRESHOLD: f64 = 200.0;
pub const CONTENT_LENGTH_GAUSSIAN_MU: f64 = 150.0;
pub const CONTENT_LENGTH_GAUSSIAN_SIGMA: f64 = 80.0;

pub const D2_LENGTH_WEIGHT: f64 = 0.25;
pub const D2_MEDIA_WEIGHT: f64 = 0.20;
pub const D2_HASHTAG_WEIGHT: f64 = 0.04; // per hashtag, max 0.12 (3 hashtags)
pub const D2_MENTION_WEIGHT: f64 = 0.03; // per mention, max 0.09
pub const D2_EMOJI_WEIGHT_ENTHUSIASTIC: f64 = 0.03; // max 0.10
pub const D2_EXCLAMATION_WEIGHT: f64 = 0.03; // max 0.06
pub const D2_QUESTION_WEIGHT_CURIOUS: f64 = 0.04; // max 0.10
pub const D2_URL_WEIGHT: f64 = 0.03; // varies by personality
pub const D2_KEYWORD_WEIGHT: f64 = 0.03; // per keyword match, max 0.15

// ─── D3 : Social Graph ──────────────────────────────────────────────────────

pub const D3_DIRECT_FOLLOW_BOOST: f64 = 0.55;
pub const D3_MUTUAL_FOLLOW_BOOST: f64 = 0.25;
pub const D3_SECOND_DEGREE_BOOST: f64 = 0.12;
pub const D3_PRIOR_AFFINITY_WEIGHT: f64 = 0.20; // max
pub const D3_AUTHOR_INFLUENCE_WEIGHT: f64 = 0.08;

// ─── D4 : Temporal Dynamics ────────────────────────────────────────────────

pub const RECENCY_HALF_LIFE_HOURS: f64 = 4.0; // Phase 1: 6h → 4h pour contenu plus frais
pub const RECENCY_DECAY_RATE: f64 = 0.173; // ln(2)/4h ≈ 0.173

pub const D4_RECENCY_WEIGHT: f64 = 0.45;
pub const D4_HOUR_MATCH_WEIGHT: f64 = 0.25;
pub const D4_DAY_MATCH_WEIGHT: f64 = 0.15;
pub const D4_MOMENTUM_WEIGHT: f64 = 0.15;

pub const MOMENTUM_THRESHOLD_2H: f64 = 2.0;
pub const MOMENTUM_THRESHOLD_6H: f64 = 6.0;
pub const MOMENTUM_ENGAGEMENT_SCALE: f64 = 20.0;

// ─── D5 : Behavioral Prediction ────────────────────────────────────────────

pub const D5_POWERUSER_BONUS: f64 = 0.20;
pub const D5_REGULAR_BONUS: f64 = 0.12;
pub const D5_CASUAL_BONUS: f64 = 0.05;

pub const D5_MEDIA_PREFERENCE_BONUS: f64 = 0.18;
pub const D5_LENGTH_MATCH_BONUS: f64 = 0.15;
pub const D5_RETWEET_PREDICTION_WEIGHT: f64 = 0.20; // max
pub const D5_ENGAGEMENT_TREND_THRESHOLD: f64 = 1.2;
pub const D5_ENGAGEMENT_TREND_BONUS: f64 = 0.10;
pub const D5_LOYALTY_WEIGHT: f64 = 0.12;

// ─── D6 : Content Diversity ────────────────────────────────────────────────

pub const D6_BASE_SCORE: f64 = 0.70;
pub const D6_FIRST_TWEET_SCORE: f64 = 0.80;
pub const D6_MEDIA_DIVERSITY_THRESHOLD: f64 = 0.4;
pub const D6_MEDIA_BONUS: f64 = 0.15;
pub const D6_HASHTAG_NOVELTY_BONUS: f64 = 0.10;
pub const D6_CONTENT_NOVELTY_BONUS: f64 = 0.05;

// ─── D7 : Viral Prediction ─────────────────────────────────────────────────

pub const D7_SHARE_RATIO_WEIGHT: f64 = 0.30;
pub const D7_CASCADE_WEIGHT: f64 = 0.25;
pub const D7_SPREAD_VELOCITY_WEIGHT: f64 = 0.25;
pub const D7_SHAREABILITY_WEIGHT: f64 = 0.20;

pub const CASCADE_RATIO_MAX: f64 = 3.0;
pub const VIRAL_SPREAD_THRESHOLD: f64 = 100.0;

pub const D7_MEDIA_SHAREABILITY: f64 = 0.3;
pub const D7_NO_MEDIA_SHAREABILITY: f64 = 0.1;
pub const D7_HASHTAG_SHAREABILITY: f64 = 0.1;
pub const D7_URL_SHAREABILITY: f64 = 0.1;

// ─── D8 : Personalization ──────────────────────────────────────────────────

pub const D8_AUTHOR_AFFINITY_WEIGHT: f64 = 0.40;
pub const D8_INTEREST_SCORE_WEIGHT: f64 = 0.35; // max
pub const D8_EMOTIONAL_BONUS: f64 = 0.10;
pub const D8_PEAK_HOUR_BONUS: f64 = 0.15;
pub const D8_EMOTIONAL_POSITIVITY_THRESHOLD: f64 = 0.6;

// ─── Modifiers ──────────────────────────────────────────────────────────────

pub const DIVERSITY_MULT_NONE: f64 = 1.00;
pub const DIVERSITY_MULT_2: f64 = 0.88;
pub const DIVERSITY_MULT_3: f64 = 0.72;
pub const DIVERSITY_MULT_4: f64 = 0.55;
pub const DIVERSITY_MULT_5: f64 = 0.38;
pub const DIVERSITY_MULT_MANY: f64 = 0.22;

pub const MODERATION_REPORT_MULTIPLIER: f64 = 3.0;
pub const MODERATION_PENALTY_MAX: f64 = -0.50;

pub const SOURCE_WEIGHT_SOCIAL_GRAPH: f64 = 0.08;
pub const SOURCE_WEIGHT_PERSONALIZED: f64 = 0.05;
pub const SOURCE_WEIGHT_VIRAL: f64 = 0.04;
pub const SOURCE_WEIGHT_INFLUENCER: f64 = 0.03;
pub const SOURCE_WEIGHT_TRENDING: f64 = 0.02;
pub const SOURCE_WEIGHT_TEMPORAL: f64 = 0.02;
pub const SOURCE_WEIGHT_DISCOVERY: f64 = 0.01;
pub const SOURCE_WEIGHT_QUALITY: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// DATABASE & CACHE
// ═══════════════════════════════════════════════════════════════════════════════

pub const DB_POOL_SIZE: u32 = 10;
pub const DB_STATEMENT_CACHE_SIZE: usize = 100;
pub const DB_QUERY_TIMEOUT_SECS: u64 = 30;

pub const CACHE_PROFILE_TTL_SECS: u64 = 300; // 5 minutes
pub const CACHE_RECOMMENDATIONS_TTL_MIN: u64 = 30;
pub const CACHE_RECOMMENDATIONS_TTL_MAX: u64 = 300;

// ─── Profile Cache TTL by user type ─────────────────────────────────────────

pub const CACHE_TTL_POWERUSER_BASE: u64 = 45;
pub const CACHE_TTL_REGULAR_BASE: u64 = 90;
pub const CACHE_TTL_CASUAL_BASE: u64 = 180;

// ─── Recommendation Cache TTL adjustments ────────────────────────────────────

pub const CACHE_TTL_TRENDING_MAX: u64 = 60;
pub const CACHE_TTL_DISCOVER_MAX: u64 = 120;
pub const CACHE_TTL_ENGAGEMENT_TREND_REDUCTION: u64 = 20; // if trend > 1.5

// ═══════════════════════════════════════════════════════════════════════════════
// API & LIMITS
// ═══════════════════════════════════════════════════════════════════════════════

pub const API_DEFAULT_LIMIT: usize = 50;
pub const API_MIN_LIMIT: usize = 1;
pub const API_MAX_LIMIT: usize = 200;
pub const API_DEFAULT_OFFSET: usize = 0;

pub const REQUEST_TIMEOUT_SECS: u64 = 60;
pub const RECOMMENDATION_REQUEST_TIMEOUT_SECS: u64 = 30;

// ═══════════════════════════════════════════════════════════════════════════════
// CANDIDATE COLLECTION
// ═══════════════════════════════════════════════════════════════════════════════

pub const CANDIDATE_LIMIT_TRENDING: usize = 600; // Phase 1: +50% (était 400)
pub const CANDIDATE_LIMIT_SOCIAL: usize = 300;
pub const CANDIDATE_LIMIT_VIRAL: usize = 250;
pub const CANDIDATE_LIMIT_DISCOVERY: usize = 150;
pub const CANDIDATE_LIMIT_TEMPORAL: usize = 150;
pub const CANDIDATE_LIMIT_INFLUENCER: usize = 150;
pub const CANDIDATE_LIMIT_PERSONALIZED: usize = 200;
pub const CANDIDATE_LIMIT_QUALITY: usize = 100;

pub const CANDIDATE_TOTAL_EXPECTED: usize =
    CANDIDATE_LIMIT_TRENDING
        + CANDIDATE_LIMIT_SOCIAL
        + CANDIDATE_LIMIT_VIRAL
        + CANDIDATE_LIMIT_DISCOVERY
        + CANDIDATE_LIMIT_TEMPORAL
        + CANDIDATE_LIMIT_INFLUENCER
        + CANDIDATE_LIMIT_PERSONALIZED
        + CANDIDATE_LIMIT_QUALITY;

// ─── Time windows for candidate collection ──────────────────────────────────

pub const WINDOW_TRENDING_FEED: &str = "12 hours";
pub const WINDOW_SOCIAL_FEED: &str = "72 hours";
pub const WINDOW_DISCOVER_FEED: &str = "48 hours";
pub const WINDOW_VIRAL_FEED: &str = "6 hours";

pub const WINDOW_TRENDING_TRENDING: &str = "6 hours";
pub const WINDOW_SOCIAL_TRENDING: &str = "24 hours";
pub const WINDOW_DISCOVER_TRENDING: &str = "48 hours";
pub const WINDOW_VIRAL_TRENDING: &str = "3 hours";

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILE BUILDING
// ═══════════════════════════════════════════════════════════════════════════════

pub const PROFILE_HISTORY_DAYS: i32 = 60;
pub const PROFILE_CONTENT_HISTORY_DAYS: i32 = 30;
pub const PROFILE_TOP_AUTHORS_LIMIT: usize = 20;
pub const PROFILE_TOP_WORDS_LIMIT: usize = 50;
pub const PROFILE_SEEN_TWEETS_LIMIT: usize = 500;
pub const PROFILE_SECOND_DEGREE_LIMIT: usize = 200;

pub const PROFILE_LIKE_COUNT_POWERUSER_THRESHOLD: i64 = 200;
pub const PROFILE_LIKE_COUNT_REGULAR_THRESHOLD: i64 = 30;

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

pub const MIN_USER_ID_LENGTH: usize = 36; // UUID
pub const MAX_CONTENT_LENGTH: usize = 280;
pub const MIN_SCORE: f64 = 0.0;
pub const MAX_SCORE: f64 = 1.0;

// ═══════════════════════════════════════════════════════════════════════════════
// LOGGING & DEBUGGING
// ═══════════════════════════════════════════════════════════════════════════════

pub const LOG_TOP_N_TWEETS: usize = 5;
pub const LOG_SAMPLE_SIZE: usize = 3;
