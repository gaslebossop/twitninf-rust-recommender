use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Bidding ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BidType {
    Cpm,  // cost per mille impressions
    Cpc,  // cost per click
    Cpa,  // cost per action (app install, sign-up…)
}

// ─── Campaign lifecycle ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CampaignStatus { Active, Paused, Exhausted, Ended }

// ─── Targeting building blocks ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterestCategory {
    Technology, Sports, Music, Gaming, Fashion, Food,
    Finance, Politics, Entertainment, Health, Science,
    Travel, Education, Automotive, Business,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudienceSegment {
    All,
    PowerUsers,     // high engagement, tech-savvy
    CasualUsers,    // light users, wider reach
    HighValue,      // lifetime_value > 50
    Churning,       // churn_risk > 0.6 — re-engagement campaigns
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTargeting {
    pub hours: Vec<u8>,     // 0–23; empty = all hours
    pub weekdays: Vec<u8>,  // 0=Mon … 6=Sun; empty = all days
}

// ─── Per-ad targeting specification ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdTargetingSpec {
    pub keywords: Vec<String>,           // matched against user.top_words
    pub negative_keywords: Vec<String>,  // exclude user if any hit
    pub interests: Vec<InterestCategory>,
    pub audience_segments: Vec<AudienceSegment>,
    pub min_ltv: Option<f64>,            // skip users below this LTV
    pub max_churn_risk: Option<f64>,     // skip users above this churn score
    pub prefers_media: Option<bool>,     // target media-preferring users
    pub time_targeting: Option<TimeTargeting>,
    pub frequency_cap_daily: u32,        // max impressions per user per day
    pub frequency_cap_weekly: u32,
}

impl Default for AdTargetingSpec {
    fn default() -> Self {
        Self {
            keywords: vec![],
            negative_keywords: vec![],
            interests: vec![],
            audience_segments: vec![AudienceSegment::All],
            min_ltv: None,
            max_churn_risk: None,
            prefers_media: None,
            time_targeting: None,
            frequency_cap_daily: 5,
            frequency_cap_weekly: 20,
        }
    }
}

// ─── Ad creative ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ad {
    pub id: String,
    pub campaign_id: String,
    pub advertiser_id: String,
    pub tweet_id: String,       // tweet used as the sponsored content
    pub bid_type: BidType,
    pub bid_amount: f64,        // USD — per 1000 impressions (CPM) or per click (CPC)
    pub targeting: AdTargetingSpec,
    pub created_at: DateTime<Utc>,
}

// ─── Campaign ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdCampaign {
    pub id: String,
    pub advertiser_id: String,
    pub name: String,
    pub status: CampaignStatus,
    pub total_budget: f64,
    pub spent: f64,
    pub daily_budget: f64,
    pub spent_today: f64,
    pub start_at: DateTime<Utc>,
    pub end_at: Option<DateTime<Utc>>,
    pub ads: Vec<Ad>,
}

impl AdCampaign {
    pub fn budget_remaining(&self) -> f64 { self.total_budget - self.spent }
    pub fn daily_remaining(&self) -> f64 { self.daily_budget - self.spent_today }

    pub fn is_active(&self) -> bool {
        self.status == CampaignStatus::Active
            && self.budget_remaining() > 0.0
            && self.daily_remaining() > 0.0
            && self.end_at.map_or(true, |end| end > Utc::now())
    }
}

// ─── Scored ad ────────────────────────────────────────────────────────────────

/// Why this ad was selected for this user — for debugging and audit logs.
#[derive(Debug, Clone, Serialize)]
pub struct MatchReason {
    pub signal: String,
    pub weight: f64,
    pub score: f64,
}

/// An ad that has been scored and is ready for auction and injection.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredAd {
    pub ad_id: String,
    pub campaign_id: String,
    pub tweet_id: String,
    pub relevance_score: f64,   // 0–1  keyword/interest fit (shown to advertisers)
    pub quality_score: f64,     // 0–1  overall quality for this specific user
    pub effective_bid: f64,     // quality-adjusted bid used in second-price auction
    pub match_reasons: Vec<MatchReason>,
}

// ─── Feed assembly ────────────────────────────────────────────────────────────

/// A slot in the assembled feed — either organic tweet or sponsored.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FeedItem {
    Tweet { tweet_id: String },
    Ad    { ad: ScoredAd },
}

impl FeedItem {
    pub fn is_ad(&self) -> bool { matches!(self, FeedItem::Ad { .. }) }
    pub fn tweet_id(&self) -> Option<&str> {
        match self { FeedItem::Tweet { tweet_id } => Some(tweet_id), _ => None }
    }
}

// ─── Request context passed to the targeting engine ──────────────────────────

#[derive(Debug, Clone)]
pub struct AdContext {
    pub current_hour: u8,       // 0–23
    pub current_weekday: u8,    // 0=Mon … 6=Sun
    pub impressions_today: HashMap<String, u32>,   // campaign_id → count
    pub impressions_week: HashMap<String, u32>,
}

impl AdContext {
    pub fn new(hour: u8, weekday: u8) -> Self {
        Self {
            current_hour: hour,
            current_weekday: weekday,
            impressions_today: HashMap::new(),
            impressions_week: HashMap::new(),
        }
    }

    pub fn impressions_today_for(&self, campaign_id: &str) -> u32 {
        *self.impressions_today.get(campaign_id).unwrap_or(&0)
    }

    pub fn impressions_week_for(&self, campaign_id: &str) -> u32 {
        *self.impressions_week.get(campaign_id).unwrap_or(&0)
    }
}
