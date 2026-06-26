use std::collections::HashMap;

use crate::models::{UserProfile, UserType};

use super::models::{
    Ad, AdCampaign, AdContext, AudienceSegment, FeedItem, MatchReason, ScoredAd, TimeTargeting,
};

// ─── Scoring weights ──────────────────────────────────────────────────────────

const W_KEYWORD:   f64 = 0.35;
const W_SEGMENT:   f64 = 0.20;
const W_TEMPORAL:  f64 = 0.15;
const W_ENGAGE:    f64 = 0.20;
const W_REVENUE:   f64 = 0.10;

const MIN_QUALITY:       f64 = 0.10;  // ads below this are dropped before auction
const LTV_HIGH_MARK:     f64 = 200.0; // normalisation ceiling for lifetime_value
const BID_HIGH_MARK:     f64 = 10.0;  // normalisation ceiling for CPM bid (USD)
const MAX_ADS_PER_PAGE:  usize = 5;
const DEFAULT_INTERVAL:  usize = 6;   // 1 sponsored slot every N organic tweets

// ─── AdTargetingEngine ────────────────────────────────────────────────────────

/// Core class for targeted ad selection and feed injection.
///
/// Usage:
/// ```rust
/// let engine = AdTargetingEngine::new();
/// let scored = engine.select_ads(&campaigns, &profile, &ctx, 3);
/// let feed   = engine.inject_into_feed(&tweet_ids, scored, None);
/// ```
pub struct AdTargetingEngine;

impl AdTargetingEngine {
    pub fn new() -> Self { Self }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Score a single ad for a user. Returns `None` if the ad is hard-excluded
    /// (negative keyword hit, frequency cap, LTV gate, bad time window).
    pub fn score_ad(
        &self,
        ad: &Ad,
        profile: &UserProfile,
        context: &AdContext,
    ) -> Option<ScoredAd> {
        let spec = &ad.targeting;

        // ── Hard exclusion gates ──────────────────────────────────────────────
        if context.impressions_today_for(&ad.campaign_id) >= spec.frequency_cap_daily {
            return None;
        }
        if context.impressions_week_for(&ad.campaign_id) >= spec.frequency_cap_weekly {
            return None;
        }
        if let Some(min_ltv) = spec.min_ltv {
            if profile.lifetime_value < min_ltv { return None; }
        }
        if let Some(max_churn) = spec.max_churn_risk {
            if profile.churn_risk > max_churn { return None; }
        }

        // Negative keyword — if any user top-word is in negative list, skip
        let top_words: HashMap<&str, u32> = profile.top_words.iter()
            .map(|(w, c)| (w.as_str(), *c))
            .collect();
        let blocked = spec.negative_keywords.iter()
            .any(|kw| top_words.contains_key(kw.as_str()));
        if blocked { return None; }

        // ── Scoring signals ──────────────────────────────────────────────────
        let mut reasons = Vec::with_capacity(5);

        let kw_score  = self.keyword_score(&spec.keywords, &top_words, &profile.top_words);
        let seg_score = self.segment_score(&spec.audience_segments, profile);
        let tmp_score = self.temporal_score(spec.time_targeting.as_ref(), profile, context);
        let eng_score = self.engagement_score(profile);
        let rev_score = self.revenue_score(profile, ad.bid_amount);

        reasons.push(MatchReason { signal: "keyword_match".into(),       weight: W_KEYWORD,  score: kw_score });
        reasons.push(MatchReason { signal: "user_segment".into(),        weight: W_SEGMENT,  score: seg_score });
        reasons.push(MatchReason { signal: "temporal_alignment".into(),  weight: W_TEMPORAL, score: tmp_score });
        reasons.push(MatchReason { signal: "engagement_potential".into(),weight: W_ENGAGE,   score: eng_score });
        reasons.push(MatchReason { signal: "revenue_potential".into(),   weight: W_REVENUE,  score: rev_score });

        let quality_score = kw_score  * W_KEYWORD
                          + seg_score * W_SEGMENT
                          + tmp_score * W_TEMPORAL
                          + eng_score * W_ENGAGE
                          + rev_score * W_REVENUE;

        if quality_score < MIN_QUALITY { return None; }

        // Relevance = keyword + segment fit only (the metric surfaced to advertisers)
        let relevance_score = (kw_score * 0.60 + seg_score * 0.40).clamp(0.0, 1.0);

        // Second-price auction key: higher quality earns a better effective bid
        let effective_bid = ad.bid_amount * quality_score;

        Some(ScoredAd {
            ad_id: ad.id.clone(),
            campaign_id: ad.campaign_id.clone(),
            tweet_id: ad.tweet_id.clone(),
            relevance_score,
            quality_score,
            effective_bid,
            match_reasons: reasons,
        })
    }

    /// Select the best ads from active campaigns via a second-price auction.
    pub fn select_ads(
        &self,
        campaigns: &[AdCampaign],
        profile: &UserProfile,
        context: &AdContext,
        max_ads: usize,
    ) -> Vec<ScoredAd> {
        let cap = max_ads.min(MAX_ADS_PER_PAGE);

        let mut scored: Vec<ScoredAd> = campaigns.iter()
            .filter(|c| c.is_active())
            .flat_map(|c| c.ads.iter())
            .filter_map(|ad| self.score_ad(ad, profile, context))
            .collect();

        // Sort by effective_bid descending — highest bid wins the slot
        scored.sort_by(|a, b| b.effective_bid.partial_cmp(&a.effective_bid).unwrap());
        scored.truncate(cap);
        scored
    }

    /// Weave sponsored slots into an organic tweet feed at regular intervals.
    ///
    /// `injection_interval` — number of organic tweets between each ad slot.
    /// Defaults to `DEFAULT_INTERVAL` (6) if `None`.
    pub fn inject_into_feed(
        &self,
        tweet_ids: &[String],
        ads: Vec<ScoredAd>,
        injection_interval: Option<usize>,
    ) -> Vec<FeedItem> {
        let interval = injection_interval.unwrap_or(DEFAULT_INTERVAL);
        let mut feed = Vec::with_capacity(tweet_ids.len() + ads.len());
        let mut ad_iter = ads.into_iter();
        let mut since_last_ad = 0usize;

        for tweet_id in tweet_ids {
            feed.push(FeedItem::Tweet { tweet_id: tweet_id.clone() });
            since_last_ad += 1;

            if since_last_ad >= interval {
                if let Some(ad) = ad_iter.next() {
                    feed.push(FeedItem::Ad { ad });
                    since_last_ad = 0;
                }
            }
        }
        feed
    }

    // ── Private scoring helpers ───────────────────────────────────────────────

    /// Keyword relevance: weighted overlap between ad keywords and user's top words.
    fn keyword_score(
        &self,
        keywords: &[String],
        top_words: &HashMap<&str, u32>,
        raw_top_words: &[(String, u32)],
    ) -> f64 {
        if keywords.is_empty() {
            return 0.50; // untargeted creative: neutral score
        }
        let total_weight: u32 = raw_top_words.iter().map(|(_, c)| c).sum();
        if total_weight == 0 {
            return 0.0;
        }
        let matched_weight: u32 = keywords.iter()
            .filter_map(|kw| top_words.get(kw.as_str()))
            .sum();

        // Scale: matched_weight / total * 10 so even 10% overlap gives a decent score
        ((matched_weight as f64 / total_weight as f64) * 10.0).clamp(0.0, 1.0)
    }

    /// User segment match: 1.0 if any targeting segment matches, 0.0 otherwise.
    fn segment_score(
        &self,
        segments: &[AudienceSegment],
        profile: &UserProfile,
    ) -> f64 {
        if segments.is_empty() || segments.contains(&AudienceSegment::All) {
            return 0.60; // broad targeting: neutral-positive score
        }
        let matched = segments.iter().any(|seg| match seg {
            AudienceSegment::All        => true,
            AudienceSegment::PowerUsers => matches!(profile.user_type, UserType::PowerUser),
            AudienceSegment::CasualUsers=> matches!(profile.user_type, UserType::Casual),
            AudienceSegment::HighValue  => profile.lifetime_value > 50.0,
            AudienceSegment::Churning   => profile.churn_risk > 0.60,
        });
        if matched { 1.0 } else { 0.0 }
    }

    /// Temporal relevance: time-window gate × user activity at this hour.
    fn temporal_score(
        &self,
        time_targeting: Option<&TimeTargeting>,
        profile: &UserProfile,
        context: &AdContext,
    ) -> f64 {
        if let Some(spec) = time_targeting {
            let hour_ok = spec.hours.is_empty()
                || spec.hours.contains(&context.current_hour);
            let day_ok  = spec.weekdays.is_empty()
                || spec.weekdays.contains(&context.current_weekday);
            if !hour_ok || !day_ok {
                return 0.0; // hard gate: outside the campaign's time window
            }
        }
        self.user_activity_now(profile, context)
    }

    /// User activity score at the current hour and weekday (from profile).
    fn user_activity_now(&self, profile: &UserProfile, context: &AdContext) -> f64 {
        let hour_act = profile.hourly_activity
            .get(context.current_hour as usize)
            .copied()
            .unwrap_or(0.0);
        let day_act  = profile.daily_activity
            .get(context.current_weekday as usize)
            .copied()
            .unwrap_or(0.0);
        ((hour_act + day_act) / 2.0).clamp(0.0, 1.0)
    }

    /// Engagement potential: how likely is this user to interact with any content.
    fn engagement_score(&self, profile: &UserProfile) -> f64 {
        let velocity = (profile.engagement_velocity / 100.0).clamp(0.0, 1.0);
        let trend_bonus = if profile.engagement_trend > 0.0 { 0.10 } else { 0.0 };
        (velocity + trend_bonus).clamp(0.0, 1.0)
    }

    /// Revenue potential: high-LTV users × bid premium.
    fn revenue_score(&self, profile: &UserProfile, bid_amount: f64) -> f64 {
        let ltv_factor  = (profile.lifetime_value / LTV_HIGH_MARK).clamp(0.0, 1.0);
        let bid_factor  = (bid_amount / BID_HIGH_MARK).clamp(0.0, 1.0);
        ltv_factor * 0.60 + bid_factor * 0.40
    }
}

impl Default for AdTargetingEngine {
    fn default() -> Self { Self::new() }
}
