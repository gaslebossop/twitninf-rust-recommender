use std::collections::HashSet;

use chrono::Utc;
use tracing::debug;

use crate::models::RawTweet;

use super::models::{AccountQualityScore, GarbageSignals, ShadowbanLevel};

// ─── Seuils de détection ──────────────────────────────────────────────────────

const SPAM_HASHTAG_RATIO: f64 = 0.30; // hashtags / nb mots
const SPAM_HASHTAG_MIN: i32 = 3; // minimum absolu pour déclencher
const MENTION_SPAM_THRESHOLD: i32 = 5;
const ZERO_ENG_MIN_VIEWS: i64 = 200;
const REPORT_RATE_THRESHOLD: f64 = 0.03; // 3% des vues → report
const LINK_SPAM_MAX_CHARS: i32 = 50;
const LINK_SPAM_MIN_URLS: i32 = 2;
const EMOJI_OVERLOAD_COUNT: i32 = 8;
const EMOJI_OVERLOAD_MAX_CHARS: i32 = 100;
const REPEAT_UNIQUE_RATIO: f64 = 0.25; // < 25% mots uniques = texte répétitif
const REPEAT_MIN_WORDS: usize = 10; // pas de pénalité sur tweets très courts

// ─── Thresholds pour le niveau de shadowban du compte ────────────────────────

const THRESHOLD_GHOSTED_GARBAGE_RATIO: f64 = 0.70;
const THRESHOLD_GHOSTED_REPORT_RATE: f64 = 0.10;
const THRESHOLD_GHOSTED_STRIKES: u32 = 10;

const THRESHOLD_SUPPRESSED_GARBAGE_RATIO: f64 = 0.50;
const THRESHOLD_SUPPRESSED_REPORT_RATE: f64 = 0.05;
const THRESHOLD_SUPPRESSED_STRIKES: u32 = 5;

const THRESHOLD_MONITORING_GARBAGE_RATIO: f64 = 0.25;
const THRESHOLD_MONITORING_REPORT_RATE: f64 = 0.02;
const THRESHOLD_MONITORING_STRIKES: u32 = 2;

// ─── GarbageContentDetector ──────────────────────────────────────────────────

/// Détecte les signaux de contenu poubelle dans un tweet ou sur un compte.
pub struct GarbageContentDetector;

impl GarbageContentDetector {
    pub fn new() -> Self {
        Self
    }

    /// Analyse un tweet individuel et retourne ses signaux de qualité.
    pub fn detect(&self, tweet: &RawTweet) -> GarbageSignals {
        let word_count = tweet.words.len().max(1);

        // Densité de hashtags spam : > 30% des mots sont des #tags
        let spam_hashtag_density = tweet.hashtag_count >= SPAM_HASHTAG_MIN
            && (tweet.hashtag_count as f64 / word_count as f64) > SPAM_HASHTAG_RATIO;

        // Mention spam : > 5 @mentions (bot ou astroturfing)
        let spam_mentions = tweet.mention_count > MENTION_SPAM_THRESHOLD;

        // Zéro engagement : beaucoup de vues, aucune réaction → contenu ignoré
        let zero_engagement = tweet.view_count > ZERO_ENG_MIN_VIEWS
            && tweet.like_count == 0
            && tweet.comment_count == 0
            && tweet.retweet_count == 0;

        // Taux de signalements élevé
        let report_rate = tweet.report_count as f64 / tweet.view_count.max(100) as f64;
        let high_report_rate = report_rate > REPORT_RATE_THRESHOLD;

        // Spam de liens purs : beaucoup d'URLs, très peu de texte original
        let pure_link_spam =
            tweet.url_count >= LINK_SPAM_MIN_URLS && tweet.content_length < LINK_SPAM_MAX_CHARS;

        // Surcharge d'émojis sans contenu textuel
        let emoji_overload = tweet.emoji_count > EMOJI_OVERLOAD_COUNT
            && tweet.content_length < EMOJI_OVERLOAD_MAX_CHARS;

        // Contenu répétitif : ratio mots uniques < 25% (copie-colle, template)
        let unique_words: HashSet<&str> = tweet.words.iter().map(|w| w.as_str()).collect();
        let repeat_content = word_count >= REPEAT_MIN_WORDS
            && (unique_words.len() as f64 / word_count as f64) < REPEAT_UNIQUE_RATIO;

        let signals = GarbageSignals {
            spam_hashtag_density,
            spam_mentions,
            zero_engagement,
            high_report_rate,
            pure_link_spam,
            emoji_overload,
            repeat_content,
        };

        if signals.is_garbage() {
            debug!(
                tweet_id = %tweet.id,
                score = signals.score(),
                signals = ?signals.active_signals(),
                "Garbage content detected"
            );
        }

        signals
    }

    /// Calcule le ShadowbanLevel pour un compte à partir de son score agrégé.
    /// Les seuils sont appliqués par ordre décroissant de gravité.
    pub fn compute_level(score: &AccountQualityScore) -> ShadowbanLevel {
        if score.manual_override.is_some() {
            return score.manual_override.unwrap();
        }

        if score.garbage_ratio_7d > THRESHOLD_GHOSTED_GARBAGE_RATIO
            || score.avg_report_rate_7d > THRESHOLD_GHOSTED_REPORT_RATE
            || score.spam_strikes >= THRESHOLD_GHOSTED_STRIKES
        {
            return ShadowbanLevel::Ghosted;
        }

        if score.garbage_ratio_7d > THRESHOLD_SUPPRESSED_GARBAGE_RATIO
            || score.avg_report_rate_7d > THRESHOLD_SUPPRESSED_REPORT_RATE
            || score.spam_strikes >= THRESHOLD_SUPPRESSED_STRIKES
        {
            return ShadowbanLevel::Suppressed;
        }

        if score.garbage_ratio_7d > THRESHOLD_MONITORING_GARBAGE_RATIO
            || score.avg_report_rate_7d > THRESHOLD_MONITORING_REPORT_RATE
            || score.spam_strikes >= THRESHOLD_MONITORING_STRIKES
        {
            return ShadowbanLevel::Monitoring;
        }

        ShadowbanLevel::Clean
    }

    /// Recalcule le AccountQualityScore depuis un lot de tweets récents (fenêtre 7j).
    pub fn update_account_score(
        &self,
        existing: &AccountQualityScore,
        recent_tweets: &[RawTweet],
    ) -> AccountQualityScore {
        if recent_tweets.is_empty() {
            // Pas de nouveaux posts → score inchangé, niveau recalculé
            let mut updated = existing.clone();
            updated.level = Self::compute_level(&updated);
            updated.computed_at = Utc::now();
            return updated;
        }

        let total = recent_tweets.len() as f64;

        // Ratio de posts poubelle
        let garbage_count = recent_tweets
            .iter()
            .filter(|t| self.detect(t).is_garbage())
            .count();
        let garbage_ratio = garbage_count as f64 / total;

        // Taux moyen de signalements
        let avg_report_rate = recent_tweets
            .iter()
            .map(|t| t.report_count as f64 / t.view_count.max(100) as f64)
            .sum::<f64>()
            / total;

        // Strikes consécutifs (les plus récents d'abord)
        let spam_strikes = recent_tweets
            .iter()
            .rev()
            .take_while(|t| self.detect(t).is_garbage())
            .count() as u32;

        // Déficit d'engagement : avg (likes+comments+retweets)/vues vs baseline 2%
        let avg_eng_rate = recent_tweets
            .iter()
            .map(|t| {
                let eng = (t.like_count + t.comment_count + t.retweet_count) as f64;
                eng / t.view_count.max(1) as f64
            })
            .sum::<f64>()
            / total;
        let engagement_deficit = avg_eng_rate - 0.02; // baseline plateforme 2%

        let mut updated = existing.clone();
        updated.garbage_ratio_7d = garbage_ratio;
        updated.avg_report_rate_7d = avg_report_rate;
        updated.spam_strikes = spam_strikes;
        updated.engagement_deficit = engagement_deficit;
        updated.level = Self::compute_level(&updated);
        updated.computed_at = Utc::now();

        debug!(
            user_id = %updated.user_id,
            level = updated.level.label(),
            garbage_ratio,
            avg_report_rate,
            spam_strikes,
            "Account quality score updated"
        );

        updated
    }
}

impl Default for GarbageContentDetector {
    fn default() -> Self {
        Self::new()
    }
}
