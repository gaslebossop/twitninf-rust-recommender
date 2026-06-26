use crate::constants::*;
use crate::models::{RawTweet, UserProfile, ContentLength, PersonalityType};
use crate::utils::math::gaussian;
use tracing::{debug, trace};

/// D2 — CONTENT INTELLIGENCE (20%)
/// Analyse réelle du contenu : longueur idéale, richesse, format, style
pub fn calculate(t: &RawTweet, profile: &UserProfile) -> f64 {
    let mut score = 0.0;
    trace!(content_length = t.content_length, personality = ?profile.personality_type, "D2 Start");

    // Length score
    let len = t.content_length as f64;
    let len_score = calculate_length_score(len, &profile.preferred_content_length);
    score += len_score * D2_LENGTH_WEIGHT;
    trace!(len_score, "D2 Length score");

    // Content richness
    let richness_score = calculate_richness(t);
    score += richness_score;
    trace!(richness_score, "D2 Richness (media, hashtags, mentions)");

    // Personality match
    let personality_score = calculate_personality_match(t, &profile.personality_type);
    score += personality_score;
    trace!(personality_score, "D2 Personality match");

    // Keyword matching
    let keyword_score = calculate_keyword_match(t, profile);
    score += keyword_score;
    trace!(keyword_score, "D2 Keyword matches");

    let final_d2 = score.clamp(0.0, 1.0);
    debug!(final_d2, len_score, richness_score, personality_score, keyword_score, "D2 Final");
    final_d2
}

#[inline]
fn calculate_length_score(len: f64, preference: &ContentLength) -> f64 {
    match preference {
        ContentLength::Short => {
            if len < CONTENT_LENGTH_SHORT_THRESHOLD {
                1.0
            } else {
                1.0 - ((len - CONTENT_LENGTH_SHORT_THRESHOLD) / 200.0).clamp(0.0, 0.8)
            }
        }
        ContentLength::Medium => gaussian(len, CONTENT_LENGTH_GAUSSIAN_MU, CONTENT_LENGTH_GAUSSIAN_SIGMA),
        ContentLength::Long => {
            if len > CONTENT_LENGTH_LONG_THRESHOLD {
                1.0
            } else {
                len / CONTENT_LENGTH_LONG_THRESHOLD
            }
        }
    }
}

#[inline]
fn calculate_richness(t: &RawTweet) -> f64 {
    let mut score = 0.0;
    if t.has_media {
        score += D2_MEDIA_WEIGHT;
    }
    score += (t.hashtag_count as f64 * D2_HASHTAG_WEIGHT).min(0.12);
    score += (t.mention_count as f64 * D2_MENTION_WEIGHT).min(0.09);
    score
}

#[inline]
fn calculate_personality_match(t: &RawTweet, personality: &PersonalityType) -> f64 {
    let len = t.content_length as f64;
    match personality {
        PersonalityType::Enthusiastic => {
            (t.emoji_count as f64 * D2_EMOJI_WEIGHT_ENTHUSIASTIC).min(0.10)
                + (t.exclamation_count as f64 * D2_EXCLAMATION_WEIGHT).min(0.06)
        }
        PersonalityType::Curious => {
            (t.question_count as f64 * D2_QUESTION_WEIGHT_CURIOUS).min(0.10)
                + (t.url_count as f64 * D2_URL_WEIGHT).min(0.06)
        }
        PersonalityType::Thoughtful => {
            let long_bonus = if len > 150.0 { 0.10 } else { 0.0 };
            long_bonus + (t.url_count as f64 * D2_URL_WEIGHT).min(0.08)
        }
        PersonalityType::Balanced => {
            (t.emoji_count as f64 * 0.01).min(0.05) + (t.hashtag_count as f64 * 0.02).min(0.05)
        }
    }
}

#[inline]
fn calculate_keyword_match(t: &RawTweet, profile: &UserProfile) -> f64 {
    let content_lower = t.content.to_lowercase();
    let matches: usize = profile
        .top_words
        .iter()
        .filter(|(word, _)| content_lower.contains(word.as_str()))
        .count();
    (matches as f64 * D2_KEYWORD_WEIGHT).min(0.15)
}
