# Phase 1 Implementation Guide - Quick Wins for +1-2% CTR

## Overview
Phase 1 focuses on quick, low-risk optimizations that can boost CTR by 1-2% in the first 2 weeks.

---

## Task 1.1: Tune Engagement Velocity Weight (D1)

### Change 1: Increase D1 Weight from 25% to 32%

**File**: `src/constants.rs`

**Before**:
```rust
const W_D1_ENGAGEMENT_VELOCITY: f64 = 0.25;
const W_D2_CONTENT_INTELLIGENCE: f64 = 0.20;
const W_D3_SOCIAL_GRAPH: f64 = 0.15;
const W_D4_TEMPORAL: f64 = 0.10;
const W_D5_BEHAVIORAL: f64 = 0.10;
const W_D6_DIVERSITY: f64 = 0.08;
const W_D7_VIRAL: f64 = 0.07;
const W_D8_PERSONALIZATION: f64 = 0.05;
// Total: 1.00
```

**After** (Engagement-heavy):
```rust
const W_D1_ENGAGEMENT_VELOCITY: f64 = 0.32;  // +0.07
const W_D2_CONTENT_INTELLIGENCE: f64 = 0.18; // -0.02
const W_D3_SOCIAL_GRAPH: f64 = 0.15;
const W_D4_TEMPORAL: f64 = 0.10;
const W_D5_BEHAVIORAL: f64 = 0.10;
const W_D6_DIVERSITY: f64 = 0.06; // -0.02 (slight hit OK)
const W_D7_VIRAL: f64 = 0.07;
const W_D8_PERSONALIZATION: f64 = 0.04; // -0.01
// Total: 1.00
```

**Expected CTR Gain**: +0.5-1%

**Rationale**: 
- Engagement velocity is the strongest predictor of CTR
- Increasing weight will prioritize viral tweets
- Small hits on diversity acceptable for CTR gain

---

## Task 1.2: Aggressive Recent Content Boost

### Change 2: Reduce Recency Half-life from 6h to 4h

**File**: `src/constants.rs`

**Before**:
```rust
const RECENCY_HALF_LIFE_HOURS: f64 = 6.0;
const RECENCY_DECAY_RATE: f64 = 0.115; // ln(2)/6h
```

**After**:
```rust
const RECENCY_HALF_LIFE_HOURS: f64 = 4.0;
const RECENCY_DECAY_RATE: f64 = 0.173; // ln(2)/4h (~0.173)
```

**Calculation**: ln(2) / 4 = 0.1732...

**Expected CTR Gain**: +0.3-0.5%

**Rationale**:
- Recent tweets have higher engagement
- Shortens half-life → older tweets decay faster
- Users prefer fresh content

---

### Change 3: Add Recent Engagement Multiplier in D1

**File**: `src/algorithm/dimensions/d1_engagement_velocity.rs`

**Add**:
```rust
// New: Boost multiplier for very recent tweets
const VERY_RECENT_MULTIPLIER: f64 = 1.5; // tweets < 30 min old
const RECENT_MULTIPLIER: f64 = 1.3;      // tweets < 2 hours old

// In calculate() function, add:
let recency_boost = if age_h < 0.5 {
    VERY_RECENT_MULTIPLIER
} else if age_h < 2.0 {
    RECENT_MULTIPLIER
} else {
    1.0
};

// Apply to velocity_raw
let velocity_raw_boosted = velocity_raw * recency_boost;
```

**Expected CTR Gain**: +0.2-0.3%

---

## Task 1.3: Trending Source Boost

### Change 4: Increase Trending Candidate Limit

**File**: `src/constants.rs`

**Before**:
```rust
const CANDIDATE_LIMIT_TRENDING: usize = 400;
```

**After**:
```rust
const CANDIDATE_LIMIT_TRENDING: usize = 600; // +50%
```

**Expected CTR Gain**: +0.2%

**Rationale**:
- More trending tweets = higher engagement
- Minimal latency impact (still parallel queries)

---

### Change 5: Add Trending Multiplier to D7

**File**: `src/algorithm/dimensions/d7_viral_prediction.rs`

**Add trending boost**:
```rust
// In calculate() function:
let trending_bonus = if t.source == TweetSource::Trending {
    1.2 // 20% boost for trending tweets
} else {
    1.0
};

// Apply to final score
let d7_boosted = d7 * trending_bonus;
```

**Expected CTR Gain**: +0.3-0.5%

---

## Implementation Checklist

### Week 1

- [ ] **Monday**: Implement changes 1-5
  - [ ] Update `src/constants.rs`
  - [ ] Update `src/algorithm/dimensions/d1_engagement_velocity.rs`
  - [ ] Update `src/algorithm/dimensions/d7_viral_prediction.rs`
  - [ ] Code review
  - [ ] Unit tests pass

- [ ] **Tuesday**: Deploy to 5% of users (beta canary)
  - [ ] Build release binary
  - [ ] Deploy to beta instance
  - [ ] Monitor metrics
  - [ ] Check latency (must be < 300ms)

- [ ] **Wednesday**: Monitor for regressions
  - [ ] Track CTR (target: +0.5-1% improvement)
  - [ ] Check diversity (should stay > 80%)
  - [ ] Check retention (should stay stable)
  - [ ] Measure latency (p50, p95, p99)

- [ ] **Thursday-Friday**: Expand rollout
  - [ ] If metrics good: expand to 25% of users
  - [ ] Continue monitoring
  - [ ] Document findings

### Week 2

- [ ] **Monday-Wednesday**: Full rollout decision
  - [ ] If confident: expand to 100% of users
  - [ ] If issues: rollback to v1.0.0 (< 5 min)
  - [ ] Finalize metrics

- [ ] **Thursday-Friday**: Prepare Phase 2
  - [ ] Review Phase 2 ML approach
  - [ ] Prepare data collection pipeline
  - [ ] Plan A/B testing infrastructure

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_d1_with_engagement_boost() {
    let tweet = create_test_tweet(100, 10); // 100 views, 10 engagement
    let d1_score = calculate_d1(&tweet);
    assert!(d1_score > 0.6); // Should be high engagement
}

#[test]
fn test_recency_decay_4h() {
    let tweet_4h_old = create_tweet_with_age(4.0);
    let decay = exponential_decay(4.0, RECENCY_DECAY_RATE);
    assert!((decay - 0.5).abs() < 0.01); // Should be ~50% at 4h
}
```

### Integration Test

```rust
#[test]
fn test_phase1_ctr_improvement() {
    // Compare old vs new weights
    let tweets = load_test_data(1000);
    
    let old_scores = score_with_old_weights(&tweets);
    let new_scores = score_with_new_weights(&tweets);
    
    let old_top_10 = get_top_10(&old_scores);
    let new_top_10 = get_top_10(&new_scores);
    
    // Trending tweets should rank higher
    let trending_boost = measure_ranking_change(&old_top_10, &new_top_10);
    assert!(trending_boost > 1.1); // At least 10% boost
}
```

---

## Monitoring & Metrics

### Key Metrics Dashboard

```
Real-time (1-minute window):
- CTR: Target +0.5-1%
- Engagement Rate: Target +0.3-0.8%
- Click Volume: Trend analysis

5-minute window:
- Latency p50, p95, p99: Must stay < 500ms
- Error Rate: Must stay < 0.1%
- Cache Hit Rate: Must stay > 70%

Daily:
- DAU (Daily Active Users): Must not decrease
- Retention: Check 7-day retention
- Diversity Score: Target > 80%
- Freshness: Target > 90% < 24h old
```

### Alerting

Set up alerts for:
- CTR drop > 10% from baseline (anomaly)
- Latency p99 > 500ms (performance issue)
- Error rate > 1% (system issue)
- Cache hit rate < 50% (cache issue)

---

## Rollback Plan

If any metric regresses significantly:

```bash
# Immediate rollback (< 5 minutes)
git revert <commit>
cargo build --release
systemctl restart twitninf-recommender

# Verify
curl https://api/health  # Check healthy
# Monitor metrics for 30 minutes
```

---

## Expected Results (Week 2 Target)

| Metric | Current | Phase 1 Target | Change |
|--------|---------|----------------|--------|
| CTR | 6-8% | 8-10% | +1-2% |
| Engagement Rate | 5% | 5.5-6% | +0.5-1% |
| Latency (p99) | 250ms | < 300ms | ✅ Stable |
| Diversity | 85% | > 83% | ✅ Acceptable |
| Freshness | 92% | > 90% | ✅ Stable |

---

## Post-Phase-1 Decision

### If Successful (CTR > 8%)
✅ Proceed to Phase 2: ML Integration
- Start collecting interaction data for ML model
- Plan A/B test infrastructure
- Begin Phase 2.1 implementation

### If Partial (CTR +0.5-0.7%)
⚠️ Optimize further
- Fine-tune constants more aggressively
- Try different weight combinations
- Consider alternative approach

### If No Improvement (CTR ≤ 6%)
🔄 Rollback & Rethink
- Revert all changes
- Analyze why changes didn't help
- Try different approach
- Consider user segment targeting

---

## Code Changes Summary

### Files to Modify
1. `src/constants.rs` - Weight adjustments
2. `src/algorithm/dimensions/d1_engagement_velocity.rs` - Recent boost
3. `src/algorithm/dimensions/d7_viral_prediction.rs` - Trending boost
4. `src/algorithm/scoring.rs` - If needed for testing

### Expected Build Time
- Cargo check: ~10 seconds
- Cargo build --release: ~2-3 minutes

### Deployment Checklist
- [ ] All tests pass
- [ ] Code review approved
- [ ] Metrics baseline captured
- [ ] Rollback plan documented
- [ ] Team notified
- [ ] Monitoring dashboard ready
- [ ] Canary rollout configured

---

## Timeline

```
Week 1:
├─ Mon: Code changes
├─ Tue: Deploy canary (5%)
├─ Wed: Monitor & decide
└─ Thu-Fri: Rollout (25%)

Week 2:
├─ Mon-Wed: Full rollout or rollback
├─ Thu: Analysis
└─ Fri: Phase 2 planning
```

---

**Status**: 🚀 Ready to Implement  
**Expected Impact**: +0.8-1.5% CTR  
**Risk Level**: 🟢 Low (easy rollback)  
**Confidence**: 85%
