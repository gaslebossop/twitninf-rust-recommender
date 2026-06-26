# 🚀 ROADMAP BETA - CTR Target: 15%

**Current Status**: v1.0.0 with 6-8% CTR  
**Target**: 15% CTR (88% improvement)  
**Timeline**: 3-6 months  
**Branch**: `beta` (experimental features)

---

## 🎯 Phase 1: Quick Wins (Week 1-2) - Target: +1-2% CTR

### 1.1 **Tune Engagement Velocity (D1)** ⚡
- **Current**: W_D1 = 25%
- **Test**: Increase to 30-35%
- **Expected CTR gain**: +0.5-1%
- **Implementation**: Adjust `constants.rs` D1 weights
- **Risk**: Low (easy to revert)
- **Status**: 📋 Planning

```rust
// Before
const W_D1_ENGAGEMENT_VELOCITY: f64 = 0.25;

// After
const W_D1_ENGAGEMENT_VELOCITY: f64 = 0.32;
// Adjust others to sum to 1.0
```

### 1.2 **Aggressive Recent Content Boost** 🔥
- **Current**: 6h half-life (D4 Temporal)
- **Test**: 4h half-life + recent engagement multiplier
- **Expected CTR gain**: +0.5%
- **Implementation**: 
  - Reduce `RECENCY_HALF_LIFE_HOURS` from 6.0 to 4.0
  - Add bonus multiplier for tweets < 1h old
- **Risk**: May reduce freshness perception
- **Status**: 📋 Planning

### 1.3 **Trending Bias Increase** 📈
- **Current**: Trending = 2% of sources
- **Test**: Trending = 5% of sources + boost multiplier
- **Expected CTR gain**: +0.5-1%
- **Implementation**:
  - Increase `CANDIDATE_LIMIT_TRENDING` from 400 to 600
  - Add trending multiplier to D7 (Viral)
- **Risk**: May hurt diversity
- **Status**: 📋 Planning

---

## 🎯 Phase 2: Machine Learning Integration (Week 3-4) - Target: +2-3% CTR

### 2.1 **Add ML-based CTR Predictor** 🤖
- **Current**: Pure rule-based (no ML)
- **New**: Lightweight ML model for CTR prediction
- **Architecture**:
  ```
  Input: [D1, D2, D3, D4, D5, D6, D7, D8 scores]
  Model: Gradient Boosting (LightGBM)
  Output: Predicted CTR probability
  
  Final Score = (Base Score * 0.6) + (Predicted CTR * 0.4)
  ```
- **Expected CTR gain**: +1.5-2%
- **Data requirement**: 10,000+ interactions
- **Training time**: 1 hour
- **Implementation**:
  - Collect interaction data
  - Train on historical CTR patterns
  - Deploy as scoring factor
- **Risk**: Model overfitting
- **Status**: 📋 Planning

### 2.2 **User-specific Weighting** 👤
- **Current**: Same weights for all users
- **New**: Personalized dimension weights
- **Approach**:
  - Calculate CTR per dimension per user
  - Adjust weights based on user segment
  - PowerUsers: Higher D1 (engagement)
  - Casual: Higher D4 (freshness) + D3 (social)
- **Expected CTR gain**: +0.8-1%
- **Status**: 📋 Planning

### 2.3 **Add D9: User-Tweet Similarity** 💫
- **Current**: 8 dimensions
- **New**: D9 content similarity scoring
- **Calculation**:
  ```
  similarity = (user_interests ∩ tweet_topics) / |user_interests|
  
  Score based on:
  - Keyword overlap
  - Author similarity
  - Topic embedding similarity (future)
  ```
- **Weight**: 5-8%
- **Expected CTR gain**: +0.5%
- **Status**: 📋 Planning

---

## 🎯 Phase 3: Advanced Personalization (Week 5-6) - Target: +2-3% CTR

### 3.1 **Contextual Bandit Algorithm** 🎲
- **Current**: Static scoring for all users
- **New**: Explore vs Exploit trade-off
- **Implementation**:
  ```
  - Exploit (80%): Show highest-scoring tweets
  - Explore (20%): Show diverse/experimental tweets
  - Learn user preferences dynamically
  ```
- **Expected CTR gain**: +1-1.5%
- **Library**: `contextual-bandits` crate
- **Status**: 📋 Planning

### 3.2 **Real-time User Feedback Loop** ⚡
- **Current**: Batch scoring (5 min cache)
- **New**: Real-time CTR feedback
- **Mechanism**:
  - User clicks tweet → Immediate score boost for similar tweets
  - User skips tweet → Immediate score reduction
  - Learn in sub-second feedback loop
- **Expected CTR gain**: +1-1.5%
- **Implementation**:
  - Add click tracking
  - Update scores in real-time
  - Use Redis for fast updates
- **Status**: 📋 Planning

### 3.3 **A/B Testing Infrastructure** 🧪
- **Current**: No A/B testing
- **New**: Statistical A/B testing framework
- **Setup**:
  ```
  Control: Current algorithm (v1.0.0)
  Variant A: Phase 1 + 2 optimizations
  Variant B: Phase 3 advanced features
  
  Measure: CTR, engagement, retention
  Duration: 2 weeks per variant
  ```
- **Expected CTR gain**: Validate gains
- **Status**: 📋 Planning

---

## 🎯 Phase 4: Advanced ML (Week 7-8) - Target: +2% CTR

### 4.1 **Deep Learning CTR Model** 🧠
- **Current**: Gradient Boosting
- **New**: Neural network for CTR prediction
- **Architecture**:
  ```
  Input: [8 dimensions + user features]
  Hidden: 3 layers (256, 128, 64 neurons)
  Output: CTR probability (sigmoid)
  
  Loss: Binary cross-entropy
  Optimizer: Adam
  ```
- **Framework**: TensorFlow/PyTorch (Python API)
- **Expected CTR gain**: +1-1.5%
- **Inference time**: < 10ms
- **Status**: 📋 Planning

### 4.2 **Collaborative Filtering Layer** 👥
- **Current**: Content-based only
- **New**: User-user + item-item similarity
- **Approach**:
  - Build user similarity matrix (from behavior)
  - Build tweet similarity matrix (embeddings)
  - Combine with content scores
- **Expected CTR gain**: +0.5-1%
- **Status**: 📋 Planning

### 4.3 **Graph Neural Network (GNN)** 🕸️
- **Current**: Local features only
- **New**: Network-based features
- **Graph structure**:
  ```
  Nodes: Users, Tweets, Authors
  Edges: Follows, Likes, Retweets
  Task: Node embeddings for scoring
  ```
- **Expected CTR gain**: +0.5%
- **Complexity**: High
- **Status**: 🔮 Future

---

## 📊 Expected CTR Growth Timeline

```
v1.0.0 (Current):      6-8% CTR  ├─ Baseline
                              │
Phase 1 (Week 1-2):   +1-2% → 7-10% CTR
                              │
Phase 2 (Week 3-4):   +2-3% → 9-13% CTR
                              │
Phase 3 (Week 5-6):   +2-3% → 11-16% CTR  ✅ TARGET!
                              │
Phase 4 (Week 7-8):   +2% → 13-18% CTR
```

---

## 🛠️ Implementation Priority Matrix

| Phase | Effort | Expected Gain | Complexity | Priority |
|-------|--------|---------------|------------|----------|
| 1.1 Tune D1 | Low | +0.5-1% | Low | 🔴 P0 |
| 1.2 Recent boost | Low | +0.5% | Low | 🔴 P0 |
| 1.3 Trending boost | Low | +0.5-1% | Low | 🔴 P0 |
| 2.1 ML CTR | Medium | +1.5-2% | Medium | 🟠 P1 |
| 2.2 User weights | Medium | +0.8-1% | Medium | 🟠 P1 |
| 2.3 Add D9 | Medium | +0.5% | Low | 🟡 P2 |
| 3.1 Bandit | High | +1-1.5% | High | 🟡 P2 |
| 3.2 Real-time | High | +1-1.5% | High | 🟡 P2 |
| 3.3 A/B Testing | Medium | Validation | Medium | 🔴 P0 |
| 4.1 Deep Learning | High | +1-1.5% | High | 🟡 P2 |
| 4.2 Collab Filter | High | +0.5-1% | High | 🟡 P2 |
| 4.3 GNN | Very High | +0.5% | Very High | 🔮 P3 |

---

## 📋 Checkpoint: Phase 1 Checklist

- [ ] Tune D1 engagement weight
- [ ] Increase recent content multiplier
- [ ] Boost trending source
- [ ] A/B test changes
- [ ] Measure CTR improvement
- [ ] Decide on Phase 2

**Goal**: Reach 8-10% CTR by end of Week 2

---

## 📊 Measurement Strategy

### Metrics to Track
```
Primary:
  - CTR (Click-Through Rate)
  - Engagement Rate (likes, retweets, replies)
  
Secondary:
  - User Retention (DAU/MAU)
  - Session Length
  - Diversity (unique authors)
  - Freshness (content age)
  - Personalization Accuracy
  
Health:
  - Latency (must stay < 500ms)
  - Error Rate (must stay < 0.1%)
  - Cache Hit Rate (target > 70%)
```

### Data Collection
```
Track per:
- User segment (PowerUser, Regular, Casual)
- Content type (text, media, article)
- Time of day
- User tenure (new vs veteran)
```

---

## 🚀 Deployment Strategy

### Canary Rollout
1. **Week 1**: Deploy to 5% of users (beta branch)
2. **Week 2**: Monitor metrics, expand to 25% if good
3. **Week 3**: Expand to 100% if no regressions
4. **Fallback**: Always able to revert in < 5 minutes

### A/B Test Structure
```
Control:  v1.0.0 (current, 50% of users)
Variant:  Phase 1 + 2 (50% of users)

Duration: 2 weeks
Stats required: 95% confidence, 0.5% minimum detectable effect
```

---

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Over-optimization | Engagement ↑, Health ↓ | Track diversity/retention metrics |
| User churn | DAU drops | Canary rollout, quick rollback |
| Latency increase | UX degradation | Strict SLA: < 500ms p99 |
| Model overfitting | Performance drop on new data | Use validation set, regularization |
| Data privacy | User data misuse | No storing personal data |

---

## 💡 Future Ideas (Beyond Phase 4)

1. **Reinforcement Learning**: Train agent on long-term user value
2. **Causal Inference**: Understand which features actually drive CTR
3. **Counterfactual Explanations**: Explain why tweet was/wasn't shown
4. **Federated Learning**: Train on-device models (privacy)
5. **Multi-Objective Optimization**: CTR + Diversity + Health

---

## 📈 Success Criteria

- ✅ Reach 15% CTR without regression in diversity/retention
- ✅ Maintain latency < 300ms p99
- ✅ Keep error rate < 0.1%
- ✅ User satisfaction score stays > 4.0/5.0
- ✅ No ethical concerns (filter bubble, manipulation)

---

## 🎯 Next Steps

1. **Code Review**: Review Phase 1 implementation
2. **A/B Test**: Run statistical validation
3. **Iterate**: Based on results, proceed to Phase 2
4. **Monitor**: Track all metrics continuously
5. **Ship**: When confident, merge to main

---

**Branch**: `beta`  
**Status**: 🚀 Active Development  
**Last Updated**: 2026-06-26  
**CTR Target**: 15% (+88% from baseline)
