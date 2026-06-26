# 🧠 TwitNinf Rust Recommender — NeuralRank Fusion Engine

Un moteur de recommandation ultramoderne basé sur 8 dimensions d'analyse en temps réel.

## ⚡ Caractéristiques

- **12 dimensions de scoring** : engagement, contenu, graphe social, temporalité, comportement, diversité, viralité, personnalisation
- **Scoring temps réel** : calcul instantané pour chaque tweet
- **8 sources de candidats** : trending, social graph, viral, discovery, temporal, influencers, personnalisé, quality
- **Cache Redis** : performances optimales
- **Logs détaillés** : trace complète de chaque décision
- **4 modes de recommandation** : Feed, Discover, Trending, ForYou

## 📊 Architecture

```
┌─ Profile Building (DB)
│  ├─ Social graph (following, mutual, 2nd degree)
│  ├─ Engagement metrics (daily/weekly trends)
│  ├─ Temporal patterns (heure/jour d'activité)
│  ├─ Content preferences (longueur, médias, hashtags)
│  └─ Top authors & interests
│
├─ 8 Parallel Candidate Sources (DB)
│  ├─ Trending (6-72h window, high engagement)
│  ├─ Social Graph (tweets from following)
│  ├─ Viral (high engagement velocity)
│  ├─ Discovery (random users)
│  ├─ Temporal (user's active hours)
│  ├─ Influencers (verified/premium)
│  ├─ Personalized (top authors)
│  └─ Quality (verified premium users)
│
├─ Deduplication
│  └─ Keep highest weight source
│
├─ Scoring Pipeline
│  ├─ D1: Engagement Velocity (25%)
│  ├─ D2: Content Intelligence (20%)
│  ├─ D3: Social Graph Dynamics (15%)
│  ├─ D4: Temporal Dynamics (10%)
│  ├─ D5: Behavioral Prediction (10%)
│  ├─ D6: Content Diversity (8%)
│  ├─ D7: Viral Prediction (7%)
│  ├─ D8: Personalization Depth (5%)
│  └─ Modifiers: Anti-bubble, Moderation, Source bonus
│
├─ Feed Quality Metrics
│  ├─ Diversity score (author ratio)
│  ├─ Freshness score (recency decay)
│  ├─ Relevance score (behavioral prediction avg)
│  ├─ Viral potential (viral prediction avg)
│  └─ Novelty score (% discovery tweets)
│
└─ Response
   ├─ Sorted tweet IDs
   ├─ Pagination
   ├─ Metadata with breakdowns
   └─ Cache for next request

```

## 🚀 Démarrage rapide

### Prérequis
- Rust 1.70+
- PostgreSQL 14+
- Redis 6.0+

### Installation locale

```bash
# Installer les dépendances
cargo build

# Configurer les variables d'env
export RUST_PORT=3002
export DB_HOST=localhost
export DB_NAME=twitninf
export REDIS_URL=redis://localhost:6379
export RUST_LOG=twitninf_recommender=debug

# Lancer
cargo run --release
```

### API Endpoint

```bash
POST /api/recommend
Content-Type: application/json

{
  "user_id": "user-uuid",
  "mode": "feed",
  "limit": 50,
  "offset": 0,
  "force_refresh": false
}
```

Response:
```json
{
  "success": true,
  "user_id": "...",
  "tweet_ids": ["id1", "id2", ...],
  "count": 50,
  "algorithm": "NeuralRank Fusion",
  "algorithm_version": "2.0.0 — 12 dimensions réelles",
  "mode": "feed",
  "latency_ms": 234,
  "cache_hit": false,
  "metadata": {
    "candidates_collected": 1400,
    "sources": {
      "trending": 400,
      "social_graph": 300,
      "viral": 250,
      ...
    },
    "user_profile": {
      "user_type": "PowerUser",
      "confidence": 0.95,
      ...
    },
    "quality_metrics": {
      "diversity_score": 0.85,
      "freshness_score": 0.92,
      ...
    }
  }
}
```

## 📊 Logs & Monitoring

### Voir les logs en temps réel
```bash
sudo journalctl -u twitninf-rust-recommender -f
```

### Filtrer les logs
```bash
# Voir seulement les scores finaux
sudo journalctl -u twitninf-rust-recommender -f | grep "FINAL SCORE"

# Voir les requêtes recommendations
sudo journalctl -u twitninf-rust-recommender -f | grep "RECOMMEND REQUEST"
```

**→ [Voir LOGS_GUIDE.md pour toutes les commandes](./LOGS_GUIDE.md)**

## 🎯 Scoring Detaillé

### D1 : Engagement Velocity (25%)
- Vitesse d'accumulation d'engagement
- Pondération : likes (1x), comments (3.5x), retweets (5x), shares (4x), bookmarks (2.5x), views (0.05x)
- Accélération détectée sur fenêtres 1h vs 6h
- Multiplicateur de récence (< 1h = 3x, < 3h = 2x, < 6h = 1.5x)

### D2 : Content Intelligence (20%)
- Longueur idéale (court/medium/long selon profil)
- Richesse : médias, hashtags, mentions
- Style : émojis, exclamations, questions, URLs selon personnalité
- Correspondance mots-clés préférés

### D3 : Social Graph Dynamics (15%)
- Degré 1 : follow direct (0.55)
- Degré 1.5 : follow mutuel (0.25)
- Degré 2 : ami d'ami (0.12)
- Affinité auteur précédente (0.20 max)
- Influence auteur dans le réseau

### D4 : Temporal Dynamics (10%)
- Récence : demi-vie 6h (ln(2)/6h ≈ 0.115)
- Alignement heure d'activité utilisateur
- Alignement jour de la semaine
- Momentum : bonus récence + engagement

### D5 : Behavioral Prediction (10%)
- Type d'utilisateur : PowerUser (0.20), Regular (0.12), Casual (0.05)
- Préférence média : +0.18 si user aime les médias
- Match longueur préférée : +0.15
- Prédiction de retweet : ratio utilisateur × tweetabilité
- Tendance engagement : +0.10 si trending up
- Loyauté inverse : (1 - churn_risk) × 0.12

### D6 : Content Diversity (8%)
- Base 0.70, premier tweet 0.80
- Bonus média si peu dans le feed (-0.40)
- Bonus hashtags nouveaux (+0.10)
- Bonus contenu non vu (+0.05)

### D7 : Viral Prediction (7%)
- Ratio partage : (retweets + shares) / total_engagement
- Cascade effect : retweets/likes ratio
- Spread velocity : engagement/heure sigmoid
- Shareabilité : médias, hashtags, URLs

### D8 : Personalization Depth (5%)
- Affinité auteur (40%)
- Match intérêts (words top 20, weighted by frequency)
- Positivité émotionnelle + emojis
- Heure activité peak (+0.15)

### Modifiers
- **Anti-bubble** : penalise progressively (1x, 0.88x, 0.72x, 0.55x, 0.38x, 0.22x)
- **Moderation** : penalty pour rapports (max -50%)
- **Source bonus** : 0.01 - 0.08 selon source

## 🛠 Déploiement VPS

### Déployer
```bash
./deploy-vps.sh
```

### Configuration systemd
Le service est pré-configuré dans `twitninf-rust-recommender.service`:
- User: debian
- WorkDir: /home/debian/rust-recommender
- RestartPolicy: on-failure
- Logs: systemd journal

### Vérifier le status
```bash
sudo systemctl status twitninf-rust-recommender
sudo systemctl restart twitninf-rust-recommender
```

## 📈 Performance

- **Latency**: ~200-300ms pour 50 tweets
- **Throughput**: 100+ recommandations/sec
- **Memory**: ~200MB (base) + cache
- **DB Connections**: 10 pooled

## 🔧 Configuration

```ini
# .env ou systemd Environment=
RUST_PORT=3002
DB_HOST=localhost
DB_PORT=5432
DB_NAME=twitninf
DB_USER=admin
DB_PASSWORD=...
REDIS_URL=redis://localhost:6379
DB_POOL_SIZE=10
RUST_LOG=twitninf_recommender=debug
```

## 📚 Structure du Code

```
src/
├── main.rs              # Axum server + routes
├── models.rs            # Data structures
├── config.rs            # Configuration
├── handlers/
│   ├── recommendations.rs  # POST /recommend
│   ├── health.rs           # GET /health
│   └── ...
├── algorithm/
│   ├── scoring.rs       # 8 dimensions + modifiers
│   └── trending.rs      # Trending score
└── services/
    ├── recommender.rs   # Main orchestration
    └── cache_manager.rs # Redis caching
```

## 🐛 Debugging

### Logs complets (trace level)
```bash
Environment=RUST_LOG=twitninf_recommender=trace
```

### Passer en mode recherche
Voir LOGS_GUIDE.md → Filtrer par mot-clé

## 📊 Métriques importantes

À monitorer :
- **Latency p99** : < 500ms
- **Cache hit rate** : > 70%
- **Diversity score** : > 0.80
- **Freshness score** : > 0.85
- **Error rate** : < 0.1%

---

**Dernière mise à jour** : 2026-06-26 - Logs à 8 dimensions + LOGS_GUIDE.md
