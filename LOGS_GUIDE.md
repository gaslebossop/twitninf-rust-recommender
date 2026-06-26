# 📊 Guide des Logs - TwitNinf Rust Recommender

## Configuration actuelle
- **Log level**: `debug` (traces les points clés)
- **Output**: systemd journal (journalctl)
- **Service**: `twitninf-rust-recommender`

## 🔍 Voir les logs sur le VPS

### **Option 1 : Logs en temps réel**
```bash
sudo journalctl -u twitninf-rust-recommender -f
```
- `-f` = follow (temps réel)
- `-u` = unité systemd

### **Option 2 : Logs récents (dernières 50 lignes)**
```bash
sudo journalctl -u twitninf-rust-recommender -n 50
```

### **Option 3 : Logs depuis une heure**
```bash
sudo journalctl -u twitninf-rust-recommender --since "1 hour ago"
```

### **Option 4 : Filtrer par mot-clé**
```bash
# Voir seulement les scores finaux
sudo journalctl -u twitninf-rust-recommender -f | grep "FINAL SCORE"

# Voir seulement les erreurs
sudo journalctl -u twitninf-rust-recommender -f -p err

# Voir seulement les requêtes recommendation
sudo journalctl -u twitninf-rust-recommender -f | grep "RECOMMEND REQUEST"
```

### **Option 5 : Logs au format JSON (parsing)**
```bash
sudo journalctl -u twitninf-rust-recommender -f -o json | jq .
```

### **Option 6 : Statistiques des logs**
```bash
# Compter les erreurs
sudo journalctl -u twitninf-rust-recommender | grep -c "error"

# Voir les 10 tweets avec les plus hauts scores
sudo journalctl -u twitninf-rust-recommender | grep "FINAL SCORE" | tail -10
```

---

## 📈 Niveaux de logs disponibles

Modifier `RUST_LOG` dans `twitninf-rust-recommender.service` :

| Level | Description | Cas d'usage |
|-------|-------------|-----------|
| `error` | Seulement les erreurs | Production sécurisée |
| `warn` | Avertissements + erreurs | Monitoring basique |
| `info` | Infos générales | Statistiques |
| `debug` | Points clés du workflow | **👈 Actuellement activé** |
| `trace` | TOUS les détails | Débugage détaillé |

### Changer le niveau

```bash
sudo nano /etc/systemd/system/twitninf-rust-recommender.service
```

Modifier la ligne :
```ini
Environment=RUST_LOG=twitninf_recommender=trace
```

Puis reload + restart :
```bash
sudo systemctl daemon-reload
sudo systemctl restart twitninf-rust-recommender
```

---

## 🎯 Logs importants à regarder

### Démarrage d'une requête
```
━━━ RECOMMEND REQUEST ━━━
user_id: abc123
mode: feed
limit: 50
```

### Profil utilisateur chargé
```
User profile built
following_count: 245
top_authors: 12
```

### Candidats collectés de 8 sources
```
Candidates collected from 8 sources
trending: 400
social_graph: 300
viral: 250
discovery: 150
temporal: 150
influencer: 150
personalized: 200
quality: 100
```

### Scoring d'un tweet
```
━━━ SCORING TWEET START ━━━
D1 Engagement Velocity: 0.75
D2 Content Intelligence: 0.62
D3 Social Graph Dynamics: 0.55
D4 Temporal Dynamics: 0.88
D5 Behavioral Prediction: 0.68
D6 Content Diversity: 0.80
D7 Viral Prediction: 0.45
D8 Personalization Depth: 0.72
━━━ FINAL SCORE ━━━
final_score: 0.67
```

### Métriques du feed final
```
Feed metrics calculated
diversity_score: 0.85
freshness_score: 0.92
relevance_score: 0.78
viral_potential: 0.55
novelty_score: 0.62
```

---

## 🚨 Dépannage

### L'app tourne mais aucun log ?
```bash
# Vérifier que le service tourne
sudo systemctl status twitninf-rust-recommender

# Relancer avec logs debug
sudo systemctl restart twitninf-rust-recommender
sudo journalctl -u twitninf-rust-recommender -f
```

### Trop de logs ?
Réduire le niveau :
```bash
Environment=RUST_LOG=twitninf_recommender=warn
```

### Sauvegarder les logs dans un fichier
```bash
sudo journalctl -u twitninf-rust-recommender > logs.txt

# Ou en temps réel
sudo journalctl -u twitninf-rust-recommender -f >> logs.txt &
```

---

## 📝 Exemples de commandes utiles

```bash
# Voir les 20 dernières lignes
sudo journalctl -u twitninf-rust-recommender -n 20

# Logs depuis hier 10:00
sudo journalctl -u twitninf-rust-recommender --since "yesterday 10:00"

# Voir seulement les warnings et erreurs
sudo journalctl -u twitninf-rust-recommender -p warning

# Exporter en JSON
sudo journalctl -u twitninf-rust-recommender -o json > logs.json

# Statistiques
sudo journalctl -u twitninf-rust-recommender | wc -l  # total de lignes
```

---

**Update**: `2026-06-26` - Logs à 8 dimensions ajoutés à l'algorithme NeuralRank Fusion
