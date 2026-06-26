# Guide : Comment écrire des tweets qui performent dans l'algo

> Basé sur le moteur NeuralRank Fusion — 8 dimensions de scoring en temps réel.

---

## Les 8 dimensions qui te scorent

| # | Dimension | Poids | Ce qui compte |
|---|-----------|-------|---------------|
| D1 | Engagement Velocity | **32%** | Vitesse d'accumulation des réactions |
| D2 | Content Intelligence | **18%** | Qualité et format du contenu |
| D3 | Social Graph | **15%** | Qui te suit et comment |
| D4 | Temporal Dynamics | **10%** | Quand tu postes |
| D5 | Behavioral Prediction | **8%** | Correspond-il aux habitudes du lecteur |
| D6 | Content Diversity | **6%** | Est-ce que ça apporte quelque chose de nouveau |
| D7 | Viral Prediction | **7%** | Potentiel de partage en cascade |
| D8 | Personalization Depth | **4%** | Affinité avec l'auteur |

---

## D1 — Engagement Velocity (le plus important)

L'algo mesure la **vitesse** d'engagement, pas juste le volume.
Un tweet avec 50 retweets en 30 min bat un tweet avec 500 likes en 2 jours.

### Valeur de chaque action

| Action | Poids |
|--------|-------|
| Retweet | ×5.0 |
| Share | ×4.0 |
| Commentaire | ×3.5 |
| Bookmark | ×2.5 |
| Like | ×1.0 |
| Vue | ×0.05 |

**→ Priorité absolue : déclenche des retweets et commentaires, pas des likes.**

### Bonus de fraîcheur (Phase 1)

- Tweet < 30 min → score D1 ×1.5
- Tweet < 2h → score D1 ×1.3
- Tweet > 2h → pas de bonus

**→ Les premières 30 minutes sont critiques. Si ça ne décolle pas vite, ça restera enterré.**

---

## D2 — Content Intelligence

### Longueur idéale

| Audience cible | Longueur optimale |
|----------------|------------------|
| Casual | < 80 caractères |
| Standard | 100–200 caractères |
| PowerUsers | > 200 caractères |

**→ Ne vise pas 280 chars par défaut. Le bon format dépend de ta cible.**

### Ce qui booste D2

- **Média (image/vidéo)** → +0.20 immédiat
- **2–3 hashtags** → sweet spot (+0.08 max)
- **1–2 @mentions pertinentes** → +0.06 max
- **Questions** → bonus si ton audience est curieuse
- **URLs sources** → bonus si ton audience aime les liens informatifs
- **Mots-clés en commun avec les intérêts du lecteur** → +0.15 max

### Format selon le profil lecteur

| Personnalité | Ce qui fonctionne |
|-------------|------------------|
| Enthousiaste | Émojis + points d'exclamation |
| Curieux | Questions + liens sources |
| Thoughtful | Texte long + URLs |
| Balanced | Mix équilibré |

---

## D4 — Temporal Dynamics (quand poster)

La demi-vie du contenu est de **4 heures**. Après 4h, la moitié de ta puissance temporelle est perdue.

### Stratégie de timing

1. **Poste quand tes abonnés sont actifs** — l'algo croise l'heure de publication avec les patterns d'activité de chaque lecteur.
2. **Évite minuit–7h** sauf si ta niche est internationale.
3. **Fenêtres idéales** (général) : 7h–9h, 12h–14h, 18h–21h.

### Momentum

Si ton tweet est **récent ET engagé** (< 2h avec des réactions) → le score de momentum est ×1.5. C'est une fenêtre d'accélération : interagis avec les premiers commentaires pour maintenir la vélocité.

---

## D7 — Viral Prediction

L'algo regarde le **ratio retweets/likes** (cascade effect).

- Beaucoup de likes, peu de retweets → audience passive → score viral bas
- Retweets > likes → signal fort de viralité → score viral élevé

### Shareabilité intrinsèque

| Élément | Bonus shareabilité |
|---------|------------------|
| Média | +0.30 |
| Au moins 1 hashtag | +0.10 |
| Au moins 1 URL | +0.10 |

**→ Un tweet avec image + 1 hashtag pertinent est structurellement plus viral qu'un tweet texte pur.**

---

## Formule du tweet performant

```
✅ Format gagnant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Accroche forte en 10 mots max]

[Développement — 100 à 180 chars]
[Inclut les mots-clés de ta niche]

[Image ou vidéo]
[1 à 3 hashtags ciblés]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Ce qui génère des retweets
- Opinions tranchées (accord ou désaccord fort)
- Infos exclusives ou contre-intuitives
- Formats listes (thread ou numéroté)
- Questions ouvertes à la fin

✅ Ce qui génère des commentaires
- Questions directes à l'audience
- Déclarations qui invitent au débat
- "Ça vous arrive aussi ?"
```

---

## Zones rouges — Shadowban automatique

Le système détecte et pénalise les comptes qui postent du contenu poubelle. La suppression est **graduelle et silencieuse** : tu continues à poster, mais personne ne voit tes tweets.

### 7 signaux qui dégradent ton compte

| Signal | Seuil | Gravité |
|--------|-------|---------|
| Taux de signalements | > 3% des vues | ⚠️ Critique |
| Zéro engagement | > 200 vues, 0 réaction | ⚠️ Élevée |
| Spam de liens | ≥ 2 URLs + texte < 50 chars | 🔶 Élevée |
| Trop de hashtags | > 30% des mots = #tags | 🔶 Moyenne |
| Contenu répété | < 25% de mots uniques | 🔶 Moyenne |
| Mention spam | > 5 @mentions | 🔻 Faible |
| Surcharge émojis | > 8 émojis + texte < 100 chars | 🔻 Faible |

### Les 4 niveaux de shadowban

| Niveau | Score multiplicateur | Effet |
|--------|---------------------|-------|
| Clean | ×1.00 | Visibilité normale |
| Monitoring | ×0.85 | Légère suppression (-15%) |
| Suppressed | ×0.45 | Exclu de Discovery & Trending |
| Ghosted | ×0.05 | Visible seulement par tes abonnés |

Le niveau est recalculé sur les **7 derniers jours**. Si > 70% de tes posts récents sont détectés comme poubelle → Ghosted automatique.

### À ne jamais faire

```
❌ #follow #followback #f4f #like4like #repost (hashtag stuffing)
❌ @user1 @user2 @user3 @user4 @user5 @user6 (mention bombing)
❌ [lien] [lien] via @source (pas de texte original)
❌ Copier-coller le même tweet en boucle
❌ Poster du contenu qui génère des signalements répétés
```

---

## Stratégie par objectif

### Objectif : Reach maximal (Discovery)
1. Poste avec image ou vidéo
2. 1–2 hashtags ciblés (pas plus)
3. Entre 7h et 9h ou 18h et 21h
4. Finir par une question pour déclencher les commentaires
5. Réponds aux 5 premiers commentaires → relance la vélocité

### Objectif : Fidéliser tes abonnés
1. Poste régulièrement aux heures où ils sont actifs
2. Utilise les mots de ta niche (ils matchent leurs top_words)
3. Varie les formats : texte court / thread / image / question

### Objectif : Aller en Trending
1. Accroche toi à un sujet déjà trending (source bonus ×1.04)
2. Publie dans les **30 premières minutes** après que le sujet explose
3. Le contenu doit être original (pas un repost)
4. Ratio retweets/likes > 1.0 → signal de cascade fort

---

## Checklist avant de poster

- [ ] Le tweet a un média (image ou vidéo) ?
- [ ] La longueur est adaptée à mon audience ?
- [ ] J'ai 1 à 3 hashtags (pas plus de 30% des mots) ?
- [ ] Je poste dans une fenêtre horaire active ?
- [ ] Le contenu invite à réagir (question, opinion, info) ?
- [ ] Le texte est original (pas du copier-coller) ?
- [ ] Moins de 5 @mentions ?
- [ ] Moins de 8 émojis si le texte est court ?
- [ ] Je ne spam pas de liens sans contenu ?

---

*NeuralRank Fusion Engine — beta branch*
