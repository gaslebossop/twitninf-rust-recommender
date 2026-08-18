# Comment le fil est construit

Document de référence sur l'algorithme de recommandation de twitninf, tel
qu'il tourne réellement en production. Chaque valeur citée vient du code, pas
d'une intention.

Trois services :

| Service | Rôle | Port |
|---|---|---|
| **twitninfbeta** (React Native) | Affiche, et rapporte ce que fait le lecteur | — |
| **api** (Node/Express) | Authentifie, hydrate les tweets, masque les contenus verrouillés | 3001 |
| **rust-recommender** (Axum) | Classe. C'est ici que vit l'algorithme | 3002 |

Le moteur Rust ne renvoie **jamais** de tweets, seulement des identifiants
classés. C'est l'API Node qui les transforme en objets complets. Cette
séparation est structurante : le moteur ne sait rien des contenus payants, des
blocages entre comptes ni des comptes privés — ces règles s'appliquent en aval,
dans l'API.

---

## 1. Le trajet d'une requête de fil

```
Mobile                    API Node (3001)              Rust (3002)              PostgreSQL / Redis
  │                            │                            │                            │
  ├─ GET /api/neural-rank/recommendations?mode=for_you&limit=50&offset=0
  │                            │                            │
  │                            ├─ cache 30 s (withFeedCache) ?  ──── hit → réponse
  │                            │                            │
  │                            ├─ POST /recommendations ────►│
  │                            │   X-Service-Key             ├─ cache Redis reco ? ──► hit → IDs
  │                            │                            │
  │                            │                            ├─ profil lecteur (cache 300 s)
  │                            │                            ├─ collecte des candidats (10 sources)
  │                            │                            ├─ déduplication
  │                            │                            ├─ scoring (9 dimensions + ML + modulateurs)
  │                            │                            ├─ bandit d'exploration
  │                            │                            ├─ mise en forme (fils, étalement)
  │                            │                            ├─ mémorisation des impressions
  │                            │                            └─ mise en cache (TTL adaptatif)
  │                            │◄──── tweet_ids + threads ───┤
  │                            │
  │                            ├─ hydratation SQL (auteur, médias, compteurs)
  │                            ├─ masquage des contenus payants
  │                            └─ filtrage des comptes privés
  │◄─── tweets complets ───────┤
```

### Les quatre modes

| Mode | Surface | Particularité |
|---|---|---|
| `feed` | Fil d'abonnement | Jamais filtré par le shadowban |
| `for_you` | Fil principal de l'app | Mélange abonnements + découverte |
| `discover` | Découverte explicite | — |
| `trending` | Onglet Explorer | Score mélangé 40 % personnel / 60 % vélocité |

---

## 2. Trois étages de cache

Une requête peut être servie sans qu'aucun classement ne soit recalculé.

| Étage | Clé | Durée | Purge |
|---|---|---|---|
| API Node | `feed:neural:{user}:{params}` | 30 s | Publication, achat, **recalibration** |
| Moteur — classement | `twitninf:reco:{user}:{mode}` | 30–180 s (adaptatif) | Toute interaction sauf une vue |
| Moteur — profil lecteur | `twitninf:profile:{user}` | 300 s | Recalibration |

**TTL adaptatif du classement** — il dépend du lecteur, pas d'une constante :

```
PowerUser 45 s · Regular 90 s · Casual 180 s
  − 20 s si la tendance d'engagement dépasse 1,5
  plafonné à 60 s en trending, 120 s en discover
  plancher absolu : 30 s
```

Un lecteur actif voit son fil se renouveler plus vite qu'un lecteur
occasionnel — recalculer pour quelqu'un qui ne reviendra pas dans l'heure ne
sert à rien.

> **Piège connu** : une `View` n'invalide **pas** le cache de classement. Sinon
> le classement serait jeté à chaque tweet regardé, et la pagination avancerait
> dans un ordre qui n'est plus celui d'où venait la page précédente — sautant
> des tweets et en resservant d'autres.

---

## 3. Le profil du lecteur

Reconstruit toutes les 300 s, en requêtes parallèles (`join!`) :

- **Graphe social** — comptes suivis (max 1000), réciproques, second degré (max 200)
- **Historique** — 500 derniers likes, 300 retweets, réponses, favoris
- **Affinités** — 20 auteurs les plus likés sur 60 jours, normalisés
- **Rythme** — activité par heure (24) et par jour (7), heure la plus active
- **Style** — longueur moyenne des contenus consommés, mots-clés
- **Vu récemment** — set Redis, 24 h
- **Auteurs mis en sourdine** — via « pas intéressé », 30 jours
- **Vecteur de goût** — moyenne des embeddings des tweets likés (90 j, 40 max)

Le type de lecteur (`PowerUser` / `Regular` / `Casual`) est déduit de son
volume d'activité et pilote le TTL ci-dessus.

---

## 4. Collecte des candidats — 10 sources

Huit sources SQL en une seule requête (`UNION ALL`), plus deux hydratations.
Chaque source a son propre `LIMIT` : le vivier reste borné quelle que soit la
taille de la base.

| # | Source | Poids | Sélection | Limite |
|---|---|---|---|---|
| 1 | Tendance | 0.15 | Likes de la dernière heure | 400 |
| 2 | Graphe social | 0.12 | Comptes suivis, récents | 300 |
| 3 | Viral | 0.08 | Likes des 6 dernières heures | 250 |
| 4 | Découverte | 0.05 | Aléatoire parmi les 2000 plus récents | 150 |
| 5 | Temporel | 0.06 | Publiés à l'heure d'activité du lecteur ±1 h | 150 |
| 6 | Influenceur | 0.04 | Comptes vérifiés / premium | 150 |
| 7 | Personnalisé | 0.10 | Auteurs à forte affinité, 7 jours | 200 |
| 8 | Qualité | 0.02 | Meilleur taux d'engagement réel | 100 |
| 9 | **Sémantique** | — | Plus proches voisins du vecteur de goût (pgvector) | 40 |
| 10 | **Co-occurrence** | — | Auteurs co-likés par d'autres lecteurs | — |

**Filtres de visibilité** appliqués à toutes les sources : non supprimé,
`moderation_status = 'approved'`, non privé, auteur actif et non suspendu, pas
l'auteur lui-même, pas un compte banni. Une réponse doit avoir **au moins un
like** pour entrer, et son parent doit être affichable pour ce lecteur.

> La source 4 borne à 2000 lignes **avant** de mélanger. Un `ORDER BY RANDOM()`
> sur tout le corpus coûte proportionnellement au volume de la plateforme, sur
> chaque requête de fil — invisible aujourd'hui, premier maillon à céder en cas
> de montée en charge.

**Plafond par auteur sur le vivier : 12 tweets.** Sans lui, un compte prolifique
saturait le pool à lui seul — 32 tweets du même auteur sur 50 servis, relevé en
production le 2026-07-29. Diversifier une liste où un auteur occupe déjà les
deux tiers est impossible : le plafond doit agir *avant* le classement.

Puis **déduplication** : un tweet trouvé par plusieurs sources garde le poids le
plus élevé.

---

## 5. Le scoring

### 5.1 Neuf dimensions

Chaque tweet reçoit neuf scores dans [0,1] :

| Dim | Nom | Ce qu'elle mesure |
|---|---|---|
| **D1** | Vélocité d'engagement | Likes/commentaires/retweets rapportés au temps et aux vues |
| **D2** | Intelligence du contenu | Longueur, structure, médias, correspondance aux mots-clés du lecteur |
| **D3** | Graphe social | Abonnement direct, réciprocité, second degré, affinité |
| **D4** | Dynamique temporelle | Fraîcheur, et coïncidence avec le rythme du lecteur |
| **D5** | Prédiction comportementale | Ressemblance avec ce que ce lecteur consomme d'habitude |
| **D6** | Diversité du contenu | Écart par rapport à ce qui est déjà dans le fil |
| **D7** | Prédiction virale | Accélération de l'engagement (×1,2 si source Tendance) |
| **D8** | Profondeur de personnalisation | Signaux fins propres au lecteur |
| **D9** | Compréhension LLM | Qualité, thème et ton établis hors-ligne par l'annotateur |

### 5.2 Composition du score de base

```
score_global  = Σ (Dn × poids_n)
score_lecteur = poids personnels du lecteur appliqués à D1..D8
score_base    = score_global × 0,60 + score_lecteur × 0,40
```

Poids par défaut (somme = 1,00) :

```
D1 0,24 · D2 0,12 · D3 0,22 · D4 0,09 · D5 0,08
D6 0,06 · D7 0,06 · D8 0,03 · D9 0,10
```

D9 a été financée par un prélèvement sur les autres, pas par une inflation du
total : si la somme dérive, tous les scores montent et les seuils calibrés
ailleurs sautent.

### 5.3 Modulateurs multiplicatifs

Appliqués ensuite, dans cet ordre :

| Modulateur | Effet |
|---|---|
| Diversité par auteur | Décroît à mesure qu'un auteur revient dans le fil |
| **Diversité par thème** | Idem, sur le thème LLM |
| Pénalité de modération | Contenu signalé |
| **Shadowban** | ×1,00 / ×0,85 / ×0,45 / ×0,05 selon le niveau |
| Pénalité « poubelle » | Jusqu'à −60 % selon les signaux de spam |
| Fatigue d'exposition | Décroît avec le nombre de fois déjà vu par CE lecteur |
| Visibilité auteur | `users.algorithmic_visibility_multiplier`, borné [0, 3] |
| Toxicité | Pondérée par la confiance de l'annotation, plancher 0,15 |
| Qualité | Petit appui aux contenus bien notés |
| Abonnement | Bonus Plus / Pro |
| **Frein de vélocité** | ×0,5 pendant 1 h (voir §8) |

### 5.4 Le modèle CTR

Une régression logistique entraînée en continu (SGD) prédit la probabilité de
clic sur **15 caractéristiques** : les 9 dimensions, l'âge, le caractère
tendance, la présence de média, le log des abonnés, la fraîcheur,
l'accélération, et — seule à décrire *qui regarde* plutôt que *ce qui est
regardé* — l'activité du lecteur.

```
score = score × 0,60 + CTR_prédit × 0,40
```

**Activé seulement au-delà de 200 échantillons**, pour ne pas classer sur un
modèle qui n'a rien appris. Persisté sur disque tous les 50 nouveaux
échantillons, et **migré** si le nombre de caractéristiques change (les poids
appris sont conservés, le nouveau est initialisé à sa valeur par défaut).

### 5.5 Boost temps réel

Un like ou un rejet ajuste immédiatement l'auteur concerné pour **30 minutes** :
`+0,15` après un signal positif, `−0,08` après un négatif. C'est ce qui fait
qu'un geste se voit dès la page suivante, sans attendre le rechargement du
profil.

---

## 6. Le bandit d'exploration

**80 % exploitation, 20 % exploration.** Le pool d'exploitation prend les
meilleurs scores (plancher 0,30). Le pool d'exploration est classé par **UCB1** :

```
UCB1 = récompense_moyenne + √(2 × ln(impressions_totales) / (impressions_bras + 1))
```

Chaque *bras* est un auteur, à l'échelle de toute la plateforme. Un auteur
jamais servi reçoit le bonus d'incertitude maximal et passe devant — c'est
l'optimisme face à l'inconnu, la raison d'être d'un bandit.

Auparavant ce pool était un simple tirage aléatoire : le nom promettait plus
que le code ne faisait.

---

## 7. Mise en forme du fil

Une fois classé, le fil est réorganisé :

1. **Fils de discussion** — une réponse est précédée du tweet auquel elle
   répond (profondeur max 4). Réponses plafonnées à **25 %** du fil.
2. **Étalement par auteur** — max **3 tweets par auteur** sur une fenêtre
   glissante de **50 positions**. Un fil de trois tweets du même auteur compte
   pour trois.

Puis les **impressions sont mémorisées** : pour chaque tweet servi, le vecteur
de caractéristiques réellement utilisé est stocké 30 min en Redis. Sans ça, le
modèle CTR s'entraînerait sur des valeurs reconstruites après coup.

---

## 8. Ce que le client envoie en direct

### `POST /api/neural-rank/track`

```jsonc
{
  "tweetId": "uuid",
  "interactionType": "like",
  "dwellMs": 4200,           // temps passé, plafonné à 60 s
  "dwellMedia": "image",     // text | image | video
  "contentChars": 180,
  "videoDurationMs": null,
  "authorId": "uuid"         // indispensable pour "pas intéressé"
}
```

**Poids par type d'interaction :**

| Positif | | Négatif | |
|---|---|---|---|
| Retweet | +5,0 | Skip | −0,5 |
| Partage | +4,0 | Unlike | −1,0 |
| Commentaire | +3,5 | Unretweet | −2,0 |
| Intéressé | +3,0 | **Pas intéressé** | **−8,0** |
| Favori | +2,5 | Signalement | −12,0 |
| Vue de profil | +1,5 | Blocage | −20,0 |
| Like | +1,0 | | |
| Vue | +0,2 | | |

### Le temps passé n'est pas lu brut

Un temps de lecture est confondu avec la **longueur** du contenu : un pavé
survolé dure plus longtemps qu'un tweet court adoré. Le temps observé est donc
rapporté au temps que *ce contenu-là* demandait — sans quoi le classement
apprend que le public préfère les contenus longs, ce qui est une propriété du
chronomètre, pas du public.

### « Pas intéressé » a deux effets

1. Le tweet est marqué vu — il ne peut plus revenir.
2. **L'auteur est mis en sourdine pour ce lecteur** (30 jours).

Sans le second, le geste reste un bouton décoratif : il reste mille tweets du
même compte. Mesuré chez YouTube (Mozilla, 2022) : « pas intéressé » évite 11 %
des recommandations non voulues, « ne plus recommander cette chaîne » 43 %.

### Ce qu'une interaction déclenche

```
track
 ├─ enregistrement A/B (PostgreSQL, indépendant de Redis)
 ├─ marquage « vu » (24 h)
 ├─ invalidation du classement          (sauf pour une vue)
 ├─ si « pas intéressé » → sourdine de l'auteur
 ├─ si signal tranchant  → entraînement du modèle CTR
 │                       → boost temps réel de l'auteur (30 min)
 │                       → récompense du bras de bandit
 └─ si like              → co-occurrence d'auteurs (filtrage collaboratif)
```

---

## 9. Les boucles d'apprentissage en tâche de fond

| Boucle | Cadence | Rôle |
|---|---|---|
| **Balayage CTR** | 60 s | Les impressions expirées sans engagement deviennent les **exemples négatifs**. Sans elles le modèle n'a que des positifs et ne discrimine rien. Persiste le modèle tous les 50 échantillons. |
| **Auto-tuner** | continu | Au-delà de **500 échantillons**, réajuste les poids D1–D8 depuis ceux appris par le modèle CTR. Désactivé si un administrateur a fixé les poids à la main. |
| **Rattrapage d'embeddings** | 30 s | Calcule les vecteurs manquants par petits lots (798/798 aujourd'hui). |

Le modèle d'embeddings (`all-MiniLM-L6-v2`, 384 dimensions, ONNX local) est
chargé **en tâche de fond au démarrage** : le serveur écoute immédiatement et
les fonctions sémantiques s'activent quand il est prêt. Bloquer le démarrage
sur un téléchargement de 90 Mo ferait échouer le contrôle de santé du
déploiement.

---

## 10. Restriction de portée

Deux plans **séparés** — un compte peut être sain et un post écarté, ou
l'inverse.

### Au niveau du contenu

Un post qui franchit le seuil de signaux de spam (0,40), de toxicité établie
(0,60 avec confiance ≥ 0,60) ou de qualité plancher **reste en ligne, sur le
profil et dans le fil de ses abonnés** — il n'entre simplement plus dans les
surfaces de recommandation, avec un motif nommé et affichable à l'auteur.

Signaux détectés : densité de hashtags, mentions en masse, zéro engagement
malgré les vues, taux de signalement, spam de liens, surcharge d'émojis, texte
dupliqué, et **compte neuf publiant en rafale** (≤ 2 jours, ≥ 50 tweets).

### Au niveau du compte

Un registre d'**avertissements datés qui expirent seuls au bout de 90 jours**.
Rien à lever à la main, rien à oublier de lever.

| Niveau | Score | Surfaces fermées |
|---|---|---|
| Clean | ×1,00 | aucune |
| Monitoring | ×0,85 | aucune — palier d'alerte |
| Suppressed | ×0,45 | Tendances, Découverte |
| Ghosted | ×0,05 | toute recommandation |

**Le fil d'abonnement n'est fermé à aucun niveau, et un compte suivi n'est
jamais retiré de « Pour toi ».** Retirer en silence à quelqu'un un compte qu'il
a explicitement demandé est précisément ce que le mot « shadowban » désigne
péjorativement — et ça n'apporte rien, le lecteur ira le chercher sur le profil.

Les seuils dépendent du **domaine** de l'infraction, et ne s'additionnent pas
entre domaines : dix spams ne valent pas un appel à la violence. Une menace
coupe la recommandation dès le premier fait ; le spam laisse une marge de 7.

Un contenu écarté automatiquement émet désormais un avertissement sur le compte
(un seul par tweet), avec le domaine déduit de la catégorie réelle posée par
l'annotateur.

### Le frein de vélocité — à ne pas confondre

Un **×0,5 pendant 1 heure**, posé automatiquement après une suppression de
tweet, un changement d'avatar/bio, ou 10 tweets en 10 minutes. Ce n'est pas une
sanction : pas de motif, pas de registre, pas de date de retour. Le niveau du
compte peut rester `Clean` pendant que le frein est actif.

---

## 11. La recalibration manuelle

Paramètres → « Recalibrer l'algorithme ». **Jamais proposée automatiquement.**

3 tours de 6 cartes, triées au doigt. La sélection se fait dans l'espace des
embeddings, pas sur des étiquettes d'auteur ou de thème :

- **Tour 1 — couverture.** Les 6 cartes sont choisies pour être le plus
  éloignées possible les unes des autres, afin que n'importe quel goût trouve
  une prise.
- **Tours 2-3 — la frontière.** On cherche les contenus **à égale distance** de
  ce qui a été accepté et de ce qui a été refusé. Montrer les plus proches
  voisins de ce qui vient d'être aimé n'apprend rien : la réponse est connue
  d'avance.

Dans les deux cas, chaque carte est pénalisée si elle ressemble à une autre du
même tour — poser deux fois la même question sur six cartes en gâche une.

**Ce n'est jamais un like public** : aucune écriture dans `tweet_likes`, aucune
notification à l'auteur, aucun compteur qui bouge. La session produit un
vecteur de goût dédié, mélangé au vecteur naturel à **65 % / 35 %**, puis purge
les trois étages de cache pour que l'effet soit immédiat.

Mesuré en production : distance moyenne entre cartes 0,737 → 0,663 → 0,607 sur
les trois tours — le resserrement est réel et mesurable.

---

## 12. Ce qui reste faible

Par honnêteté, les limites connues :

- **Le corpus est mince.** 10 auteurs éligibles, dont un pèse plus que tous les
  autres réunis. Aucune diversification ne peut créer une variété qui n'existe
  pas encore en base.
- **Le modèle CTR réapprend.** Le passage à 15 caractéristiques a conservé les
  43 751 échantillons, mais tout élargissement futur repart d'un poids par
  défaut sur la nouvelle dimension.
- **Le bandit part de zéro.** Aucune récompense accumulée tant que les
  interactions ne se comptent pas en milliers par auteur.
- **Aucune mesure avant/après.** Il n'existe pas de tableau de bord comparant
  le CTR sur deux fenêtres. Tant qu'il n'existe pas, toute affirmation de gain
  est une impression, pas un résultat.

---

## Références

Le modèle de restriction de portée suit celui de TikTok (refonte 2023) :
inéligibilité au fil de recommandation distincte du retrait, avertissements
expirant à 90 jours, seuils par domaine, page « état du compte ».

La sélection de la recalibration suit la littérature sur l'élicitation de
préférences : sélection par entropie/variance en apprentissage actif, et le
résultat central de *Deep Rating Elicitation* (arXiv 2402.16327) — noter les
items isolément produit des sélections redondantes, il faut tenir compte des
interactions entre eux.
