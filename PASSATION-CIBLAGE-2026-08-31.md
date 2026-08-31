# Ciblage du classement — chantier du 2026-08-31

Objectif donné : « l'algo le plus puissant, le plus ciblé, qui connaît le mieux
ses users ».

Ce document dit ce qui a été **mesuré** avant de toucher au code, ce qui a été
changé, et ce qui reste. Rien ici n'est une intention : chaque chiffre vient
d'un appel réel à la production du 2026-08-31.

> Ce chantier ne touche pas à la vitesse. Pour ça, voir
> [PASSATION-PERFORMANCE-2026-08-31.md](PASSATION-PERFORMANCE-2026-08-31.md).

---

## 1. Ce que la production disait

Appel réel `POST /recommendations`, `mode=for_you`, lecteur réel :

```
candidats collectés     74  →  61 dédupliqués  →  41 servis
total_available        993
freshness_score          0,021
diversity_score          0,179

sources : trending 33 · discovery 17 · personalized 3
          social_graph 0 · influencer 0 · temporal 0 · viral 0 · quality 0
```

Poids réellement servis (`/admin/algo/stats`, `auto_tuned: true`) :

```
d1 vélocité       0,232      d6 diversité      0,0008
d2 contenu        0,183      d7 viralité       0,189
d3 graphe social  0,082      d8 personnalis.   0,172
d4 temporel       0,039      d9 LLM            0,100
d5 comportemental 0,0008
```

Évaluation hors-ligne (`/admin/algo/eval`) — **toutes les AUC valent `null`** :

| tête | échantillons | taux de base | positifs réels |
|---|---|---|---|
| reply | 6 481 | 0,108 % | **7** |
| reject | 7 558 | 0,132 % | **10** |
| amplify | 7 561 | 0,278 % | **21** |
| fav | 6 625 | 2,234 % | 148 |

Et l'audience réelle, en base :

| | |
|---|---|
| lecteurs avec ≥ 5 likes sur 90 jours | **27** |
| lecteurs actifs sur 7 jours | **11** |
| tweets éligibles sur 30 jours | 382 |
| auteurs distincts sur 30 jours | 20 |
| tweets publiés sur 72 h | 34 |
| tweets porteurs d'un plongement | 1 117 |

**Conclusion du diagnostic : le classement n'était pas sous-puissant, il était
mal visé.** Avec onze lecteurs réellement actifs, aucun modèle global ne peut
apprendre quoi que ce soit — c'est pour ça que toutes les AUC sont nulles. La
puissance ne peut venir que de signaux qui marchent sur peu de données :
le contenu, le graphe, le goût explicite. Or c'est précisément ce que le
moteur avait éteint.

---

## 2. Les cinq défauts, et ce qui a été fait

### 2.1 Le réglage automatique dépersonnalisait le fil

`AutoTuner` lisait les coefficients bruts de la régression CTR comme des
« importances de dimension », **écrêtait les négatifs à un plancher de 0,001**,
puis normalisait.

D5 et D6 valaient exactement `0,000827` = `0.001 / somme × budget`. Leurs
coefficients appris étaient donc NÉGATIFS, et l'écrêtage les a transformés en
une part de 0,08 % — c'est-à-dire supprimés du classement. Pendant ce temps D3,
le graphe social du lecteur, tombait de 0,22 à 0,082 et la popularité
(D1 + D7) montait à 0,42.

Et la boucle se referme sur elle-même : plus de contenu populaire servi, plus
de contenu populaire cliqué, plus de poids sur la popularité.

**Trois corrections :**

1. **Un coefficient négatif ne devient plus un plancher.** Blanchir « cette
   dimension nuit au CTR » en « cette dimension vaut 0,08 % » jette
   l'information du signe et supprime la dimension. Elle garde désormais sa
   part **par défaut** : le modèle n'a rien de fiable à en dire.
2. **Rétrécissement vers les défauts** (`w = (1−α)·défaut + α·appris`, avec
   `α = n/(n+20 000)`). Les coefficients d'une régression logistique sur des
   traits **corrélés** ne sont pas des importances : D1 et D7 mesurent tous
   deux la vélocité, D2 et D8 la correspondance au contenu. Entre deux traits
   colinéaires, le coefficient se répartit arbitrairement et peut changer de
   signe. Le `K` est volontairement élevé parce que 9 219 échantillons venant
   de **onze** lecteurs, ce sont onze avis, pas neuf mille.
3. **Plancher de personnalisation à 0,40.** D3 + D5 + D8 + D10 ne peuvent pas
   descendre plus bas, quoi qu'apprenne le modèle. Le défaut vaut 0,45 : le
   plancher ne mord donc que dans le sens de la dépersonnalisation.

Le point de référence redevient le **défaut** et non l'état courant :
rétrécir vers l'état courant laissait dériver sans limite, par petits pas dont
aucun n'était aberrant. C'est comme ça que D3 est descendu de 0,22 à 0,082.

### 2.2 D10 — le signal le plus personnel était éteint sur le fil principal

`UserProfile::taste_vector` — la moyenne des plongements des tweets que ce
lecteur a aimés ou longuement lus — est le signal le plus personnel du moteur.
Il ne demande **aucun entraînement**, il parle dès le cinquième like, et
1 117 tweets en portent un.

Il ne servait qu'à deux choses : cibler les publicités, et renforcer l'onglet
Explorer. **`for_you`, le seul mode que les applications demandent, ne le
consultait jamais.**

C'est maintenant une dimension à part entière, à **0,12**, financée par un
prélèvement sur vélocité et viralité.

⚠ **Elle classe, elle ne mesure pas.** La similarité cosinus médiane entre un
lecteur et un candidat quelconque vaut ~0,65 — deux textes courts en français
partagent énormément de structure. Verser cette valeur brute ajouterait une
constante au score de presque tous les candidats sans déplacer un seul rang.
D10 rend donc la **position** du candidat dans le vivier, de 0 à 1.

Neutre (0,5) et jamais pénalisante quand la mesure manque : un tweet publié il
y a trois minutes n'a pas encore son plongement, et le rétrograder pour ça
reviendrait à filtrer la fraîcheur.

### 2.3 Les abonnements étaient réétiquetés « tendance »

`deduplicate` gardait le canal de plus fort **poids SQL** (tendance 0,15 >
graphe social 0,12). Le classement, lui, valorise l'inverse (`source_bonus` :
tendance 0,04 < graphe social 0,08). Les deux tables se contredisaient, et le
SQL gagnait.

Ce n'était pas un cas limite : la fenêtre « tendance » de `for_you` fait 72 h
et il se publie 34 tweets en 72 h. La source « tendance » ramassait donc TOUT
— d'où `social_graph: 0` dans les statistiques d'un lecteur qui suit 27 comptes
ayant publié 44 fois pendant la fenêtre. Chacun de ces tweets perdait la
moitié de son bonus de canal.

Une seule table désormais : `TweetSource::feed_bonus()`, celle du classement.

### 2.4 Trois têtes sur cinq pesaient sur du bruit

Le seuil d'activation comptait les **échantillons** (200), pas les
**événements**. Une régression logistique n'apprend pas d'un échantillon, elle
apprend d'un événement : `reply` était déclarée prête avec **sept** positifs
pour 22 traits.

`Head::is_ready()` exige désormais `MIN_POSITIVES = 30` en plus des 200
échantillons. La règle usuelle demande une dizaine d'événements par trait —
220 ici ; 30 est le minimum absolu en dessous duquel l'ajustement ne veut rien
dire, choisi indulgent pour qu'une tête parle dès qu'elle le peut
plausiblement. Ça écarte tout de même `reply`, `reject` et `amplify`, et
laisse `fav` (148 positifs) active.

### 2.5 Le classement voyait 16 % de ce qu'il aurait pu trier

**382 tweets étaient éligibles sur trente jours ; le vivier en contenait 61**,
et 41 étaient servis. Le classement retenait deux candidats sur trois : à ce
taux il ne classe plus, il transmet. Aucun réglage de poids ne rattrape ça.

La cause n'est pas un filtre trop sévère mais des fenêtres calibrées pour un
flux dense qui n'existe pas ici : « tendance » et « graphe social » regardent
72 h.

L'élargissement adaptatif, qui n'existait que pour Trending, vaut maintenant
pour **tous les modes** : on rouvre les fenêtres par paliers (×1, ×4, ×12)
jusqu'à `CANDIDATE_TARGET_POOL = 200`, soit quatre candidats par place servie.

Le premier palier vaut 1 : **sur un flux dense, rien ne change**, ni le vivier
ni le coût. L'élargissement ne se paie que là où il sert.

⚠ Élargir ne veut pas dire servir du vieux. La fenêtre décide de ce qu'on
REGARDE ; c'est D4 (demi-vie de 4 h) qui décide de ce qui remonte. Un vivier
large avec une décote de fraîcheur forte sert du frais quand il y en a et du
choix quand il n'y en a pas. Un vivier étroit, lui, sert ce qu'il a.

---

## 3. Les poids par défaut, avant et après

| | avant | après | |
|---|---|---|---|
| D1 vélocité | 0,24 | **0,18** | popularité |
| D2 contenu | 0,12 | 0,10 | |
| D3 graphe social | 0,22 | 0,22 | **personnel** |
| D4 temporel | 0,09 | 0,08 | |
| D5 comportemental | 0,08 | 0,08 | **personnel** |
| D6 diversité | 0,06 | 0,06 | |
| D7 viralité | 0,06 | **0,04** | popularité |
| D8 personnalisation | 0,03 | 0,03 | **personnel** |
| D9 LLM | 0,10 | 0,09 | |
| **D10 affinité de goût** | — | **0,12** | **personnel** |

Part personnelle : **0,45** contre 0,22 pour la popularité. C'est l'invariant
que `le_defaut_reste_majoritairement_personnel` verrouille, et que le plancher
de l'auto-réglage protège à 0,40.

---

## 3 bis. Mesuré après déploiement

Même appel, même lecteur, avant / après :

| | avant | après |
|---|---|---|
| candidats collectés | 74 | 74 |
| source `social_graph` | **0** | **37** |
| source `trending` | 33 | 1 |
| poids D5 comportemental | 0,0008 | **0,08** |
| poids D6 diversité | 0,0008 | **0,06** |
| poids D3 graphe social | 0,082 | **0,17** |
| poids D10 affinité de goût | — | **0,12** |
| part personnelle | **0,26** | **0,43** |
| part popularité (D1+D7) | 0,42 | **0,25** |
| tweets servis de moins de 72 h | **2** / 50 | **17** / 42 |
| tweets servis de moins de 7 j | 3 | **22** |
| âge médian servi | 413 h | **152 h** |

⚠ **La page est passée de 50 à 42 entrées.** C'est le prix de l'arrêt anticipé
de l'élargissement : un vivier plus étroit, mais nettement plus frais. Servir
cinquante tweets vieux de dix-sept jours n'était pas une meilleure page.

⚠ **Deux allers-retours ont été nécessaires, et le second corrigeait le
premier.** Élargir la collecte a d'abord EMPIRÉ la fraîcheur (2 tweets récents
sur 50, âge médian 17 jours) parce que ça a révélé un défaut que les fenêtres
courtes masquaient : D4 valait zéro passé trois jours, donc ne départageait
plus rien, pendant que D1 et D7 comptent un engagement qui s'accumule avec
l'âge. C'est la mesure APRÈS déploiement qui l'a montré — pas la revue de code.

---

## 4. Ce qui reste

### 🟠 La calibration mesurée n'est toujours pas appliquée
`ml/calibrator.rs` existe depuis le 2026-08-21, expose son gain sur
`/admin/algo/eval`, et **rien n'est appliqué au classement**. Le gain affiché
est ajusté puis mesuré sur la même fenêtre — c'est un plafond, pas une
promesse. Décision à prendre sur un service vivant.

### 🟠 L'évaluation ne peut rien conclure
Toutes les AUC sont `null` : 192 échantillons d'évaluation, 11 lecteurs actifs.
Tant que l'audience n'a pas grandi, **aucun changement d'algorithme ne peut
être validé par les chiffres** — seulement par le raisonnement et par des
invariants testés. C'est la limite honnête de tout ce document.

### 🟡 Le quota in/out-network
Toujours pas de part réservée entre abonnements et hors-réseau ; l'appartenance
agit par multiplicateur. Un plafond dur produirait des trous sur ce vivier.
À rouvrir quand le vivier tiendra la cible de 200 sans élargissement.

### 🔴 Le vrai plafond n'est plus l'algorithme
20 auteurs actifs, 731 tweets sur 36 jours, 11 lecteurs actifs sur 7 jours. Le
vivier plafonne à ~74 candidats parce que `MAX_CANDIDATES_PER_AUTHOR = 12`
multiplié par une vingtaine d'auteurs ne peut pas donner davantage — élargir
les fenêtres au-delà n'ajoute que de l'âge, ce que l'arrêt anticipé constate
désormais tout seul.

**Aucun réglage supplémentaire ne fera mieux à ce volume.** Ce qui bougerait
l'aiguille maintenant, c'est plus d'auteurs qui publient, pas plus de
dimensions.

### 🟡 Les 243 réponses sans like
Sur 30 jours, 243 réponses sont écartées faute d'un seul like — c'est le plus
gros filtre du CTE `visible` (748 tweets → 382 éligibles). La règle est
défendable (une réponse sans contexte est illisible), mais elle mériterait
d'être mesurée : sur un vivier maigre, elle coûte la moitié du corpus.

---

## 5. Vérification

- **292 tests** (241 avant les deux chantiers du jour), dont ceux qui
  verrouillent chaque correction ci-dessus : la dimension à coefficient négatif
  n'est plus supprimée, le plancher tient à tout niveau d'échantillons, D9 et
  D10 ne bougent pas, un tweet suivi ET populaire reste du graphe social, une
  tête à sept positifs reste muette, D10 départage assez pour compter.
- Aucun avertissement clippy nouveau (22, la ligne de base).
- Banc de scoring inchangé : l'ajout de D10 est une lecture de table de
  hachage par candidat.
