# Passation — chantier « algo niveau TikTok/X » (branche `feat/algo-niveau-tiktok`)

> ⚠️ **Ce document n'a pas été écrit par l'agent qui a fait le travail.** L'agent ALGO a été coupé
> par épuisement de contexte avant d'écrire sa passation. Ce fichier a été **reconstitué après coup**
> à partir de `git log`, `git diff` et d'une relecture du code, par la session suivante (2026-08-21).
>
> Conséquence : les **preuves de compilation ci-dessous ont été relancées et vérifiées** par le
> rédacteur, mais les **chiffres de performance** proviennent des messages de commit de l'agent et
> **n'ont pas été reproduits**. Les traiter comme des affirmations à vérifier, pas comme des mesures.

---

## 1. État de la branche

- Branche : `feat/algo-niveau-tiktok`, partant de `main` (`ed721da`).
- **9 commits**, `+2991 / −218` sur 19 fichiers.
- Arbre **propre**. **Rien n'a été poussé** (la branche n'existe pas sur `origin`), **rien n'a été
  fusionné dans `main`**, **rien n'a été déployé**.

Le 9ᵉ commit (`457ba2f`) n'est pas de l'agent : sa modification traînait **non commitée** dans
l'arbre au moment de la coupure. Elle a été relue, vérifiée et commitée telle quelle par la session
suivante — voir §5.

## 2. Preuves de compilation — relancées le 2026-08-21

| Commande | Résultat |
|---|---|
| `cargo check --all-targets` | **exit 0** — 39 warnings, aucune erreur |
| `cargo test` | **199 passed ; 0 failed** ; 0 ignored |

Les warnings sont des `never used` préexistants (`utils/math.rs` : `normalize`, `safe_ratio`,
`sigmoid_scaled`, etc.), sans rapport avec le chantier.

## 3. Les commits, du plus récent au plus ancien

| Commit | Objet |
|---|---|
| `457ba2f` | **refus sur soi-même** — un `skip`/`report`/`block` avec `author_id == user_id` ne porte plus sur le compte |
| `ba41d8d` | **perf scoring** — `FeedShape` remplace un recomptage du fil par candidat ; une seule minuscule par tweet ; `score_tweet_ml` (morte) supprimée |
| `de1f0ac` | **contrat** — `interaction_type: "open"` (poids 2,0) + `scores: [{tweet_id, score, confidence}]` dans `/recommendations` |
| `cdc234f` | **biais de position** — « tour peu profonde » (YouTube) : entraîner avec le rang réel, prédire à rang fixe |
| `37e7bc4` | **évaluation** — `src/eval.rs` : AUC, log-loss, ECE + courbe de fiabilité, NDCG@k, en validation progressive |
| `1351406` | **multi-objectif** — 2 têtes ajoutées (amplification, rejet) ; le mélange passe d'un `match` à une somme pondérée renormalisée |
| `2329a5a` | **rétroaction négative** — signalement et blocage ne marquaient même pas le tweet vu |
| `ea9e4d4` | **bandit** — frontière exploit/explore brouillée + trois balayages linéaires |
| `29b2d2a` | **perf profil** — indexation du graphe social au lieu d'un balayage par tweet |

Les messages de commit sont longs et portent le raisonnement complet (méthode, alternative écartée,
mesure). Les lire avant de toucher au code : ils contiennent l'essentiel de ce qu'aurait dit la
passation manquante.

## 4. La grille « niveau TikTok/X » — où en est-on

Grille reprise de `PASSATION-ALGO-ET-FIL2B.md` §6.

| # | Critère | État |
|---|---|---|
| 1 | Pipeline multi-étages | ⚠️ **partiel** — toujours un vivier → un score, pas de ranker léger/lourd séparés |
| 2 | Multi-objectif | ✅ **fait** (`1351406`) — 4 têtes : CTR, dwell, amplification, rejet ; rejet **soustraite** |
| 3 | Signaux temps réel | ✅ préexistant, non retouché |
| 4 | Exploration / bandit | ✅ **branché sur `for_you`** (`recommender.rs:1138`) ; frontière corrigée (`ea9e4d4`). Trending ne passe **volontairement pas** par le bandit |
| 5 | Démarrage à froid | ⚠️ **non traité ici** — voir la branche séparée `feat/poids-abonnements-coldstart` |
| 6 | Rétroaction négative | ✅ **fait** (`2329a5a` + `457ba2f`) |
| 7 | Diversité / anti-bulle | ❌ **non traité** — rappel : les pénalités anti-bulle ont été volontairement retirées du mode Trending (`a9d74d5`), ne pas les y remettre sans le dire |
| 8 | Calibration des scores | ⚠️ **mesurée, pas corrigée** — `eval.rs` sort l'ECE et la courbe de fiabilité, mais aucun correcteur Platt/isotonique n'est posé. ⚠️ `src/calibration.rs` **n'est pas ça** : c'est la calibration de **goût à l'inscription** (tours de tweets aimés) |
| 9 | Performance | ✅ **mesurée** — banc `examples/bench_scoring.rs`. Chiffres annoncés (non reproduits) : 17,67 ms → 5,49 ms par recommandation, ×3,2 |
| 10 | Évaluabilité | ✅ **fait** (`37e7bc4`) — c'était le trou le plus grave ; exposé sur `GET /admin/algo/eval` |

## 5. Le commit `457ba2f` — ce qu'il faut savoir

Il ne vient pas de l'agent ALGO mais de son homologue côté app : l'audit de `FeedGutterScreen` a
trouvé que le menu « … » proposait « Signaler / Ignorer / Bloquer » sur **ses propres tweets**
(liste figée par un `useCallback` — corrigé côté app en `9c61951`). Des événements
`author_id == user_id` ont donc pu partir en production.

La garde est posée **côté moteur** délibérément : un correctif d'écran ne protège que la version
installée, et l'ancienne continuera d'émettre pendant des semaines.

Le geste reste honoré **pour le tweet** ; il ne porte simplement plus **sur le compte** (sourdine,
CTR, dwell ferme, co-occurrence passent par `author_id_for_account = None`).

## 6. Ce qui reste à faire — par impact décroissant

### 🔴 Bloquant : l'API Node ne relaie pas `scores`
`de1f0ac` expose `scores` sur `/recommendations`, et l'app sait déjà le consommer
(`withRecommendationScores`, commit app `f93c648`). **Le maillon du milieu manque.**

- `api/src/services/rustRecommenderClient.js` → `getRecommendations()` (~l. 219) construit son objet
  de retour **sans** `result.scores`.
- `api/src/routes/neuralRankRoutes.js` → le `producer` (~l. 552‑600) hydrate les tweets puis
  renvoie `data: { recommendations, count, ... }` **sans** `scores`.

Tant que ce relais n'existe pas, `_recommendation_confidence` vaut 0 côté app, et le déclencheur de
la question explicite (`HESITATION_CEILING = 0.45`) **ne peut jamais s'armer** — l'écran retombe sur
son heuristique de silence. Deux fonctionnalités livrées des deux côtés pour rien.

⚠️ Attacher les scores **avant** `withFeedCache`, sinon une charge cachée sortira sans eux.

### 🟠 Décider du sort de `src/algorithm/dimensions/`
`d1_*`..`d8_*` — **toujours du code mort, jamais compilé** : `src/algorithm/mod.rs` n'expose que
`d9_llm_understanding`, `dwell`, `scoring`, `trending`. L'agent n'y a pas touché et n'a pas tranché.
Le supprimer ou le brancher, mais ne pas le laisser piéger le prochain lecteur.

### 🟠 Diversité / anti-bulle (grille #7)
Non traité. Plafond par auteur, plafond par sujet, mélange in/out-network sur `for_you`.

### 🟡 Poser un correcteur de calibration (grille #8)
La mesure existe maintenant. Si l'ECE est mauvaise, Platt ou isotonique sur les têtes.

### 🟡 Pipeline multi-étages (grille #1)
Le plus gros morceau, et le moins urgent : à la volumétrie actuelle (97 % de la table `users` vient
de deux rafales scriptées) un vivier → un score tient encore.

### 🟢 Reproduire le banc
`cargo run --release --example bench_scoring` — les chiffres du §4 n'ont pas été revérifiés.

## 7. Rappels qui n'ont pas changé

- **Les apps appellent `mode=for_you`**, jamais `feed`. Une amélioration hors de ce chemin n'a aucun
  effet en production.
- Les poids d'exécution viennent de `AlgoWeights`, chargé depuis la clé Redis `admin:algo:weights`.
  **Si la clé existe, modifier `AlgoWeights::default()` ne fait rien.** Un test verrouille la somme
  des poids à 1,0.
- `/recommendations` est en **POST**, en-tête **`X-Service-Key`** (= `INTERNAL_SECRET`),
  sur `127.0.0.1:3002`. Unité systemd : **`rust-recommender.service`**
  (`twitninf-rust-recommender.service` est un doublon périmé, à laisser éteint).
- Le repli JS du client Node est **silencieux** sur `/track` : un moteur injoignable ne produit
  aucune erreur visible, seulement une baisse de qualité. Premier suspect de tout diagnostic.
  (Sur `/for-you`, le repli a été supprimé : la panne sort en 503.)
- `deploy-vps.sh` compile **sur le VPS** — compiler sous Windows produit un binaire Windows.
  **Ne déployer que sur demande explicite.**
- Documents à lire avec méfiance, la doc peut mentir sur le code : `ALGORITHME.md`,
  `BENCHMARK_VS_TWITTER.md`, `OPTIMIZATIONS.md`, `ROADMAP_BETA.md`.
