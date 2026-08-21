# Passation — chantier « algo niveau TikTok/X » (branche `feat/algo-niveau-tiktok`)

> ⚠️ **Ce document n'a pas été écrit par l'agent qui a fait le travail.** L'agent ALGO a été coupé
> par épuisement de contexte avant d'écrire sa passation. Ce fichier a été **reconstitué après coup**
> à partir de `git log`, `git diff` et d'une relecture du code, par la session suivante (2026-08-21).
>
> Conséquence : tout ce qui est affirmé ici a été **revérifié** par le rédacteur — compilation, tests
> et banc de performance rejoués (§2). Ce qui n'a PAS pu l'être est signalé comme tel à chaque fois.
>
> ⚠️ **Quatre affirmations de la passation d'origine (`PASSATION-ALGO-ET-FIL2B.md`) se sont révélées
> périmées à la vérification** — dont le fameux dossier `dimensions/`. Voir §4 et §6.

---

## 1. État de la branche

- Branche : `feat/algo-niveau-tiktok`, partant de `main` (`ed721da`).
- **11 commits** (9 de l'agent + 2 de la session suivante), `+2991 / −218` sur 19 fichiers pour la
  part de l'agent.
- ✅ **Fusionné dans `main`, poussé et DÉPLOYÉ en production le 2026-08-21** (merge `3e43526`).
  Déploiement via `./deploy-vps.sh` : compilation sur le VPS, redémarrage de `rust-recommender`,
  `/health` → `db: ok`, `redis: ok`, `NRestarts = 0`.

Le 9ᵉ commit (`457ba2f`) n'est pas de l'agent : sa modification traînait **non commitée** dans
l'arbre au moment de la coupure. Elle a été relue, vérifiée et commitée telle quelle par la session
suivante — voir §5.

## 2. Preuves de compilation — relancées le 2026-08-21

| Commande | Résultat |
|---|---|
| `cargo check --all-targets` | **exit 0** — warnings préexistants seulement, aucune erreur |
| `cargo test` | **207 passed ; 0 failed** (199 de l'agent + 8 du correcteur de calibration) |
| `cargo run --release --example bench_scoring` | **15,96 ms → 5,06 ms, ×3,16** ; écart de score entre les deux chemins **exactement 0** |

Les warnings sont des `never used` préexistants (`utils/math.rs` : `normalize`, `safe_ratio`,
`sigmoid_scaled`, etc.), sans rapport avec le chantier.

## 3. Les commits, du plus récent au plus ancien

Les deux premiers (`be78533`, `457ba2f`) sont de la session suivante, pas de l'agent.

| Commit | Objet |
|---|---|
| `be78533` | **calibration** — `ml/calibrator.rs` : Platt sur les têtes, exposé sur `/admin/algo/eval` ; **non appliqué au classement** |
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
| 5 | Démarrage à froid | ✅ **déjà sur `main`** — `cold_start_follow_multiplier` : renfort aux abonnements décroissant jusqu'à `COLD_START_INTERACTION_FLOOR`. Non retouché par le chantier |
| 6 | Rétroaction négative | ✅ **fait** (`2329a5a` + `457ba2f`) |
| 7 | Diversité / anti-bulle | ✅ **largement en place** (préexistant) — **trois verrous** : vivier plafonné à `MAX_CANDIDATES_PER_AUTHOR = 12`, pénalité de score `diversity_multiplier`, plafond dur `MAX_PER_AUTHOR_PER_PAGE = 3` par page de 50 appliqué **à la construction, donc avant cache**. Plus `theme_diversity_multiplier` (sujet) et « un fil n'occupe qu'une entrée ». **Reste** : aucun quota explicite in/out-network. Rappel : pénalités anti-bulle volontairement retirées de Trending (`a9d74d5`) |
| 8 | Calibration des scores | ✅ **mesurée ET corrigée** (`be78533`) — `src/ml/calibrator.rs` : mise à l'échelle de Platt, exposée sur `/admin/algo/eval` (`calibration_gain`). ⚠️ **Rien n'est appliqué au classement** — c'est une mesure, la décision de brancher se prend sur ce chiffre. ⚠️ `src/calibration.rs` **n'est pas ça** : c'est la calibration de **goût à l'inscription** |
| 9 | Performance | ✅ **mesurée et REPRODUITE** — `cargo run --release --example bench_scoring` rejoué le 2026-08-21 : **15,96 ms → 5,06 ms, ×3,16**, et écart de score entre les deux chemins **exactement 0** (le gain ne vient pas d'un changement de résultat). Conforme aux ×3,2 annoncés |
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

### ✅ ~~Bloquant : l'API Node ne relaie pas `scores`~~ — **posé le 2026-08-21**
Le maillon manquait dans le dépôt `api`, hors du périmètre des deux agents. Il est écrit sur
`api` branche **`feat/relais-scores-reco`**, commit `26b4ae7` (non poussé, non déployé).

Les scores y sont restreints aux tweets **réellement servis** (l'hydratation en perd, l'injection
publicitaire en ajoute qui n'en ont pas) et attachés **dans le `producer`, donc avant
`withFeedCache`** — sinon une charge cachée sortirait sans eux et la question resterait muette pour
tout lecteur tombant sur un cache chaud.

✅ **Confirmé sur la production le 2026-08-21.** Appel réel à `/recommendations` sur le VPS : le
champ `scores` sort bien, avec des confiances de **0,215 et 0,083** — donc **sous le seuil de 0,45**
qui arme la question explicite. Le déclencheur peut réellement se déclencher, ce qu'aucune
vérification statique ne pouvait établir.

Relevé au passage sur ce même appel : `candidates_collected: 78`, `deduplicated_total: 63`,
`latency_ms: 82`. **Le vivier réel fait quelques dizaines de candidats, pas des milliers** — voir la
section pipeline plus bas, ce chiffre la tranche.

### ✅ ~~Poser un correcteur de calibration (grille #8)~~ — **posé le 2026-08-21** (`be78533`)
`src/ml/calibrator.rs` — mise à l'échelle de Platt, exposée sur `/admin/algo/eval`
(`calibration_gain` : ECE avant, ECE après, paramètres ajustés, pour les quatre têtes).

**Rien n'est appliqué au classement.** Brancher la correction déplace le fil de tout le monde, et le
gain affiché est ajusté PUIS mesuré sur la même fenêtre — c'est un **plafond**, pas une promesse.
La conduite est donc : regarder le chiffre sur un service vivant, puis décider.

### ✅ ~~Reproduire le banc~~ — **fait le 2026-08-21**
×3,16 mesuré, écart de score nul. Voir §2.

### ❌ ~~Décider du sort de `src/algorithm/dimensions/`~~ — **le dossier n'existe pas**
**L'affirmation de la passation d'origine était périmée.** `src/algorithm/dimensions/` a été supprimé
il y a longtemps par le commit `b5be392` (« remove dead code across the engine ») et **il est absent
de `main` comme de la branche**. Il n'y a rien à trancher.

⚠️ La note mémoire `rust-recommender-pieges` répète encore cette information périmée — à corriger.

### 🟠 Le seul vrai reste de la diversité (grille #7) : le quota in/out-network
Les **trois verrous** de diversité par auteur existent et sont préexistants (voir §4). Ce qui n'existe
pas, c'est un **quota explicite** entre contenu d'abonnements et contenu hors-réseau : aujourd'hui
l'appartenance au réseau agit par **multiplicateur de score** (`FOLLOW_FEED_BOOST`,
`FOLLOW_MUTUAL_BOOST`, renfort de démarrage à froid), pas par part réservée.

**Ce n'est délibérément pas fait ici, et la raison compte** : un quota est un plafond dur, et les
plafonds durs sont exactement ce qui produit des trous quand le vivier est petit — c'est déjà
documenté pour `MAX_PER_AUTHOR_PER_PAGE` (« le plafond n'est tenable que s'il existe au moins
`ceil(PAGE_WINDOW / MAX_PER_AUTHOR_PER_PAGE)` auteurs distincts »). Avec 97 % de la table `users`
issue de deux rafales scriptées, un quota 50/50 servirait surtout des pages courtes.

À faire **après** avoir mesuré la part in/out réellement servie aujourd'hui — le harnais d'éval
existe maintenant pour ça.

### 🟡 Pipeline multi-étages (grille #1) — **chiffré, et la réponse est non**
Le principe : au lieu de noter tous les candidats avec le même modèle, noter beaucoup de candidats
avec un modèle bon marché puis ne repasser le modèle cher que sur les survivants. C'est un
**entonnoir de coût**, rien d'autre.

Ton moteur a déjà **trois étages** (candidats → scoring → re-ranking `shape_feed`/`spread_by_author`).
Ce qui manquerait, c'est seulement le dédoublement de l'étage de scoring en léger + lourd.

**Mesuré en production le 2026-08-21 : `candidates_collected: 78`, `deduplicated_total: 63`,
`latency_ms: 82`.** Et le banc note 1700 candidats en 5 ms. L'entonnoir existe pour éviter de payer
cher sur un vivier énorme — ici le vivier fait quelques dizaines et le scoring complet est gratuit.
Découper ajouterait un modèle à entraîner, évaluer et maintenir pour supprimer un coût qui n'existe
pas, en introduisant le défaut propre à l'entonnoir : le ranker léger jette des tweets que le lourd
aurait remontés, et cette erreur est invisible et définitive.

À rouvrir si le vivier change d'ordre de grandeur (dizaines de milliers) ou si une tête ML devient
nettement plus lourde que la régression logistique actuelle. Le harnais d'éval (`37e7bc4`) permet
alors de le **prouver** au lieu de le parier.

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
