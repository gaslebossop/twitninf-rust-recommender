# Performance du moteur — chantier du 2026-08-31

Ce document raconte ce qui a été mesuré, ce qui a été changé, et ce qui reste.
Il ne remplace pas `ALGORITHME.md` (qui décrit *ce que* le moteur calcule) :
ici, rien ne change dans le classement. **Aucun tweet ne bouge d'une place.**

> ⚠ `OPTIMIZATIONS.md` est un document d'une autre époque : il décrit un
> `src/algorithm/dimensions/`, un `src/error.rs` et un `src/utils/validation.rs`
> qui n'existent plus. Ne pas s'y fier.

---

## 1. Le résultat, d'abord

Toutes les mesures portent sur un vivier de **1700 candidats** (l'ordre de
grandeur relevé en production), minimum de N passages, machine de dev.

| Étape (1700 candidats)                     | Avant      | Après     | Gain     |
| ------------------------------------------ | ---------- | --------- | -------- |
| Analyse du texte à l'ingestion             | 3,39 ms    | 0,71 ms   | ×4,8     |
| Scoring des 9 dimensions + modificateurs   | 5,39 ms    | 1,19 ms   | ×4,5     |
| `shape_feed` (mise en forme des fils)      | 0,55 ms    | 0,30 ms   | ×1,8     |
| `spread_by_author`, vivier varié (40 auteurs) | 1,98 ms | 0,86 ms   | ×2,3     |
| **`spread_by_author`, vivier réel (10 auteurs)** | **339,0 ms** | **1,51 ms** | **×224** |
| **Total processeur d'une recommandation**  | **≈ 348 ms** | **≈ 3,7 ms** | **≈ ×94** |

Le classement est inchangé, et c'est vérifié et non affirmé :

* `bench_scoring` compare l'ancien chemin et le nouveau **dans le même
  binaire** et affiche l'écart de score total : `0.000000000000` ;
* `bench_dimensions` affiche un checksum par poste, identique avant/après ;
* `l_etalement_indexe_rend_exactement_le_meme_fil` compare la nouvelle
  répartition à l'implémentation d'origine, recopiée dans le test, sur 132
  configurations tirées au sort ;
* 270 tests passent (241 avant le chantier, +29 écrits ici).

---

## 2. Ce qui coûtait cher — et pourquoi personne ne l'avait vu

Le réflexe est de regarder le scoring : c'est là qu'est « l'algorithme ». Le
profilage poste par poste (`examples/bench_dimensions.rs`, écrit pour ça) a
montré autre chose.

### 2.1 Le vrai gouffre : `spread_by_author`

Cette fonction ne calcule rien. Elle réordonne le fil pour qu'aucun auteur
n'occupe plus de 3 places sur 50. Elle cherchait, **pour chaque place du fil**,
le premier bloc dont l'auteur avait encore du quota, en balayant tout ce qui
restait à placer — et en reconstruisant une table de hachage à chaque essai
pour compter les auteurs du bloc.

Quand la fenêtre sature, ce balayage échoue jusqu'au bout, et un second le
reprend pour choisir le moins mauvais. Deux parcours complets par place.

Or la fenêtre **sature en permanence dès que le vivier compte moins de
17 auteurs distincts** — et la production en compte une dizaine (voir la note
de mémoire sur la volumétrie). Ce n'était donc pas un cas dégradé rare : c'était
le régime normal.

**339 ms.** Cent fois le coût du scoring des mêmes 1700 tweets.

La correction tient en une observation : deux blocs du même auteur demandant la
même chose sont interchangeables pour le quota. Si la tête de la file d'un
auteur ne tient pas, aucun de ses autres blocs ne tiendra ; et s'il faut forcer,
c'est la tête qu'on prend, puisque c'est le plus petit index. Il suffit donc de
regarder la tête de la file de chaque auteur — une dizaine de tests au lieu de
mille sept cents. Les blocs qui échappent au raisonnement (un fil à plusieurs
auteurs) restent examinés un par un ; ils sont rares.

S'y ajoute un **chemin rapide** : si le meilleur candidat restant tient, on le
prend sans rien regarder d'autre. Sans lui, l'indexation aurait fait *perdre* du
temps sur un vivier varié — on aurait payé un test par auteur là où le tout
premier bloc convenait. C'est ce qui explique que les deux régimes s'améliorent.

### 2.2 Trois formatages de date par candidat

D4 lisait l'heure et le jour de publication ainsi :

```rust
created_at.format("%H").to_string().parse::<u32>()
created_at.format("%w").to_string().parse::<usize>()
```

et D8 refaisait l'heure. Un formatage strftime complet, une allocation de
chaîne et un parsing d'entier, **trois fois par candidat**. À eux seuls, ces
trois appels pesaient plus que les huit dimensions réunies.

`Timelike::hour()` et `Datelike::weekday().num_days_from_sunday()` rendent
exactement les mêmes entiers. Deux tests le vérifient sur 48 heures et 14 jours
consécutifs, et un troisième garantit que les index restent dans les bornes des
tableaux `hourly_activity` (24) et `daily_activity` (7) — un dépassement serait
une panique, donc un 500 sur le fil du lecteur.

### 2.3 Cinquante recherches de sous-chaîne par candidat

D2 comptait les centres d'intérêt du lecteur présents dans le tweet ; D8 les
pondérait. Chacune faisait sa propre boucle de `str::contains` sur les ~30
mots-clés du profil : une cinquantaine de balayages du même texte par candidat,
**85 000 par recommandation**, pour une réponse qui tient en un seul passage.

Un automate Aho–Corasick est maintenant construit **une fois par profil** (les
mots-clés ne changent pas d'un candidat à l'autre) et porté par `UserProfile`.
Il donne exactement le même verdict que `contains` : « ce motif est-il une
sous-chaîne ? », sans frontière de mot ni casse.

Trois pièges, tous couverts par des tests :

* **Chevauchement.** Un mot-clé peut être contenu dans un autre (« chat » dans
  « chateau »). Avec des correspondances non chevauchantes, le plus long
  masquerait le plus court et D2 compterait une correspondance de moins.
  D'où `find_overlapping_iter`.
* **Ordre de sommation.** D8 additionne des flottants : l'ordre des termes *est*
  le résultat. L'automate rend ses motifs dans l'ordre du TEXTE, alors que la
  boucle les sommait dans l'ordre des MOTS-CLÉS. Les motifs trouvés sont donc
  mémorisés dans un masque de bits, puis parcourus dans l'ordre des mots-clés.
  D'où la borne de 64 mots-clés indexables (`top_words` en contient 30 au plus) :
  au-delà, on retombe sur le balayage linéaire plutôt que d'en perdre.
* **Motif vide.** `"".contains("")` vaut `true` : un mot-clé vide serait trouvé
  partout et décalerait les index. On refuse alors de construire l'automate.

À noter : un DFA forcé a été essayé et **rejeté sur mesure** (0,297 ms contre
0,270). Il perd le préfiltre, qui est justement ce qui rend l'automate très
rapide sur des mots-clés réels — absents de la plupart des tweets.

### 2.4 Le texte analysé six fois, et jusqu'à 50 `String` par tweet

`map_rows` balayait le texte de chaque candidat six fois (minuscules, émojis,
`!`, `?`, deux recherches d'URL) puis le découpait en mots — en allouant
**jusqu'à cinquante `String` par tweet**, soit 85 000 allocations par
recommandation. Pour des mots dont un seul consommateur se sert : le détecteur
de contenu poubelle, qui les compte et les déduplique.

Le module `src/content.rs` fait tout en un passage, et garde les mots comme des
bornes dans la chaîne déjà en minuscules. Trois détails ont compté :

* un test de plage unique (`u < 0x2600`) avant les six comparaisons de la
  détection d'émoji — aucun bloc émoji ne commence en dessous, ce qui écarte
  d'un coup lettres, chiffres, ponctuation et accents ;
* la capacité du vecteur de mots réservée d'avance (il partait de zéro et
  doublait cinq fois par tweet) ;
* une garde `contains("http")` avant les deux recherches d'URL.

**Effet de bord gratuit** : la chaîne en minuscules est désormais conservée sur
le tweet. Le scoring la recalculait à chaque recommandation servie ; il la lit
maintenant (0,134 ms → 0,003 ms).

### 2.5 Le reste

* **Détecteur de contenu poubelle appelé deux fois par candidat** — une fois par
  `score_all` pour l'admission par surface, une fois par le scoring pour sa
  pénalité. Le `ScoringContext` transporte le résultat déjà calculé.
* **`Utc::now()` lu par D1, D4 et D7 séparément.** En plus d'être trois fois
  trop cher, ça datait les trois dimensions d'un même tweet à trois instants
  différents. L'instant est maintenant celui du LOT : le score d'un candidat ne
  dépend plus de sa position dans la boucle.
* **Une table de hachage construite pour rien à chaque candidat** dans le
  détecteur : le signal « texte répétitif » dédupliquait tous les mots, y
  compris pour les tweets trop courts pour être concernés. Il sort maintenant
  dès qu'il a vu assez de mots différents — trois à treize sur un texte normal.
* **Balayages linéaires du profil** : `top_authors` était relu par D3, par D8 et
  par le bandit ; les mots vides l'étaient pour chaque mot de chaque tweet aimé.
  Tous indexés.
* **Milliers de `String` clonées pour rien** : compteurs de `score_all`,
  réordonnancement du bandit, ensemble des tweets déjà affichés dans
  `shape_feed`, clés de déduplication.

---

## 3. Deux choix de conception qui méritent d'être connus

### 3.1 Le repli, partout

Chaque valeur dérivée du profil (index d'appartenance, affinité d'auteur,
automate de mots-clés) **retombe sur le calcul direct quand son index est
vide**. Ce n'est pas de la coquetterie défensive : ces index ne sont pas
sérialisés vers Redis, donc un profil relu du cache arrive avec des index vides.
Sans repli, un `rebuild_indexes()` oublié ne produirait ni erreur ni log — juste
un fil où le boost d'abonnement et D3 valent zéro pour tout le monde.

Même chose côté tweet : `RawTweet::content_lower()` recalcule les minuscules si
l'analyse manque. C'est le cas de la centaine de tweets de test montés par
`..Default::default()` — sans ce repli, ils se comporteraient comme des tweets
au texte vide, et D2/D8 ne trouveraient plus jamais un centre d'intérêt.

**Règle** : un champ dérivé qu'on a oublié de remplir doit coûter du temps,
jamais fausser un classement.

### 3.2 Un hacheur rapide, et ses limites

`src/utils/fxhash.rs` remplace SipHash pour les ensembles dont **les clés ne
sont choisies par personne** : des UUID que la base a produits. SipHash protège
d'un attaquant qui choisirait ses clés pour faire dégénérer la table ; il n'y a
rien à protéger ici, et D3 pose trois questions d'appartenance par candidat.

⚠ **Ne pas l'étendre à une table dont les clés viennent d'une requête entrante**
(un pseudo, un texte de recherche, une en-tête). Là, SipHash n'est pas un luxe.

Un piège mérite d'être signalé, parce que le test l'a attrapé avant la mesure :
FxHash finit par une multiplication, et les bits BAS d'un produit portent très
peu d'entropie — or `hashbrown` choisit le seau avec les bits bas. Mille UUID
consécutifs ne tombaient que dans **32 seaux sur 1024**. La rotation ajoutée par
`rustc-hash` 2.0 remonte à 404, encore loin des 639 attendus, parce que nos clés
sont extrêmement structurées. D'où l'avalanche `splitmix64` en sortie : deux
multiplications, et la dispersion rejoint l'idéal. Le test
`les_uuid_se_dispersent` verrouille ça.

---

## 4. Les bancs d'essai

Trois exemples, tous en `--release` (en debug, les mesures ne disent rien) :

```bash
cargo run --release --example bench_scoring
```
Le A/B du scoring, index du profil activés ou non, **dans le même binaire**.
Affiche l'écart de score entre les deux chemins : il doit rester nul.

```bash
cargo run --release --example bench_dimensions
```
Le profilage poste par poste : c'est lui qui dit *où* part le temps. Il mesure
aussi les anciennes implémentations, gardées à côté des nouvelles, pour chiffrer
chaque gain. Il rend le **minimum** de 25 passages et non la médiane : le
travail mesuré est purement processeur et déterministe, donc tout ce qui dépasse
le minimum est du bruit — la médiane faisait bouger les chiffres de 30 % entre
deux exécutions.

```bash
cargo run --release --example bench_mise_en_forme
```
`shape_feed` et `spread_by_author`, dans les deux régimes (vivier varié et
vivier concentré). C'est celui qui a trouvé les 339 ms.

⚠ Le workflow CI compile désormais `--bins --examples`. Sans ça, ces bancs
pourrissent en silence dès qu'une signature change — et `main.rs` redéclare sa
propre liste de modules, donc un module ajouté à `lib.rs` et oublié là ne se
voit pas non plus avec `cargo test --lib`. C'est arrivé pendant ce chantier.

---

## 5. Ce qui reste sur la table

### 5.1 Redis est sérialisé derrière un mutex — non corrigé

`CacheManager` porte un `Arc<Mutex<MultiplexedConnection>>`. Or une
`MultiplexedConnection` est faite pour être **clonée** et utilisée en
concurrence : c'est tout son intérêt. Le mutex annule ce multiplexage et rend
strictement séquentiels les ~10 allers-retours Redis d'une recommandation
(shadowban, frein de vélocité, boosts temps réel, poids admin, bras du bandit,
affinité de goût, drapeau du modèle neuronal…).

Retirer le mutex donnerait la latence d'UN aller-retour au lieu de dix. **Ça n'a
pas été fait**, pour une raison précise : deux endroits dépendent de
l'atomicité que le mutex procure aujourd'hui — `cache_manager.rs` fait un
`get` puis un `del`/`zrem` sur la même clé (lignes ~226 et ~277), un motif de
consommation où deux appels concurrents pourraient tous deux lire avant que l'un
efface. Le corriger demande un `GETDEL` (Redis ≥ 6.2) ou un script Lua, et une
vérification contre un vrai Redis que je ne pouvais pas faire ici.

Les 15 autres blocs multi-commandes ont été audités : ce sont des
écriture-puis-`expire` ou des écritures indépendantes, sans dépendance à
l'atomicité.

### 5.2 Pistes mineures

* `keyword_hits` reste le premier poste du scoring (0,268 ms). Le banc
  l'exagère : tous ses tweets contiennent un mot-clé, donc le préfiltre déclenche
  toujours. En production, la plupart des tweets ne contiennent aucun mot-clé.
* D1 (0,122 ms) est dominé par trois transcendantes (`ln`, deux `exp`). Les
  approximer changerait les scores : à ne faire que si quelqu'un le demande.
* Le scoring reste une boucle séquentielle : `feed_shape` et le compte par
  auteur dépendent de l'ordre. Le paralléliser changerait le classement.

---

## 6. Fichiers touchés

| Fichier | Nature |
| --- | --- |
| `src/content.rs` | **nouveau** — analyse du texte en un passage |
| `src/utils/fxhash.rs` | **nouveau** — hacheur rapide pour clés internes |
| `examples/bench_dimensions.rs` | **nouveau** — profilage poste par poste |
| `examples/bench_mise_en_forme.rs` | **nouveau** — `shape_feed` + `spread_by_author` |
| `src/models.rs` | index d'affinité, automate de mots-clés, `ContentFeatures` |
| `src/algorithm/scoring.rs` | dates, `ScoringContext`, `KeywordHits` |
| `src/services/recommender.rs` | `spread_by_author`, `shape_feed`, `score_all`, `map_rows`, `deduplicate` |
| `src/shadowban/detector.rs` | sortie anticipée du signal « texte répétitif » |
| `src/shadowban/models.rs` | `GarbageSignals` devient `Copy` |
| `src/bandit/contextual.rs` | `knows_author` au lieu d'un balayage |
| `src/main.rs`, `src/lib.rs` | déclaration du module `content` |
| `Cargo.toml` | `aho-corasick` (déjà dans l'arbre via `regex`) |
| `.github/workflows/deploy.yml` | `--bins --examples` |
| `examples/bench_scoring.rs` | documentation remise à jour |
