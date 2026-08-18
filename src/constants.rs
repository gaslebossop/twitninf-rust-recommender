//! Constantes réellement lues par le moteur.
//!
//! Ce fichier portait ~150 constantes (poids D1–D8, TTL de cache par type
//! d'utilisateur, limites de candidats par source, fenêtres temporelles…).
//! Un audit exhaustif (recherche de chaque identifiant dans tout `src/`, hors
//! ce fichier) a montré qu'aucune d'entre elles n'était importée nulle part :
//! `algorithm/scoring.rs` n'a jamais fait `use crate::constants`, les poids
//! réels sont câblés en dur directement dans les fonctions de scoring, et
//! plusieurs valeurs avaient même dérivé de ce qui est réellement appliqué
//! (ex. l'ancien `DIVERSITY_MULT_2 = 0.88` contre `0.82` dans le code réel).
//!
//! Lire ce fichier pour comprendre le comportement du moteur induisait donc en
//! erreur. Les douze constantes ci-dessous sont les seules dont l'usage a été
//! vérifié par grep dans tout le dépôt — pour les poids de dimension, la
//! source de vérité est `algorithm/scoring.rs` directement.

/// Multiplicateur appliqué au score final d'un tweet écrit par un compte suivi,
/// dans les modes Feed et ForYou.
///
/// Il existe parce que D3 seul ne suffit pas : `0,55 × 0,22` pèse environ 0,12
/// sur un score borné à 1, quand D1 (engagement) en pèse trois fois plus. Sans
/// ce multiplicateur, un tweet viral d'un inconnu passe systématiquement devant
/// un abonnement, et l'abonnement ne sert quasiment à rien.
pub const FOLLOW_FEED_BOOST: f64 = 1.45;
pub const FOLLOW_MUTUAL_BOOST: f64 = 1.15;

/// Renfort supplémentaire tant que le compte n'a presque aucun historique.
///
/// À l'inscription les abonnements choisis sont le SEUL signal disponible :
/// aucun like, aucune vue, aucun auteur favori. Ce renfort s'efface au fur et à
/// mesure que les interactions réelles arrivent, pour ne pas figer durablement
/// le fil sur trois comptes choisis en dix secondes.
pub const COLD_START_FOLLOW_BOOST_MAX: f64 = 1.25;
pub const COLD_START_INTERACTION_FLOOR: f64 = 20.0;

/// Mode Trending uniquement : force de la réorganisation aléatoire pondérée par
/// score (technique Gumbel-max — voir `RecommenderService::score_all`).
///
/// Sans elle, Trending trie strictement par score et un rafraîchissement (même
/// avec le cache serveur ignoré) renvoie exactement le même ordre tant que rien
/// n'a changé dans les scores sous-jacents — c'est le seul mode qui saute la
/// réorganisation par bandit contextuel des autres modes. Cette constante ajoute
/// un bruit à `ln(score)` avant un nouveau tri : plus haute, plus le tirage se
/// rapproche d'un ordre aléatoire ; plus basse, plus il colle au tri strict.
/// 0.9 garde les tweets les mieux notés en tête la plupart du temps tout en
/// mélangeant nettement l'ordre exact et la composition du haut de page d'un
/// tirage à l'autre.
pub const TRENDING_SHUFFLE_TEMPERATURE: f64 = 0.9;

/// Nombre de tweets d'ouverture tirés avec une température réduite.
///
/// Une température unique traite la position 1 comme la position 50. Or ces
/// premières cartes ne sont pas des cartes comme les autres : ce sont elles qui
/// décident si la personne continue de faire défiler ou repart. Les tirer aussi
/// au hasard que le reste, c'est jouer à pile ou face l'intérêt de la page
/// entière. Elles sont donc tirées plus près du classement réel — le reste
/// garde la température pleine, qui est ce qui rend la suite imprévisible.
pub const TRENDING_HOOK_SIZE: usize = 6;

/// Vivier dans lequel ces tweets d'ouverture sont tirés.
///
/// Prendre simplement les `TRENDING_HOOK_SIZE` meilleurs donnerait la même
/// ouverture à chaque rafraîchissement — exactement le défaut corrigé par le
/// mélange. On échantillonne donc parmi un vivier plus large : la qualité reste
/// haute, la composition change d'un tirage à l'autre.
pub const TRENDING_HOOK_POOL: usize = 24;

/// Température du tirage d'ouverture (voir les deux constantes ci-dessus).
/// Nettement plus basse que la température de queue : l'ordre y colle au score.
pub const TRENDING_HOOK_TEMPERATURE: f64 = 0.35;

/// Mode Trending : renfort des tweets qui ont quelque chose à MONTRER.
///
/// Explorer est une grille d'images puis une lecture plein écran ; un tweet
/// sans visuel y occupe une carte de texte, qui retient moins l'attention à
/// surface égale. Le renfort reste petit — assez pour départager deux tweets de
/// vélocité comparable, jamais assez pour faire remonter un tweet faible.
pub const TRENDING_MEDIA_BOOST: f64 = 1.12;

/// Mode Trending UNIQUEMENT : renfort maximal des tweets proches du goût du
/// lecteur (`UserProfile::taste_vector`, moyenne des embeddings de ses likes).
///
/// ── Pourquoi seulement ici ────────────────────────────────────────────────
/// `ForYou` part déjà du graphe social et de l'affinité d'auteur ; `Discover`
/// fait exprès l'inverse (il déprécie les comptes suivis). Trending, lui, ne
/// regardait QUE la vélocité d'engagement : la page de découverte montrait donc
/// à tout le monde la même chose, quels que soient les goûts. Ce renfort la
/// personnalise sans lui retirer son rôle — la vélocité reste le premier
/// facteur, l'affinité ne fait que départager.
///
/// ── Pourquoi ce plafond ───────────────────────────────────────────────────
/// 1,18 est du même ordre que `TRENDING_MEDIA_BOOST` : de quoi trancher entre
/// deux tweets de vélocité comparable, jamais de quoi faire remonter un tweet
/// faible. Plus haut, la grille se refermerait sur ce que le lecteur aime déjà
/// — ce qui est précisément ce qu'une page d'exploration doit éviter.
pub const TRENDING_TASTE_BOOST_MAX: f64 = 1.18;

/// `exclude_seen` : nombre de candidats à conserver au minimum après filtrage.
///
/// Écarter tout ce que le lecteur a déjà vu est ce qui rend la page neuve d'une
/// visite à l'autre — mais sur un vivier maigre, ça peut le vider entièrement.
/// En dessous de ce seuil on renonce au filtrage : une page déjà vue vaut mieux
/// qu'une page vide, qui elle ne donne aucune raison de revenir.
pub const EXCLUDE_SEEN_MIN_REMAINING: usize = 30;

/// En dessous de ce nombre de candidats, le mode Trending rouvre sa fenêtre.
///
/// Ses fenêtres sont volontairement courtes (6 h) pour garder le haut du fil
/// frais. Ça suppose un flux de publication soutenu : quand il ne l'est pas, la
/// fenêtre ne contient presque rien et la page est identique toute la journée —
/// exactement ce qui empêche de revenir. On élargit alors une fois.
pub const TRENDING_MIN_POOL: usize = 60;

/// Facteur d'élargissement appliqué aux fenêtres courtes de Trending quand le
/// vivier est sous `TRENDING_MIN_POOL`.
pub const TRENDING_WIDEN_FACTOR: i32 = 6;
