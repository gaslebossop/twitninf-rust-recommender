//! Banc d'essai du chemin de scoring — le poste le plus cher d'une
//! recommandation, et le seul dont le coût suit la taille du vivier.
//!
//! ── Ce qu'il mesure, et pourquoi celui-là ───────────────────────────────────
//! `score_tweet_with_weights` est appelé UNE FOIS PAR CANDIDAT, sur un vivier
//! qui monte à ~1700 tweets. Tout ce qui y est linéaire en la taille du profil
//! devient quadratique à l'échelle de la requête. C'est exactement ce qui s'y
//! passait : D3 interroge trois listes d'appartenance (abonnements, mutuels,
//! second degré) et D6 une quatrième (tweets déjà vus), toutes stockées en
//! `Vec<String>`.
//!
//! ── Le A/B, sans deux binaires ──────────────────────────────────────────────
//! Les accesseurs d'appartenance replient sur le vecteur quand l'index est
//! vide (voir `UserProfile::member`). Appeler ou non `rebuild_indexes()` bascule
//! donc entre l'ancien comportement et le nouveau, dans le MÊME binaire, avec
//! la même charge et le même code — la comparaison ne peut pas être faussée par
//! un écart de compilation ou de jeu de données.
//!
//! ── Usage ───────────────────────────────────────────────────────────────────
//!   cargo run --release --example bench_scoring
//!
//! ⚠ En `--release` uniquement. En debug, les mesures sont dominées par
//! l'absence d'inlining et ne disent rien de la production.

use std::time::Instant;

use twitninf_recommender::admin::AlgoWeights;
use twitninf_recommender::algorithm::scoring::{score_tweet_with_weights, FeedShape};
use twitninf_recommender::models::{RawTweet, ScoredTweet, UserProfile};

/// Taille du vivier de candidats. Ordre de grandeur relevé en production :
/// huit sources plafonnées, dédupliquées, plus les parents de fil et les
/// candidats sémantiques.
const CANDIDATES: usize = 1_700;

/// Abonnements du lecteur. `SQL_SOCIAL` en charge jusqu'à 1000.
const FOLLOWING: usize = 1_000;

/// Second degré : `SQL_SECOND_DEGREE` en charge jusqu'à 200.
const SECOND_DEGREE: usize = 200;

/// Tweets vus dans les dernières 24 h (set Redis `twitninf:seen:<user>`).
const SEEN: usize = 500;

/// Auteurs distincts dans le vivier. Peu nombreux à cette échelle — c'est le
/// constat de production qui a motivé les trois verrous de diversité.
const AUTHORS: usize = 40;

fn author_id(i: usize) -> String {
    format!("00000000-0000-0000-0000-{:012}", i)
}

fn build_profile() -> UserProfile {
    let mut profile = UserProfile {
        following_ids: (0..FOLLOWING).map(author_id).collect(),
        // Un quart des abonnements sont mutuels : proportion plausible, et
        // surtout ce n'est pas ça qui décide du coût — c'est la longueur de la
        // liste balayée.
        mutual_follow_ids: (0..FOLLOWING / 4).map(author_id).collect(),
        second_degree_ids: (FOLLOWING..FOLLOWING + SECOND_DEGREE).map(author_id).collect(),
        seen_tweet_ids: (0..SEEN).map(|i| format!("seen-{i}")).collect(),
        profile_confidence: 0.8,
        engagement_velocity: 12.0,
        ..Default::default()
    };
    // Centres d'intérêt : D2 et D8 cherchent chacun de ces mots dans le texte
    // de chaque candidat.
    profile.top_words = (0..30).map(|i| (format!("interet{i}"), 5 + i as u32)).collect();
    profile.top_authors = (0..20).map(|i| (author_id(i), 1.0 - i as f64 * 0.04)).collect();
    profile
}

fn build_candidates() -> Vec<RawTweet> {
    (0..CANDIDATES)
        .map(|i| RawTweet {
            id: format!("tweet-{i}"),
            // Les auteurs tournent : une partie est suivie, une partie non.
            // Le cas défavorable pour un balayage linéaire est l'auteur NON
            // trouvé, qui force à parcourir la liste entière — on en met donc
            // une bonne moitié.
            user_id: author_id(i % (AUTHORS * 60)),
            content: format!(
                "Un contenu de longueur realiste pour le tweet {i}, avec un mot interet{} \
                 dedans et de quoi faire travailler l'analyse de texte. #sujet @quelquun",
                i % 40
            ),
            created_at: chrono::Utc::now() - chrono::Duration::minutes((i % 4_000) as i64),
            view_count: (i % 900) as i64,
            like_count: (i % 60) as i64,
            comment_count: (i % 11) as i64,
            retweet_count: (i % 7) as i64,
            likes_1h: (i % 5) as i64,
            likes_6h: (i % 13) as i64,
            comments_1h: (i % 3) as i64,
            retweets_1h: (i % 2) as i64,
            has_media: i % 3 == 0,
            hashtag_count: (i % 3) as i32,
            mention_count: (i % 2) as i32,
            content_length: 150,
            author_followers: (i % 5_000) as i64,
            author_account_age_days: 200,
            author_tweet_count: 300,
            viewer_impressions: (i % 4) as i64,
            ..Default::default()
        })
        .collect()
}

/// Un passage complet de scoring, tel que `score_all` l'exécute.
fn run(profile: &UserProfile, tweets: &[RawTweet], weights: &AlgoWeights) -> (f64, u128) {
    let mut feed: Vec<ScoredTweet> = Vec::with_capacity(tweets.len());
    let mut shape = FeedShape::empty();
    let start = Instant::now();
    for tweet in tweets {
        let scored = score_tweet_with_weights(tweet, profile, 0, shape, weights);
        shape.push(tweet.has_media);
        feed.push(scored);
    }
    let elapsed = start.elapsed().as_micros();
    // Somme rendue pour que rien ne puisse être éliminé comme code mort.
    let checksum = feed.iter().map(|s| s.score).sum();
    (checksum, elapsed)
}

fn main() {
    let weights = AlgoWeights::default();
    let tweets = build_candidates();

    let sans_index = build_profile();
    let mut avec_index = build_profile();
    avec_index.rebuild_indexes();

    println!("Banc d'essai du scoring");
    println!("  candidats        : {CANDIDATES}");
    println!("  abonnements      : {FOLLOWING}");
    println!("  second degre     : {SECOND_DEGREE}");
    println!("  tweets deja vus  : {SEEN}");
    println!();

    // Chauffe : première allocation, remplissage des caches. Sans elle, la
    // première mesure porte le coût de la mise en route.
    let _ = run(&sans_index, &tweets, &weights);
    let _ = run(&avec_index, &tweets, &weights);

    const ROUNDS: usize = 7;
    let mut avant = Vec::with_capacity(ROUNDS);
    let mut apres = Vec::with_capacity(ROUNDS);
    let mut check_avant = 0.0;
    let mut check_apres = 0.0;

    // Alterné, pas l'un après l'autre : une dérive de fréquence du processeur
    // pendant la mesure toucherait alors les deux également au lieu de
    // n'avantager que celui qui passe en second.
    for _ in 0..ROUNDS {
        let (c, t) = run(&sans_index, &tweets, &weights);
        avant.push(t);
        check_avant = c;
        let (c, t) = run(&avec_index, &tweets, &weights);
        apres.push(t);
        check_apres = c;
    }

    avant.sort_unstable();
    apres.sort_unstable();
    let median = |v: &[u128]| v[v.len() / 2] as f64 / 1000.0;
    let m_avant = median(&avant);
    let m_apres = median(&apres);

    println!("  balayage lineaire (sans index) : {m_avant:8.2} ms  (min {:.2}, max {:.2})",
             avant[0] as f64 / 1000.0, avant[ROUNDS - 1] as f64 / 1000.0);
    println!("  table de hachage (avec index)  : {m_apres:8.2} ms  (min {:.2}, max {:.2})",
             apres[0] as f64 / 1000.0, apres[ROUNDS - 1] as f64 / 1000.0);
    println!();
    println!("  gain : x{:.2}  ({:+.1} %)", m_avant / m_apres,
             (m_apres - m_avant) / m_avant * 100.0);
    println!();

    // La preuve que la comparaison est honnête : les deux chemins doivent
    // produire EXACTEMENT le même classement. Un gain obtenu en changeant le
    // resultat ne serait pas un gain.
    let ecart = (check_avant - check_apres).abs();
    println!("  ecart de score total entre les deux chemins : {ecart:.12}");
    assert!(
        ecart < 1e-9,
        "les deux chemins doivent produire le meme classement, ecart={ecart}"
    );
    println!("  → identiques : le gain ne vient pas d'un changement de resultat.");
}
