//! Profilage POSTE PAR POSTE du chemin de scoring.
//!
//! `bench_scoring` mesure le total ; celui-ci dit OU part le temps. Sans lui,
//! optimiser revient a parier — et le pari aurait ete perdu : les deux postes
//! les plus chers ne sont aucune des huit dimensions, mais du formatage de
//! date et une table de hachage construite pour rien.
//!
//!   cargo run --release --example bench_dimensions

use std::time::Instant;

use twitninf_recommender::admin::AlgoWeights;
use twitninf_recommender::algorithm::d9_llm_understanding::{
    d9_llm_understanding, quality_boost, toxicity_penalty,
};
use twitninf_recommender::algorithm::scoring::*;
use twitninf_recommender::content::analyze_content;
use twitninf_recommender::ml::user_weights::UserDimensionWeights;
use twitninf_recommender::models::{KeywordHits, RawTweet, ScoredTweet, UserProfile};
use twitninf_recommender::shadowban::{GarbageContentDetector, ShadowbanEnforcer};

const CANDIDATES: usize = 1_700;
const FOLLOWING: usize = 1_000;
const SECOND_DEGREE: usize = 200;
const SEEN: usize = 500;
const AUTHORS: usize = 40;
const ROUNDS: usize = 25;

fn author_id(i: usize) -> String {
    format!("00000000-0000-0000-0000-{:012}", i)
}

fn build_profile() -> UserProfile {
    let mut profile = UserProfile {
        following_ids: (0..FOLLOWING).map(author_id).collect(),
        mutual_follow_ids: (0..FOLLOWING / 4).map(author_id).collect(),
        second_degree_ids: (FOLLOWING..FOLLOWING + SECOND_DEGREE)
            .map(author_id)
            .collect(),
        seen_tweet_ids: (0..SEEN).map(|i| format!("seen-{i}")).collect(),
        profile_confidence: 0.8,
        engagement_velocity: 12.0,
        ..Default::default()
    };
    profile.top_words = (0..30)
        .map(|i| (format!("interet{i}"), 5 + i as u32))
        .collect();
    profile.top_authors = (0..20)
        .map(|i| (author_id(i), 1.0 - i as f64 * 0.04))
        .collect();
    profile.rebuild_indexes();
    profile
}

/// L'analyse de texte telle que `map_rows` la faisait : six balayages du même
/// texte et jusqu'à cinquante `String` allouées par tweet. Gardée ici, et
/// nulle part ailleurs, pour chiffrer ce que le passage unique fait gagner.
fn analyse_de_reference(content: &str) -> usize {
    let content_lower = content.to_lowercase();
    let emoji_count = content
        .chars()
        .filter(|c| {
            let u = *c as u32;
            (0x1F300..=0x1FAFF).contains(&u)
                || (0x2600..=0x27BF).contains(&u)
                || (0x1F000..=0x1F0FF).contains(&u)
                || (0xFE00..=0xFE0F).contains(&u)
                || u == 0x2B50
                || u == 0x2764
        })
        .count();
    let exclamation_count = content.matches('!').count();
    let question_count = content.matches('?').count();
    let url_count =
        content_lower.matches("http://").count() + content_lower.matches("https://").count();
    let words: Vec<String> = content_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(String::from)
        .take(50)
        .collect();
    emoji_count + exclamation_count + question_count + url_count + words.len()
}

fn build_candidates() -> Vec<RawTweet> {
    (0..CANDIDATES)
        .map(|i| {
            let content = format!(
                "Un contenu de longueur realiste pour le tweet {i}, avec un mot interet{} \
                 dedans et de quoi faire travailler l'analyse de texte. #sujet @quelquun",
                i % 40
            );
            let text = analyze_content(&content);
            RawTweet {
                id: format!("tweet-{i}"),
                user_id: author_id(i % (AUTHORS * 60)),
                content,
                text,
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
            }
        })
        .collect()
}

/// Minimum de `ROUNDS` passages complets sur tout le vivier, en millisecondes.
///
/// La somme rendue par `f` est accumulee et affichee : sans elle, LLVM est
/// libre de supprimer purement et simplement le calcul mesure.
fn mesure<F: FnMut(&RawTweet) -> f64>(tweets: &[RawTweet], mut f: F) -> (f64, f64) {
    let mut temps = Vec::with_capacity(ROUNDS);
    let mut somme = 0.0;
    for t in tweets {
        somme += f(t);
    }
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let mut s = 0.0;
        for t in tweets {
            s += f(t);
        }
        temps.push(start.elapsed().as_nanos());
        somme = s;
    }
    // Le MINIMUM, pas la mediane : le travail mesure est purement processeur
    // et deterministe, donc toute mesure au-dessus du minimum est du bruit
    // ajoute (preemption, migration de coeur, changement de frequence). La
    // mediane melangeait le signal a ce bruit et faisait bouger les chiffres
    // de 30 % entre deux executions.
    temps.sort_unstable();
    (temps[0] as f64 / 1_000_000.0, somme)
}

fn main() {
    let weights = AlgoWeights::default();
    let profile = build_profile();
    let tweets = build_candidates();
    let lowers: Vec<String> = tweets.iter().map(|t| t.content.to_lowercase()).collect();
    let hits: Vec<KeywordHits> = lowers.iter().map(|l| profile.keyword_hits(l)).collect();
    let detector = GarbageContentDetector::new();
    let enforcer = ShadowbanEnforcer::new();
    let now = chrono::Utc::now();

    // Garde-fou du banc lui-même : si l'automate et le balayage linéaire
    // divergeaient, tout ce qui suit comparerait deux calculs différents.
    for (l, h) in lowers.iter().zip(&hits) {
        assert_eq!(*h, profile.keyword_hits_linear(l));
    }

    println!("Profilage du scoring — {CANDIDATES} candidats, minimum de {ROUNDS} passages\n");

    let mut postes: Vec<(&'static str, f64)> = Vec::new();

    macro_rules! poste {
        ($nom:literal, $f:expr) => {{
            let (ms, chk) = mesure(&tweets, $f);
            println!("  {:<34} {ms:7.3} ms   (checksum {chk:.4})", $nom);
            postes.push(($nom, ms));
        }};
    }

    poste!("analyze_content (ingestion)", |t: &RawTweet| analyze_content(
        &t.content
    )
    .word_count() as f64);
    poste!("ingestion, version d'avant", |t: &RawTweet| analyse_de_reference(
        &t.content
    ) as f64);
    poste!("content_lower() (au scoring)", |t: &RawTweet| t
        .content_lower()
        .len() as f64);
    poste!("to_lowercase() (avant)", |t: &RawTweet| t
        .content
        .to_lowercase()
        .len() as f64);
    let mut k = 0usize;
    poste!("keyword_hits (automate)", |_t: &RawTweet| {
        let v = profile.keyword_hits(&lowers[k]).matches as f64;
        k = (k + 1) % CANDIDATES;
        v
    });
    let mut k2 = 0usize;
    poste!("keyword_hits (lineaire, avant)", |_t: &RawTweet| {
        let v = profile.keyword_hits_linear(&lowers[k2]).matches as f64;
        k2 = (k2 + 1) % CANDIDATES;
        v
    });
    poste!("D1 engagement velocity", |t: &RawTweet| d1_engagement_velocity(
        t, now
    )
    .0);
    let mut i = 0usize;
    poste!("D2 content intelligence", |t: &RawTweet| {
        let v = d2_content_intelligence(t, &profile, &hits[i]);
        i = (i + 1) % CANDIDATES;
        v
    });
    poste!("D3 social graph", |t: &RawTweet| d3_social_graph(t, &profile).0);
    poste!("D4 temporal dynamics", |t: &RawTweet| d4_temporal_dynamics(
        t, &profile, now
    ));
    poste!("D5 behavioral prediction", |t: &RawTweet| {
        d5_behavioral_prediction(t, &profile)
    });
    poste!("D6 content diversity", |t: &RawTweet| d6_content_diversity(
        t,
        &profile,
        FeedShape::empty()
    ));
    poste!("D7 viral prediction", |t: &RawTweet| d7_viral_prediction(t, now));
    let mut j = 0usize;
    poste!("D8 personalization depth", |t: &RawTweet| {
        let v = d8_personalization_depth(t, &profile, &hits[j]);
        j = (j + 1) % CANDIDATES;
        v
    });
    poste!("D9 llm understanding", |t: &RawTweet| d9_llm_understanding(t));
    poste!("garbage detector", |t: &RawTweet| detector.detect(t).score());
    poste!("shadowban enforcer", |t: &RawTweet| enforcer
        .apply_to_score(0.5, t.author_shadowban_level));
    poste!("moderation penalty", |t: &RawTweet| moderation_penalty(t));
    poste!("quality + toxicity", |t: &RawTweet| quality_boost(t)
        + toxicity_penalty(t));
    poste!("UserDimensionWeights", |_t: &RawTweet| {
        UserDimensionWeights::for_profile(&profile).apply(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    });
    poste!("age_hours (Utc::now)", |t: &RawTweet| age_hours(t));

    println!();
    let total_postes: f64 = postes.iter().map(|(_, ms)| ms).sum();

    // Le tout, pour comparer a la somme des parties. Deux variantes :
    // autonome (chaque candidat relit l'horloge et refait la detection de
    // contenu poubelle) et telle que `score_all` l'appelle, avec le contexte
    // du lot.
    let mut feed: Vec<ScoredTweet> = Vec::with_capacity(CANDIDATES);
    let passe = |avec_contexte: bool, feed: &mut Vec<ScoredTweet>| -> f64 {
        let mut temps = Vec::with_capacity(ROUNDS + 1);
        for _ in 0..=ROUNDS {
            feed.clear();
            let mut shape = FeedShape::empty();
            let start = Instant::now();
            for t in &tweets {
                let s = if avec_contexte {
                    let ctx = ScoringContext::at(now).with_garbage(detector.detect(t));
                    score_tweet_with_weights_at(t, &profile, 0, shape, &weights, ctx)
                } else {
                    score_tweet_with_weights(t, &profile, 0, shape, &weights)
                };
                shape.push(t.has_media);
                feed.push(s);
            }
            temps.push(start.elapsed().as_nanos());
        }
        temps.sort_unstable();
        temps[0] as f64 / 1_000_000.0
    };

    let autonome = passe(false, &mut feed);
    let checksum_autonome: f64 = feed.iter().map(|s| s.score).sum();
    let complet = passe(true, &mut feed);
    let checksum: f64 = feed.iter().map(|s| s.score).sum();
    assert!(
        (checksum - checksum_autonome).abs() < 1e-9,
        "le contexte de lot ne doit rien changer au classement"
    );

    println!(
        "  {:<34} {autonome:7.3} ms   (checksum {checksum_autonome:.6})",
        "score_tweet_with_weights"
    );
    println!(
        "  {:<34} {complet:7.3} ms   (checksum {checksum:.6})",
        "  ... avec le contexte du lot"
    );
    println!(
        "  {:<34} {total_postes:7.3} ms",
        "somme des postes ci-dessus"
    );
    println!();

    postes.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("Postes les plus chers :");
    for (nom, ms) in postes.iter().take(7) {
        println!(
            "  {nom:<34} {ms:7.3} ms   {:5.1} % du chemin complet",
            ms / complet * 100.0
        );
    }
}
