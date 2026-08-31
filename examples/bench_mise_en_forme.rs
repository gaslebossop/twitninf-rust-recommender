//! Banc d'essai de la MISE EN FORME du fil : `shape_feed` puis
//! `spread_by_author`.
//!
//! ── Pourquoi celui-ci ───────────────────────────────────────────────────────
//! Le scoring est le poste qu'on regarde par réflexe. Ces deux fonctions-là
//! passent inaperçues parce qu'elles ne « calculent » rien — elles réordonnent.
//! Mais `spread_by_author` cherche, pour CHAQUE place du fil, le premier bloc
//! dont l'auteur a encore du quota, en balayant tout ce qui reste à placer.
//! Quand la fenêtre est saturée — ce qui est l'état NORMAL sur une petite
//! communauté, où le vivier ne compte qu'une poignée d'auteurs distincts — ce
//! balayage échoue jusqu'au bout, puis un second le reprend pour choisir le
//! moins mauvais. Deux parcours complets par place, sur un fil qui en compte
//! des centaines.
//!
//! ── Les deux régimes ────────────────────────────────────────────────────────
//! On mesure les deux, parce qu'ils n'ont rien à voir :
//!   * vivier VARIÉ : la fenêtre ne sature jamais, le premier bloc passe
//!     toujours, le coût est linéaire ;
//!   * vivier CONCENTRÉ : la fenêtre sature en permanence — et c'est le cas
//!     relevé en production.
//!
//!   cargo run --release --example bench_mise_en_forme

use std::collections::HashMap;
use std::time::Instant;

use twitninf_recommender::models::{RawTweet, ScoreBreakdown, ScoredTweet};
use twitninf_recommender::services::recommender::{shape_feed, spread_by_author};

const ROUNDS: usize = 7;

fn author_id(i: usize) -> String {
    format!("00000000-0000-0000-0000-{:012}", i)
}

/// `taille` tweets répartis sur `auteurs` comptes distincts.
fn vivier(taille: usize, auteurs: usize) -> Vec<RawTweet> {
    (0..taille)
        .map(|i| RawTweet {
            id: format!("tweet-{i}"),
            user_id: author_id(i % auteurs),
            ..Default::default()
        })
        .collect()
}

fn scores(tweets: &[RawTweet]) -> Vec<ScoredTweet> {
    tweets
        .iter()
        .enumerate()
        .map(|(i, t)| ScoredTweet {
            tweet_id: t.id.clone(),
            score: 1.0 - i as f64 / tweets.len() as f64,
            breakdown: ScoreBreakdown::default(),
            ctr_features: None,
        })
        .collect()
}

fn mesure(tweets: &[RawTweet], scored: &[ScoredTweet]) -> (f64, f64, usize) {
    let carte: HashMap<&str, &RawTweet> = tweets.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut t_forme = Vec::with_capacity(ROUNDS);
    let mut t_etale = Vec::with_capacity(ROUNDS);
    let mut sortie = 0usize;

    for _ in 0..ROUNDS {
        let debut = Instant::now();
        let ids = shape_feed(scored, &carte);
        t_forme.push(debut.elapsed().as_nanos());

        let debut = Instant::now();
        let etale = spread_by_author(ids, &carte);
        t_etale.push(debut.elapsed().as_nanos());
        sortie = etale.len();
    }

    t_forme.sort_unstable();
    t_etale.sort_unstable();
    (
        t_forme[0] as f64 / 1_000_000.0,
        t_etale[0] as f64 / 1_000_000.0,
        sortie,
    )
}

fn main() {
    println!("Mise en forme du fil — minimum de {ROUNDS} passages\n");
    println!(
        "  {:<10} {:>8}  {:>12}  {:>16}",
        "candidats", "auteurs", "shape_feed", "spread_by_author"
    );

    for (taille, auteurs) in [
        (200, 40),
        (500, 40),
        (1_000, 40),
        (1_700, 40),
        // Le cas de production : le vivier ne contient qu'une poignée
        // d'auteurs, donc la fenêtre est saturée en permanence.
        (200, 10),
        (500, 10),
        (1_000, 10),
        (1_700, 10),
    ] {
        let tweets = vivier(taille, auteurs);
        let scored = scores(&tweets);
        let (forme, etale, sortie) = mesure(&tweets, &scored);
        assert_eq!(sortie, taille, "l'etalement reordonne, il ne filtre pas");
        println!("  {taille:<10} {auteurs:>8}  {forme:>9.3} ms  {etale:>13.3} ms");
    }
}
