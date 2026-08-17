//! Filtrage collaboratif léger — « les fans de ce compte aiment souvent ça
//! aussi », sans matrix factorization, sans modèle à entraîner.
//!
//! ── Ce que ça capte que les embeddings ne captent pas ───────────────────────
//! Un embedding (`crate::embeddings`) rapproche deux tweets qui parlent de la
//! même chose. Il ne rapproche jamais deux comptes sans rapport thématique
//! dont les mêmes personnes se trouvent être fans — un compte d'humour et un
//! compte de musique électronique n'ont RIEN de commun en contenu, et
//! pourtant leurs publics peuvent largement se recouper. C'est exactement le
//! signal que le filtrage collaboratif fournit et qu'aucune comparaison de
//! contenu, aussi bonne soit-elle, ne peut fournir à sa place.
//!
//! ── Comment, sans ML ─────────────────────────────────────────────────────
//! Une matrice de co-occurrence AUTEUR × AUTEUR : à chaque like, on regarde
//! les auteurs que CE lecteur a likés récemment, et on incrémente un compteur
//! pour chaque paire (auteur du nouveau like, auteur déjà liké). Deux comptes
//! likés souvent par les mêmes lecteurs accumulent un score élevé — c'est
//! tout l'algorithme. Deux ZSET Redis par auteur, pas de job d'entraînement,
//! pas de matrice à recalculer en masse.

use std::collections::HashMap;

use redis::AsyncCommands;

use crate::services::cache_manager::CacheManager;

/// Nombre d'auteurs récents conservés par lecteur — au-delà, les plus vieux
/// sortent de la fenêtre glissante.
const RECENT_AUTHORS_LIMIT: isize = 20;
/// Un lecteur inactif depuis 30 jours voit sa fenêtre récente s'effacer :
/// ses goûts d'il y a six mois ne devraient plus former de nouvelles paires.
const RECENT_AUTHORS_TTL_SECS: i64 = 60 * 60 * 24 * 30;
/// Les paires elles-mêmes durent plus longtemps que la fenêtre d'un lecteur :
/// c'est un signal agrégé sur TOUS les lecteurs, pas le goût d'un seul.
const PAIR_TTL_SECS: i64 = 60 * 60 * 24 * 90;

fn recent_key(user_id: &str) -> String {
    format!("cooccur:recent:{user_id}")
}
fn pair_key(author_id: &str) -> String {
    format!("cooccur:pair:{author_id}")
}

impl CacheManager {
    /// Appelé à chaque like avec un `author_id` connu. Incrémente la
    /// co-occurrence entre CET auteur et chaque auteur récemment liké par ce
    /// lecteur, dans les deux sens (la relation est symétrique — être
    /// co-aimé avec A est la même chose pour A que pour B).
    pub async fn record_like_cooccurrence(&self, user_id: &str, author_id: &str) {
        let rkey = recent_key(user_id);
        let recent: Vec<String> = {
            let mut c = self.conn.lock().await;
            c.lrange(&rkey, 0, RECENT_AUTHORS_LIMIT - 1)
                .await
                .unwrap_or_default()
        };

        {
            let mut c = self.conn.lock().await;
            for other in recent.iter().filter(|o| o.as_str() != author_id) {
                let _: Result<f64, _> = c.zincr(pair_key(author_id), other.as_str(), 1.0).await;
                let _: Result<f64, _> = c.zincr(pair_key(other), author_id, 1.0).await;
                let _: Result<(), _> = c.expire(pair_key(author_id), PAIR_TTL_SECS).await;
                let _: Result<(), _> = c.expire(pair_key(other), PAIR_TTL_SECS).await;
            }
            // Retiré puis remis en tête : un auteur déjà présent remonte au
            // lieu de créer un doublon dans la fenêtre glissante.
            let _: Result<i64, _> = c.lrem(&rkey, 0, author_id).await;
            let _: Result<i64, _> = c.lpush(&rkey, author_id).await;
            let _: Result<(), _> = c.ltrim(&rkey, 0, RECENT_AUTHORS_LIMIT - 1).await;
            let _: Result<(), _> = c.expire(&rkey, RECENT_AUTHORS_TTL_SECS).await;
        }
    }

    /// Auteurs co-aimés avec les auteurs favoris de ce lecteur (`profile.top_authors`).
    /// Additionne les scores de co-occurrence sur les quelques auteurs les
    /// plus affins plutôt qu'un seul, pour ne pas dépendre d'une seule
    /// relation qui pourrait être un signal faible ou accidentel.
    pub async fn co_liked_authors(&self, seed_authors: &[String], limit: usize) -> Vec<String> {
        if seed_authors.is_empty() {
            return Vec::new();
        }
        let mut scored: HashMap<String, f64> = HashMap::new();
        {
            let mut c = self.conn.lock().await;
            for seed in seed_authors.iter().take(10) {
                let pairs: Vec<(String, f64)> = c
                    .zrevrange_withscores(pair_key(seed), 0, 9)
                    .await
                    .unwrap_or_default();
                for (author, score) in pairs {
                    *scored.entry(author).or_insert(0.0) += score;
                }
            }
        }
        for seed in seed_authors {
            scored.remove(seed);
        }
        let mut ranked: Vec<(String, f64)> = scored.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(limit).map(|(a, _)| a).collect()
    }
}
