//! Recalibration de l'algorithme — page dédiée, accessible uniquement depuis
//! les Paramètres, jamais proposée automatiquement (demande explicite).
//!
//! 5 tours de 6 tweets. Les deux premiers tours explorent large — thèmes et
//! auteurs distincts, sans tenir compte de ce que l'algorithme croit déjà
//! savoir — pour cartographier ce qui intéresse le compte plutôt que de
//! confirmer un préjugé existant. Les tours suivants resserrent : plus
//! proches voisins sémantiques (même mécanisme que `embeddings::nearest_tweets`)
//! des tweets choisis dans CETTE session, sans contrainte de diversité
//! d'auteur — si le compte montre un intérêt marqué pour un auteur précis, le
//! reflet fidèle de ça, c'est de lui en proposer plus, pas de l'en priver
//! pour la forme.
//!
//! Signal privé : contrairement à un like normal, un choix de recalibration
//! n'écrit jamais dans `tweet_likes` (pas de notification à l'auteur, pas de
//! compteur public qui bouge) — seul l'algorithme en tient compte.
//! `finish()` déclenche ce que des likes ordinaires auraient déclenché côté
//! algo (boost temps réel par auteur, cooccurrence globale — mêmes fonctions
//! qu'un like normal, voir `handlers::tracking::track_handler`), plus un
//! effet qu'un like ordinaire, dilué dans l'activité normale, n'a pas : un
//! vecteur de goût dédié et concentré, mélangé au vecteur naturel (90 jours
//! de likes) au prochain rechargement de profil — voir `blend_taste`.

use std::collections::HashSet;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use pgvector::Vector;
use rand::seq::SliceRandom;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::embeddings::average_vectors;
use crate::services::cache_manager::CacheManager;

pub const ROUNDS: u8 = 5;
pub const TWEETS_PER_ROUND: i64 = 6;

/// Tours où la diversité d'auteur/thème est forcée. Au-delà, on suit le
/// signal sémantique même s'il ramène au même auteur plusieurs fois — voir
/// la doc de module.
const DIVERSITY_ENFORCED_THROUGH_ROUND: u8 = 2;

/// TTL du vecteur de goût de calibration. Rien ne le fait naturellement
/// expirer comme un avertissement (voir `shadowban::strikes`) : un compte qui
/// ne recalibre plus jamais garde son dernier réglage indéfiniment tant qu'il
/// est actif. Le TTL n'est qu'un filet — purger un compte abandonné plutôt
/// que de faire grossir Redis sans fin.
const CALIBRATION_TASTE_TTL_SECS: i64 = 180 * 24 * 3600;

/// Poids du vecteur de calibration face au vecteur naturel (90 jours de
/// likes) quand les deux existent — voir `blend_taste`. Un choix explicite et
/// concentré sur 5 tours pèse plus qu'un like ordinaire noyé dans trois mois
/// d'activité, mais ne l'efface pas entièrement : le compte a aussi un passé.
const CALIBRATION_TASTE_WEIGHT: f32 = 0.65;

#[derive(Debug, Deserialize)]
pub struct CalibrationRoundRequest {
    pub user_id: String,
    pub round: u8,
    /// Cumulés depuis le premier tour de CETTE session, pas seulement le tour
    /// précédent — round_candidates doit connaître tout ce qui a déjà été
    /// montré pour ne jamais répéter un tweet.
    #[serde(default)]
    pub liked_tweet_ids: Vec<String>,
    #[serde(default)]
    pub skipped_tweet_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CalibrationRoundResult {
    pub round: u8,
    pub tweet_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CalibrationFinishRequest {
    pub user_id: String,
    pub liked_tweet_ids: Vec<String>,
}

fn calibration_taste_key(user_id: &str) -> String {
    format!("calibration:taste:{user_id}")
}

impl CacheManager {
    async fn calibration_save_taste(&self, user_id: &str, taste: &[f32]) {
        let Ok(json) = serde_json::to_string(taste) else {
            return;
        };
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c
            .set_ex(
                calibration_taste_key(user_id),
                json,
                CALIBRATION_TASTE_TTL_SECS as u64,
            )
            .await;
    }

    /// Vecteur de goût explicite du compte, s'il a déjà recalibré — voir
    /// `blend_taste` pour comment il se combine au vecteur naturel.
    pub async fn calibration_load_taste(&self, user_id: &str) -> Option<Vec<f32>> {
        let mut c = self.conn.lock().await;
        let raw: Option<String> = c.get(calibration_taste_key(user_id)).await.ok().flatten();
        drop(c);
        raw.and_then(|s| serde_json::from_str(&s).ok())
    }
}

/// Combine le vecteur de calibration explicite et le vecteur naturel (90
/// jours de likes) — voir `CALIBRATION_TASTE_WEIGHT`. Reste correct côté
/// `pgvector` même sans renormaliser : la distance cosinus (`<=>`, utilisée
/// partout où ce vecteur sert) est déjà invariante à l'échelle.
pub fn blend_taste(calibration: &[f32], natural: &[f32]) -> Vec<f32> {
    calibration
        .iter()
        .zip(natural.iter())
        .map(|(c, n)| c * CALIBRATION_TASTE_WEIGHT + n * (1.0 - CALIBRATION_TASTE_WEIGHT))
        .collect()
}

/// Sélectionne les candidats d'UN tour.
pub async fn round_candidates(
    pg: &PgPool,
    user_id: &str,
    round: u8,
    liked_so_far: &[String],
    skipped_so_far: &[String],
) -> Result<Vec<String>> {
    let mut excluded: HashSet<String> = liked_so_far.iter().cloned().collect();
    excluded.extend(skipped_so_far.iter().cloned());

    if round > DIVERSITY_ENFORCED_THROUGH_ROUND {
        if let Some(taste) = mean_embedding(pg, liked_so_far).await? {
            let mut ids =
                similarity_candidates(pg, user_id, &taste, &excluded, TWEETS_PER_ROUND * 3).await?;
            ids.shuffle(&mut rand::thread_rng());
            ids.truncate(TWEETS_PER_ROUND as usize);
            if ids.len() as i64 == TWEETS_PER_ROUND {
                return Ok(ids);
            }
            // Pas assez de voisins sémantiques (dataset encore petit) : on
            // complète avec le vivier diversifié plutôt que de renvoyer un
            // tour incomplet.
            excluded.extend(ids.iter().cloned());
            let filler =
                diverse_candidates(pg, user_id, &excluded, TWEETS_PER_ROUND - ids.len() as i64)
                    .await?;
            ids.extend(filler);
            return Ok(ids);
        }
    }
    diverse_candidates(pg, user_id, &excluded, TWEETS_PER_ROUND).await
}

/// Vivier large et diversifié : thèmes et auteurs distincts en priorité,
/// relâché seulement si le vivier ne suffit pas à remplir le tour — un
/// dataset encore petit (voir la volumétrie réelle de la plateforme) ne doit
/// jamais renvoyer un tour vide faute de diversité disponible.
async fn diverse_candidates(
    pg: &PgPool,
    user_id: &str,
    excluded: &HashSet<String>,
    limit: i64,
) -> Result<Vec<String>> {
    if limit <= 0 {
        return Ok(Vec::new());
    }
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let pool_limit = (limit * 10).max(60);
    let rows = client
        .query(
            r#"
            SELECT t.id::text, t.user_id::text, COALESCE(ll.theme, 'autre') AS theme
            FROM tweets t
            JOIN users u ON u.id = t.user_id
            LEFT JOIN tweet_llm_labels ll ON ll.tweet_id = t.id
            WHERE t.deleted_at IS NULL
              AND t.moderation_status = 'approved'
              AND t.is_private = false
              AND COALESCE(t.is_data_test, false) = false
              AND u.is_active = true
              AND COALESCE(u.is_suspended, false) = false
              AND t.user_id != $1
              AND t.parent_tweet_id IS NULL
              AND COALESCE(t.is_retweet, false) = false
              AND COALESCE(t.content, '') != ''
              AND COALESCE(ll.theme, 'autre') != 'spam_vide'
            ORDER BY COALESCE(ll.quality_score, 0.5) DESC, t.created_at DESC
            LIMIT $2
            "#,
            &[&uid, &pool_limit],
        )
        .await?;

    let mut author_seen: HashSet<String> = HashSet::new();
    let mut theme_seen: HashSet<String> = HashSet::new();
    let mut picked: Vec<String> = Vec::new();
    let mut leftovers: Vec<String> = Vec::new();

    for row in &rows {
        if picked.len() as i64 >= limit {
            break;
        }
        let id: String = row.get(0);
        if excluded.contains(&id) {
            continue;
        }
        let author: String = row.get(1);
        let theme: String = row.get(2);
        if author_seen.contains(&author) || theme_seen.contains(&theme) {
            leftovers.push(id);
            continue;
        }
        author_seen.insert(author);
        theme_seen.insert(theme);
        picked.push(id);
    }
    for id in leftovers {
        if picked.len() as i64 >= limit {
            break;
        }
        picked.push(id);
    }
    Ok(picked)
}

/// Plus proches voisins sémantiques du goût accumulé CETTE session — même
/// filtre de visibilité que `diverse_candidates`, sans la contrainte de
/// diversité : un tour resserré est le but à partir de
/// `DIVERSITY_ENFORCED_THROUGH_ROUND`, pas un accident à corriger.
async fn similarity_candidates(
    pg: &PgPool,
    user_id: &str,
    taste: &[f32],
    excluded: &HashSet<String>,
    limit: i64,
) -> Result<Vec<String>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let vector = Vector::from(taste.to_vec());
    let pool_limit = (limit * 4).max(24);
    let rows = client
        .query(
            r#"
            SELECT t.id::text FROM tweets t
            JOIN users u ON u.id = t.user_id
            WHERE t.embedding IS NOT NULL
              AND t.deleted_at IS NULL
              AND t.moderation_status = 'approved'
              AND t.is_private = false
              AND COALESCE(t.is_data_test, false) = false
              AND u.is_active = true
              AND COALESCE(u.is_suspended, false) = false
              AND t.user_id != $1
              AND t.parent_tweet_id IS NULL
              AND COALESCE(t.is_retweet, false) = false
            ORDER BY t.embedding <=> $2
            LIMIT $3
            "#,
            &[&uid, &vector, &pool_limit],
        )
        .await?;
    Ok(rows
        .into_iter()
        .filter_map(|r| r.try_get::<_, String>(0).ok())
        .filter(|id| !excluded.contains(id))
        .take(limit as usize)
        .collect())
}

/// Moyenne des embeddings déjà stockés des tweets donnés — pas de calcul
/// d'embedding ici, seulement une lecture : `mean_embedding` ne dépend donc
/// jamais de la disponibilité du modèle ONNX (`AppState::embeddings`),
/// seulement de Postgres.
async fn mean_embedding(pg: &PgPool, tweet_ids: &[String]) -> Result<Option<Vec<f32>>> {
    if tweet_ids.is_empty() {
        return Ok(None);
    }
    let uuids: Vec<uuid::Uuid> = tweet_ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .collect();
    if uuids.is_empty() {
        return Ok(None);
    }
    let client = pg.get().await?;
    let rows = client
        .query(
            "SELECT embedding FROM tweets WHERE id = ANY($1) AND embedding IS NOT NULL",
            &[&uuids],
        )
        .await?;
    let vectors: Vec<Vec<f32>> = rows
        .iter()
        .filter_map(|row| row.try_get::<_, Vector>(0).ok())
        .map(|v| v.as_slice().to_vec())
        .collect();
    Ok(average_vectors(&vectors))
}

/// Traite les résultats d'une session complète — voir la doc de module pour
/// ce que ça déclenche et pourquoi ce n'est PAS un like public.
pub async fn finish(
    pg: &PgPool,
    cache: &CacheManager,
    user_id: &str,
    liked_tweet_ids: &[String],
) -> Result<usize> {
    if liked_tweet_ids.is_empty() {
        return Ok(0);
    }

    if let Some(taste) = mean_embedding(pg, liked_tweet_ids).await? {
        cache.calibration_save_taste(user_id, &taste).await;
    }

    let uuids: Vec<uuid::Uuid> = liked_tweet_ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .collect();
    if uuids.is_empty() {
        return Ok(0);
    }
    let client = pg.get().await?;
    let rows = client
        .query(
            "SELECT user_id::text FROM tweets WHERE id = ANY($1)",
            &[&uuids],
        )
        .await?;
    drop(client);

    for row in &rows {
        let author_id: String = row.get(0);
        cache
            .record_author_feedback(user_id, &author_id, true)
            .await;
        cache.record_like_cooccurrence(user_id, &author_id).await;
    }
    debug!(user_id, picks = rows.len(), "Recalibration terminée");
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EMBEDDING_DIM;

    #[test]
    fn blend_pondere_vers_la_calibration() {
        let calib = vec![1.0f32; EMBEDDING_DIM];
        let natural = vec![0.0f32; EMBEDDING_DIM];
        let blended = blend_taste(&calib, &natural);
        assert!((blended[0] - CALIBRATION_TASTE_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn blend_egal_aux_deux_bouts_quand_un_seul_existe() {
        // Documente juste la formule : à poids 0.65/0.35, un vecteur nul d'un
        // côté ne redonne PAS l'autre vecteur tel quel — c'est round_candidates
        // /le point d'appel dans `recommender.rs` qui gère l'absence de l'un
        // des deux en amont, pas `blend_taste`.
        let calib = vec![2.0f32; EMBEDDING_DIM];
        let natural = vec![2.0f32; EMBEDDING_DIM];
        let blended = blend_taste(&calib, &natural);
        assert!((blended[0] - 2.0).abs() < 1e-6);
    }
}
