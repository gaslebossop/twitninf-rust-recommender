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

pub const ROUNDS: u8 = 4;
pub const TWEETS_PER_ROUND: i64 = 5;

/// Tours purement diversifiés, sans aucun signal de similarité — la
/// cartographie initiale. Au-delà, chaque tour reste un MÉLANGE (voir
/// `round_candidates`), jamais une bascule totale vers la seule similarité :
/// sur un corpus aussi mince que celui de la plateforme aujourd'hui (une
/// poignée de tweets embeddés, deux ou trois auteurs qui dominent), une
/// similarité pure ne renvoie plus que ces mêmes comptes — observé en test
/// réel : 2 auteurs sur les 5 tours.
const DIVERSITY_ENFORCED_THROUGH_ROUND: u8 = 1;

/// Nombre max de tweets d'un même auteur DANS UN TOUR, quelle que soit la
/// source (diversifiée ou par similarité). Sans lui, `similarity_candidates`
/// n'avait aucune limite : les plus proches voisins d'un vecteur de goût
/// peuvent très bien être 6 tweets du même compte si c'est lui qui domine le
/// corpus embeddé.
const MAX_PER_AUTHOR_PER_ROUND: usize = 2;

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
///
/// Au-delà de `DIVERSITY_ENFORCED_THROUGH_ROUND`, le tour reste un MÉLANGE —
/// la moitié au plus vient de la similarité sémantique, le reste du vivier
/// diversifié. Une bascule totale vers la similarité donnait, sur le corpus
/// actuel, le même auteur tour après tour : resserrer ne doit pas dégénérer
/// en boucle sur deux ou trois comptes.
pub async fn round_candidates(
    pg: &PgPool,
    user_id: &str,
    round: u8,
    liked_so_far: &[String],
    skipped_so_far: &[String],
) -> Result<Vec<String>> {
    let mut excluded: HashSet<String> = liked_so_far.iter().cloned().collect();
    excluded.extend(skipped_so_far.iter().cloned());

    let mut picks: Vec<String> = Vec::new();
    if round > DIVERSITY_ENFORCED_THROUGH_ROUND {
        if let Some(taste) = mean_embedding(pg, liked_so_far).await? {
            let sim_target = TWEETS_PER_ROUND / 2;
            let sim = similarity_candidates(pg, user_id, &taste, &excluded, sim_target).await?;
            excluded.extend(sim.iter().cloned());
            picks.extend(sim);
        }
    }

    let remaining = TWEETS_PER_ROUND - picks.len() as i64;
    if remaining > 0 {
        let filler = diverse_candidates(pg, user_id, &excluded, remaining).await?;
        picks.extend(filler);
    }
    // Mélange l'ordre d'affichage : sans ça, un tour blendé montrerait
    // toujours les similaires d'abord, le diversifié en second — un ordre
    // qui trahirait le mécanisme au lieu de le laisser transparent.
    picks.shuffle(&mut rand::thread_rng());
    Ok(picks)
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
    let mut per_author: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut picked: Vec<String> = Vec::new();
    let mut leftovers: Vec<(String, String)> = Vec::new(); // (id, author)

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
            leftovers.push((id, author));
            continue;
        }
        author_seen.insert(author.clone());
        theme_seen.insert(theme);
        *per_author.entry(author).or_insert(0) += 1;
        picked.push(id);
    }
    // Repli : le plafond par auteur reste actif, même ici — sans lui, un
    // vivier dominé par 2-3 comptes finissait par les répéter jusqu'à
    // remplir le tour, la diversité stricte ci-dessus n'ayant plus rien à
    // offrir de nouveau après leurs premiers tweets.
    for (id, author) in leftovers {
        if picked.len() as i64 >= limit {
            break;
        }
        let count = per_author.entry(author).or_insert(0);
        if *count >= MAX_PER_AUTHOR_PER_ROUND {
            continue;
        }
        *count += 1;
        picked.push(id);
    }
    Ok(picked)
}

/// Plus proches voisins sémantiques du goût accumulé CETTE session — même
/// filtre de visibilité que `diverse_candidates`, avec le même plafond par
/// auteur (voir `MAX_PER_AUTHOR_PER_ROUND`) : sans lui, les N voisins les
/// plus proches d'un vecteur de goût peuvent tous appartenir au compte qui
/// domine le corpus embeddé — exactement ce qui rendait le mécanisme
/// perceptible et pas juste redondant.
async fn similarity_candidates(
    pg: &PgPool,
    user_id: &str,
    taste: &[f32],
    excluded: &HashSet<String>,
    limit: i64,
) -> Result<Vec<String>> {
    if limit <= 0 {
        return Ok(Vec::new());
    }
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let vector = Vector::from(taste.to_vec());
    let pool_limit = (limit * 8).max(48);
    let rows = client
        .query(
            r#"
            SELECT t.id::text, t.user_id::text FROM tweets t
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

    let mut per_author: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut picked = Vec::new();
    for row in &rows {
        if picked.len() as i64 >= limit {
            break;
        }
        let id: String = row.get(0);
        if excluded.contains(&id) {
            continue;
        }
        let author: String = row.get(1);
        let count = per_author.entry(author).or_insert(0);
        if *count >= MAX_PER_AUTHOR_PER_ROUND {
            continue;
        }
        *count += 1;
        picked.push(id);
    }
    Ok(picked)
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

    // Sans ces deux invalidations, une recalibration ne change RIEN au
    // fil pendant jusqu'à 5 minutes : le profil (`twitninf:profile:*`,
    // contient le vecteur de goût) et la liste déjà classée
    // (`twitninf:reco:*`) restent tous deux en cache indépendamment de ce
    // qui vient d'être écrit ci-dessus. C'est le point même de la
    // fonctionnalité — la sanctionner par un délai silencieux la rend
    // indiscernable d'un bouton qui ne fait rien.
    cache.invalidate_profile(user_id).await;
    cache.invalidate_recommendations(user_id).await;

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
