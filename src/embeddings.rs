//! Compréhension sémantique du contenu — via un modèle local, pas une API payante.
//!
//! ── Le trou que ce module comble ────────────────────────────────────────────
//! La personnalisation existante (D2, D3, D8) matche des mots-clés exacts
//! (`content.contains(word)`) et des identifiants d'auteur. Deux tweets qui
//! parlent de la même chose avec des mots différents — « le café a fermé » et
//! « plus de café dans le quartier » — ne se ressemblent jamais pour le
//! moteur. Ce module donne une vraie notion de similarité de CONTENU.
//!
//! ── Pourquoi local, et pas une API d'embedding ──────────────────────────────
//! `all-MiniLM-L6-v2` (22M paramètres, 384 dimensions) tourne en ONNX via
//! `fastembed`, entièrement sur ce serveur : aucune clé API, aucune facturation
//! par appel, aucune dépendance réseau à un tiers. Le modèle (~90 Mo) est
//! téléchargé une fois au premier démarrage et mis en cache sur disque.
//! Le CPU de ce VPS encode plusieurs milliers de phrases par seconde — largement
//! suffisant pour un tweet à la fois, à cette échelle de trafic.
//!
//! ── Ce qui est stocké, et comment ───────────────────────────────────────────
//! Un embedding par tweet, dans une colonne `vector(384)` (extension `pgvector`
//! sur le Postgres déjà en place — pas de base vectorielle séparée à opérer).
//! Un index HNSW rend la recherche des plus proches voisins rapide même sur
//! des centaines de milliers de lignes. Le « vecteur de goût » d'un lecteur est
//! la moyenne des embeddings de ses tweets likés récents — voir
//! `RecommenderService::build_user_profile`, champ `taste_vector`.

use std::sync::Arc;

use anyhow::{Context, Result};
use deadpool_postgres::Pool as PgPool;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use pgvector::Vector;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

/// Dimension du modèle `AllMiniLML6V2`. Toucher au modèle veut dire migrer la
/// colonne — les deux doivent rester synchronisés.
pub const EMBEDDING_DIM: usize = 384;

/// Un tweet plus vieux que ça n'entre plus dans le vecteur de goût : les
/// intérêts d'un lecteur dérivent, un like d'il y a un an ne devrait pas peser
/// autant qu'un like d'hier.
const TASTE_LOOKBACK_DAYS: i64 = 90;
/// Nombre de tweets likés récents moyennés pour former le vecteur de goût.
const TASTE_SAMPLE_LIMIT: i64 = 40;

#[derive(Clone)]
pub struct EmbeddingService {
    model: Arc<AsyncMutex<TextEmbedding>>,
}

impl EmbeddingService {
    /// Charge le modèle. Bloquant au premier appel si le modèle n'est pas
    /// encore en cache disque (télécharge ~90 Mo) — c'est pourquoi
    /// l'initialisation passe par `spawn_blocking` au démarrage plutôt que de
    /// bloquer le thread principal de `main()`.
    pub fn load() -> Result<Self> {
        info!("Chargement du modèle d'embedding (all-MiniLM-L6-v2)…");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )
        .context("chargement du modèle fastembed")?;
        info!("Modèle d'embedding prêt");
        Ok(Self {
            model: Arc::new(AsyncMutex::new(model)),
        })
    }

    /// Calcule l'embedding d'un texte.
    ///
    /// L'inférence ONNX est synchrone et consomme du CPU : `spawn_blocking`
    /// l'exécute sur le pool de threads bloquants de Tokio, pour ne pas geler
    /// les autres requêtes en cours pendant le calcul.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let text = text.to_string();
        let mut vecs = tokio::task::spawn_blocking(move || {
            let model = model.blocking_lock();
            model.embed(vec![text], None)
        })
        .await
        .context("tâche d'embedding")??;
        vecs.pop().context("le modèle n'a renvoyé aucun vecteur")
    }
}

/// Crée la colonne et l'index manquants. Idempotent — appelé au démarrage,
/// comme `experiments::ensure_schema`.
pub async fn ensure_schema(pg: &PgPool) -> Result<()> {
    let client = pg.get().await?;
    client
        .batch_execute(&format!(
            r#"
        CREATE EXTENSION IF NOT EXISTS vector;
        ALTER TABLE tweets ADD COLUMN IF NOT EXISTS embedding vector({dim});
        CREATE INDEX IF NOT EXISTS tweets_embedding_hnsw_idx
            ON tweets USING hnsw (embedding vector_cosine_ops);
        "#,
            dim = EMBEDDING_DIM
        ))
        .await?;
    debug!("Schéma embeddings vérifié (colonne + index HNSW)");
    Ok(())
}

/// Calcule et stocke l'embedding d'UN tweet. Appelé par l'API Node juste
/// après la création d'un tweet (fire-and-forget), et par le rattrapage au
/// démarrage pour les tweets publiés avant ce module.
pub async fn embed_and_store(
    pg: &PgPool,
    embedder: &EmbeddingService,
    tweet_id: &str,
    content: &str,
) -> Result<()> {
    if content.trim().is_empty() {
        // Un tweet image-seule n'a rien à embedder — laisser la colonne NULL,
        // pas un vecteur nul qui se rapprocherait artificiellement de tout.
        return Ok(());
    }
    let vec = embedder.embed_one(content).await?;
    let uid = uuid::Uuid::parse_str(tweet_id)?;
    let client = pg.get().await?;
    client
        .execute(
            "UPDATE tweets SET embedding = $1 WHERE id = $2",
            &[&Vector::from(vec), &uid],
        )
        .await?;
    Ok(())
}

/// Un tweet dont le temps de lecture cumulé dépasse ce plancher compte comme
/// « fortement consommé », au même titre qu'un like — voir `user_taste_vector`.
/// Pas de raisonnement en taux de complétion ici (ce module n'a pas accès à
/// `algorithm::dwell`, et dupliquer sa logique en SQL brut coûterait plus que
/// ça ne vaudrait) : un plancher absolu généreux suffit à écarter les
/// simples passages sans prétendre mesurer finement l'intérêt.
const TASTE_HEAVY_DWELL_FLOOR_MS: i64 = 15_000;

/// Vecteur de goût d'un lecteur : moyenne des embeddings des tweets LIKÉS
/// récents ∪ FORTEMENT CONSOMMÉS récents (dwell cumulé ≥
/// `TASTE_HEAVY_DWELL_FLOOR_MS`, voir `user_behavior_data` /
/// `action_type='time_spent'`, écrit par `mirrorDwell` côté API).
///
/// Avant cette union, un lecteur qui lit beaucoup et like peu — le profil
/// majoritaire — restait invisible pour ce vecteur : son goût réel n'avait
/// aucune trace ici, quel que soit son temps de lecture. `LEAST(...,600000)`
/// même plafond qu'ailleurs (`SQL_AFFINITY`, `dwellMirror.js::DWELL_CAP_MS`) :
/// un événement peut remonter jusqu'à ~10h côté client (app restée ouverte en
/// arrière-plan).
///
/// `None` si le lecteur n'a encore ni like ni lecture significative sur un
/// tweet embedded (compte neuf, ou rattrapage pas encore passé).
pub async fn user_taste_vector(pg: &PgPool, user_id: &str) -> Result<Option<Vec<f32>>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let rows = client
        .query(
            &format!(
                "SELECT embedding FROM ( \
                     SELECT DISTINCT ON (tweet_id) tweet_id, embedding, recency FROM ( \
                         SELECT t.id AS tweet_id, t.embedding AS embedding, tl.created_at AS recency \
                         FROM tweet_likes tl JOIN tweets t ON t.id = tl.tweet_id \
                         WHERE tl.user_id = $1 AND t.embedding IS NOT NULL \
                           AND tl.created_at > NOW() - INTERVAL '{TASTE_LOOKBACK_DAYS} days' \
                         UNION ALL \
                         SELECT t.id AS tweet_id, t.embedding AS embedding, heavy.last_seen AS recency \
                         FROM ( \
                             SELECT b.target_id, \
                                    MAX(b.timestamp) AS last_seen \
                             FROM user_behavior_data b \
                             WHERE b.user_id = $1 AND b.action_type = 'time_spent' AND b.target_type = 'tweet' \
                               AND COALESCE(b.is_data_test, false) = false \
                               AND b.timestamp > NOW() - INTERVAL '{TASTE_LOOKBACK_DAYS} days' \
                             GROUP BY b.target_id \
                             HAVING SUM(LEAST(COALESCE((b.context_data->>'time_spent_ms')::bigint, 0), 600000)) \
                                    >= {TASTE_HEAVY_DWELL_FLOOR_MS} \
                         ) heavy \
                         JOIN tweets t ON t.id::text = heavy.target_id \
                         WHERE t.embedding IS NOT NULL AND t.user_id <> $1 \
                     ) both_sources \
                     ORDER BY tweet_id, recency DESC \
                 ) deduped \
                 ORDER BY recency DESC LIMIT {TASTE_SAMPLE_LIMIT}"
            ),
            &[&uid],
        )
        .await?;
    Ok(average_vectors(&rows_to_vectors(&rows)))
}

fn rows_to_vectors(rows: &[tokio_postgres::Row]) -> Vec<Vec<f32>> {
    rows.iter()
        .filter_map(|row| row.try_get::<_, Vector>(0).ok())
        .map(|v| v.as_slice().to_vec())
        .collect()
}

/// Moyenne de plusieurs embeddings — le même calcul sert au vecteur de goût
/// naturel ci-dessus et à la recalibration explicite (voir `calibration`).
/// Un vecteur de mauvaise dimension est ignoré plutôt que de fausser la
/// moyenne ou de faire échouer tout l'appel.
pub fn average_vectors(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let mut sum = vec![0.0f32; EMBEDDING_DIM];
    let mut count = 0usize;
    for v in vectors {
        if v.len() != EMBEDDING_DIM {
            warn!(len = v.len(), "Embedding de dimension inattendue — ignoré");
            continue;
        }
        for (i, x) in v.iter().enumerate() {
            sum[i] += x;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    for x in sum.iter_mut() {
        *x /= count as f32;
    }
    Some(sum)
}

/// Plus proches voisins sémantiques d'un vecteur de goût — nouvelle source de
/// candidats, complémentaire aux 8 sources SQL existantes (aucune ne compare
/// du contenu, seulement récence/popularité/follow).
///
/// Filtres alignés sur `CANDIDATES_CTE` : mêmes règles de visibilité
/// (non supprimé, approuvé) que toutes les autres sources.
pub async fn nearest_tweets(
    pg: &PgPool,
    taste: &[f32],
    exclude_user_id: &str,
    limit: i64,
) -> Result<Vec<String>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(exclude_user_id)?;
    let vector = Vector::from(taste.to_vec());
    let rows = client
        .query(
            r#"
            SELECT id::text FROM tweets t
            WHERE t.embedding IS NOT NULL
              AND t.user_id != $2
              AND t.deleted_at IS NULL
              AND t.moderation_status = 'approved'
              AND t.created_at > NOW() - INTERVAL '14 days'
            ORDER BY t.embedding <=> $1
            LIMIT $3
            "#,
            &[&vector, &uid, &limit],
        )
        .await?;
    Ok(rows
        .iter()
        .filter_map(|r| r.try_get::<_, String>(0).ok())
        .collect())
}

/// Rattrapage des tweets publiés avant ce module, ou dont l'embedding a
/// échoué (panne transitoire du modèle). Tourne en tâche de fond, par petits
/// lots, pour ne jamais entrer en concurrence avec le trafic de recommandation
/// pour le CPU — voir l'appel dans `main.rs`, même patron que `ml::ctr_sweeper`.
pub async fn backfill_sweep(pg: &PgPool, embedder: &EmbeddingService, batch_size: i64) -> usize {
    let client = match pg.get().await {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "Backfill embeddings: connexion Postgres indisponible");
            return 0;
        }
    };
    let rows = match client
        .query(
            "SELECT id::text, content FROM tweets \
             WHERE embedding IS NULL AND deleted_at IS NULL AND content != '' \
             ORDER BY created_at DESC LIMIT $1",
            &[&batch_size],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Backfill embeddings: requête de sélection échouée");
            return 0;
        }
    };
    drop(client);

    let mut done = 0usize;
    for row in &rows {
        let id: String = row.get(0);
        let content: String = row.get(1);
        match embed_and_store(pg, embedder, &id, &content).await {
            Ok(_) => done += 1,
            Err(e) => warn!(tweet_id = %id, error = %e, "Backfill: embedding échoué"),
        }
    }
    if done > 0 {
        debug!(done, "Backfill embeddings: lot traité");
    }
    done
}
