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

/// Vecteur de goût d'un lecteur : moyenne des embeddings de ses tweets likés
/// récents. `None` si le lecteur n'a encore aucun like sur un tweet embedded
/// (compte neuf, ou rattrapage pas encore passé sur ses tweets aimés).
pub async fn user_taste_vector(pg: &PgPool, user_id: &str) -> Result<Option<Vec<f32>>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let rows = client
        .query(
            &format!(
                "SELECT t.embedding FROM tweet_likes tl \
                 JOIN tweets t ON t.id = tl.tweet_id \
                 WHERE tl.user_id = $1 AND t.embedding IS NOT NULL \
                   AND tl.created_at > NOW() - INTERVAL '{TASTE_LOOKBACK_DAYS} days' \
                 ORDER BY tl.created_at DESC LIMIT {TASTE_SAMPLE_LIMIT}"
            ),
            &[&uid],
        )
        .await?;

    if rows.is_empty() {
        return Ok(None);
    }

    let mut sum = vec![0.0f32; EMBEDDING_DIM];
    let mut count = 0usize;
    for row in &rows {
        let Ok(v) = row.try_get::<_, Vector>(0) else {
            continue;
        };
        let slice = v.as_slice();
        if slice.len() != EMBEDDING_DIM {
            warn!(
                len = slice.len(),
                "Embedding de dimension inattendue — ignoré"
            );
            continue;
        }
        for (i, x) in slice.iter().enumerate() {
            sum[i] += x;
        }
        count += 1;
    }

    if count == 0 {
        return Ok(None);
    }
    for x in sum.iter_mut() {
        *x /= count as f32;
    }
    Ok(Some(sum))
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
