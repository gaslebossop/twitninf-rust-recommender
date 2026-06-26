mod ads;       // Targeted advertising
mod algorithm;
mod bandit;    // Phase 3: Contextual Bandit
mod config;
mod constants;
mod error;
mod handlers;
mod middleware;
mod ml;        // Phase 2: ML CTR Predictor
mod models;
mod services;
mod utils;

use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use deadpool_postgres::Runtime;
use tokio_postgres::NoTls;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use handlers::{
    AppState,
    health::health_handler,
    invalidate::invalidate_handler,
    recommendations::recommend_handler,
    tracking::track_handler,
};
use services::{cache_manager::CacheManager, recommender::RecommenderService};

#[tokio::main]
async fn main() -> Result<()> {
    // Config
    let cfg = config::Config::from_env()?;

    // Logging structuré
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
        )
        .json()
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        port = cfg.port,
        "Starting TwitNinf Rust Recommender"
    );

    // Pool PostgreSQL
    let pg_pool = cfg
        .pg_config()
        .create_pool(Some(Runtime::Tokio1), NoTls)?;

    // Vérifier la connexion PG au démarrage
    let _ = pg_pool.get().await.expect("Cannot connect to PostgreSQL");
    info!("PostgreSQL connected (pool size: {})", cfg.db_pool_size);

    // Cache Redis
    let cache = CacheManager::new(&cfg.redis_url).await?;
    info!("Redis connected");

    // Construire le state partagé
    let recommender = Arc::new(RecommenderService::new(pg_pool.clone(), cache.clone()));

    let state = AppState {
        pg: pg_pool,
        cache,
        recommender,
        start_time: SystemTime::now(),
    };

    // Routes
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/recommendations", post(recommend_handler))
        .route("/track", post(track_handler))
        .route("/invalidate", post(invalidate_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
