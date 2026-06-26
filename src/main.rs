mod admin;
mod ads;
mod algorithm;
mod bandit;
mod shadowban;
mod config;
mod constants;
mod error;
mod handlers;
mod middleware;
mod ml;
mod models;
mod services;
mod utils;

use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use axum::{
    http::HeaderValue,
    routing::{get, post},
    Router,
};
use deadpool_postgres::Runtime;
use tokio_postgres::NoTls;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use handlers::{
    AppState,
    admin::{
        admin_algo_stats_handler, admin_ban_handler, admin_filters_handler,
        admin_get_weights_handler, admin_reset_weights_handler, admin_set_shadowban_handler,
        admin_set_weights_handler, admin_unban_handler,
    },
    health::health_handler,
    invalidate::invalidate_handler,
    recommendations::recommend_handler,
    tracking::track_handler,
};
use ml::AutoTuner;
use services::{cache_manager::CacheManager, recommender::RecommenderService};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = config::Config::from_env()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
        )
        .json()
        .init();

    info!(version = env!("CARGO_PKG_VERSION"), port = cfg.port, "Starting TwitNinf Rust Recommender");

    let pg_pool = cfg.pg_config().create_pool(Some(Runtime::Tokio1), NoTls)?;
    let _ = pg_pool.get().await.expect("Cannot connect to PostgreSQL");
    info!("PostgreSQL connected (pool size: {})", cfg.db_pool_size);

    let cache = CacheManager::new(&cfg.redis_url).await?;
    info!("Redis connected");

    let auto_tuner  = Arc::new(AutoTuner::new());
    let recommender = Arc::new(RecommenderService::new_with_tuner(
        pg_pool.clone(), cache.clone(), auto_tuner.clone(),
    ));

    let state = AppState {
        pg: pg_pool,
        cache,
        recommender,
        auto_tuner,
        admin_secret:    cfg.admin_secret.clone(),
        internal_secret: cfg.internal_secret.clone(),
        start_time: SystemTime::now(),
    };

    // CORS : restreint à l'origine du backend Node.js uniquement.
    // Ce service est un microservice interne — il ne doit pas être accessible
    // depuis des origines web arbitraires.
    let allowed_origin = cfg.node_api_url
        .parse::<HeaderValue>()
        .unwrap_or_else(|_| HeaderValue::from_static("http://localhost:3001"));

    let cors = CorsLayer::new()
        .allow_origin(allowed_origin)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::HeaderName::from_static("x-admin-key"),
            axum::http::header::HeaderName::from_static("x-service-key"),
        ]);

    let app = Router::new()
        .route("/health",          get(health_handler))
        .route("/recommendations", post(recommend_handler))
        .route("/track",           post(track_handler))
        .route("/invalidate",      post(invalidate_handler))
        // ── Admin node ────────────────────────────────────────────────────────
        .route("/admin/filters",              get(admin_filters_handler))
        .route("/admin/shadowban",            post(admin_set_shadowban_handler))
        .route("/admin/ban",                  post(admin_ban_handler))
        .route("/admin/unban",                post(admin_unban_handler))
        .route("/admin/algo/weights",         get(admin_get_weights_handler))
        .route("/admin/algo/weights",         post(admin_set_weights_handler))
        .route("/admin/algo/weights/reset",   post(admin_reset_weights_handler))
        .route("/admin/algo/stats",           get(admin_algo_stats_handler))
        .layer(cors)
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
