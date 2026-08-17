mod admin;
mod ads;
mod algorithm;
mod bandit;
mod shadowban;
mod config;
mod constants;
mod error;
mod experiments;
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
    catch_panic::CatchPanicLayer,
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use handlers::{
    AppState,
    account_status::account_status_handler,
    admin::{
        admin_algo_stats_handler, admin_ban_handler, admin_filters_handler,
        admin_get_weights_handler, admin_issue_strike_handler, admin_reset_weights_handler,
        admin_revoke_strike_handler, admin_set_shadowban_handler,
        admin_set_weights_handler, admin_unban_handler, admin_ui_handler,
        admin_logs_handler, admin_data_handler,
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
    experiments::ensure_schema(&pg_pool).await?;
    info!("A/B experiment schema ready");

    let cache = CacheManager::new(&cfg.redis_url).await?;
    info!("Redis connected");

    let auto_tuner  = Arc::new(AutoTuner::new());
    let recommender = Arc::new(RecommenderService::new_with_tuner_and_ml(
        pg_pool.clone(), cache.clone(), auto_tuner.clone(),
    ).await);

    // Boucle d'attribution CTR : convertit les impressions ignorées en exemples
    // négatifs et persiste le modèle. C'est elle qui rend l'apprentissage réel.
    ml::ctr_sweeper::spawn(recommender.clone(), cache.clone());

    let state = AppState {
        pg: pg_pool,
        cache,
        recommender,
        auto_tuner,
        admin_secret:    cfg.admin_secret.clone(),
        internal_secret: cfg.internal_secret.clone(),
        start_time: SystemTime::now(),
    };

    // CORS : autorise le backend Node.js et l'IP publique (pour le panel admin)
    let node_origin = cfg.node_api_url
        .parse::<HeaderValue>()
        .unwrap_or_else(|_| HeaderValue::from_static("http://localhost:3001"));

    let public_origin = std::env::var("PUBLIC_ORIGIN")
        .unwrap_or_else(|_| "http://51.210.11.74".to_string())
        .parse::<HeaderValue>()
        .unwrap_or_else(|_| HeaderValue::from_static("http://51.210.11.74"));

    let cors = CorsLayer::new()
        .allow_origin([node_origin, public_origin])
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
        // État de compte lisible par le créateur concerné (service-à-service).
        .route("/account-status",  get(account_status_handler))
        // ── Admin node ────────────────────────────────────────────────────────
        .route("/admin/panel",                get(admin_ui_handler))
        .route("/admin/filters",              get(admin_filters_handler))
        .route("/admin/shadowban",            post(admin_set_shadowban_handler))
        .route("/admin/strike",               post(admin_issue_strike_handler))
        .route("/admin/strike/revoke",        post(admin_revoke_strike_handler))
        .route("/admin/ban",                  post(admin_ban_handler))
        .route("/admin/unban",                post(admin_unban_handler))
        .route("/admin/algo/weights",         get(admin_get_weights_handler))
        .route("/admin/algo/weights",         post(admin_set_weights_handler))
        .route("/admin/algo/weights/reset",   post(admin_reset_weights_handler))
        .route("/admin/algo/stats",           get(admin_algo_stats_handler))
        .route("/admin/logs",                 get(admin_logs_handler))
        .route("/admin/data",                 get(admin_data_handler))
        .layer(cors)
        .layer(CompressionLayer::new())
        // Un panic pendant le scoring d'un seul tweet ne doit pas emporter le
        // processus — et donc le fil de tous les utilisateurs. Il devient un
        // 500 sur la requête fautive.
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("{}:{}", cfg.bind_host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
