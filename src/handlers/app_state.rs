use deadpool_postgres::Pool as PgPool;
use std::time::SystemTime;

use crate::services::cache_manager::CacheManager;
use crate::services::recommender::RecommenderService;

#[derive(Clone)]
pub struct AppState {
    pub pg: PgPool,
    pub cache: CacheManager,
    pub recommender: std::sync::Arc<RecommenderService>,
    pub start_time: SystemTime,
}
