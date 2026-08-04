use deadpool_postgres::Pool as PgPool;
use std::sync::Arc;
use std::time::SystemTime;

use crate::ml::auto_tuner::AutoTuner;
use crate::services::cache_manager::CacheManager;
use crate::services::recommender::RecommenderService;

#[derive(Clone)]
pub struct AppState {
    pub pg: PgPool,
    pub cache: CacheManager,
    pub recommender: Arc<RecommenderService>,
    pub auto_tuner: Arc<AutoTuner>,
    pub admin_secret: String,
    pub internal_secret: String,
    pub start_time: SystemTime,
}
