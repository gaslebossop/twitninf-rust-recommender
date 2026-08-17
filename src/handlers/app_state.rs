use deadpool_postgres::Pool as PgPool;
use std::sync::Arc;
use std::time::SystemTime;

use crate::embeddings::EmbeddingService;
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
    /// Vide tant que le modèle n'a pas fini de charger — le téléchargement du
    /// premier démarrage (~90 Mo) prendrait plus longtemps que le contrôle de
    /// santé du déploiement ne patiente. Le serveur écoute donc immédiatement ;
    /// ce champ se remplit en tâche de fond (voir `main.rs`) et les
    /// fonctionnalités liées aux embeddings se désactivent silencieusement
    /// jusque-là — jamais un service entier indisponible pour une brique non
    /// critique en train de démarrer, ou qui aurait échoué à charger.
    pub embeddings: Arc<tokio::sync::OnceCell<EmbeddingService>>,
}
