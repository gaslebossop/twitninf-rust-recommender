/// Balayage d'attribution CTR.
///
/// Une impression servie ouvre une fenêtre : si le lecteur like, commente ou
/// retweete pendant cette fenêtre, le handler de tracking l'entraîne en positif
/// et retire l'impression de la file. Ce qui reste après expiration, c'est
/// exactement l'inverse : un tweet montré que le lecteur a ignoré. C'est la
/// source des exemples négatifs, sans laquelle le modèle n'a que des positifs
/// et ne peut rien discriminer.
///
/// Ce balayage est aussi le seul endroit qui persiste le modèle sur disque :
/// sans lui, tout l'apprentissage repartait de zéro à chaque redémarrage.
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, info};

use crate::services::cache_manager::CacheManager;
use crate::services::recommender::RecommenderService;

/// Fréquence de balayage. Nettement plus court que la fenêtre d'attribution
/// pour que les négatifs arrivent au fil de l'eau, pas par à-coups.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Plafond par passage : borne le travail d'un tick même si la file a gonflé.
const MAX_PER_SWEEP: usize = 500;

/// Nombre de nouveaux samples avant réécriture du modèle sur disque.
const SAVE_EVERY_SAMPLES: u64 = 50;

pub fn spawn(recommender: Arc<RecommenderService>, cache: CacheManager) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
        // Le premier tick d'un interval() se déclenche immédiatement : on le
        // consomme pour ne pas balayer avant que le service ait servi quoi que
        // ce soit.
        ticker.tick().await;

        let mut last_saved_at = recommender.ctr_stats().0;

        loop {
            ticker.tick().await;

            let expired = cache.drain_expired_impressions(MAX_PER_SWEEP).await;
            let n = expired.len();
            for (_user_id, _tweet_id, features) in expired {
                recommender.record_ctr_event(&features, false);
            }

            let (samples, global_ctr) = recommender.ctr_stats();
            if n > 0 {
                let pending = cache.pending_impressions().await;
                debug!(negatives = n, samples, global_ctr, pending,
                       "Balayage CTR : impressions sans engagement");
            }

            if samples >= last_saved_at + SAVE_EVERY_SAMPLES {
                recommender.persist_ctr_model().await;
                last_saved_at = samples;
                info!(samples, global_ctr, "Modèle CTR persisté");
            }
        }
    });
}
