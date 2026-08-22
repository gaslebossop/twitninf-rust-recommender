/// Balayage d'attribution CTR.
///
/// Une impression servie ouvre une fenêtre : si le lecteur like, commente ou
/// retweete pendant cette fenêtre, le handler de tracking l'entraîne en positif
/// et retire l'impression de la file. Ce qui reste après expiration, c'est
/// exactement l'inverse : un tweet montré que le lecteur a ignoré. C'est la
/// source des exemples négatifs, sans laquelle le modèle n'a que des positifs
/// et ne peut rien discriminer.
///
/// Ce balayage est aussi le seul endroit qui persiste le modèle CTR sur
/// disque : sans lui, tout l'apprentissage repartait de zéro à chaque
/// redémarrage. Le modèle de dwell (`ml::dwell_predictor`) n'a pas besoin de
/// négatifs (le handler de tracking l'entraîne directement sur le poids déjà
/// normalisé — voir `handlers::tracking`), mais il n'a pas d'autre boucle de
/// fond : ce tick lui sert aussi d'horloge de sauvegarde, avec son propre
/// compteur d'échantillons.
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

/// Passages de balayage entre deux reconstructions de l'espace collaboratif.
///
/// Quinze minutes. La co-occurrence bouge à l'échelle des j'aime, pas des
/// secondes, et la factorisation lit toute la matrice : la refaire plus souvent
/// coûterait sans rien apprendre de neuf. Le repère doit d'ailleurs être STABLE
/// d'un rafraîchissement au suivant, sinon le poids appris sur le trait
/// d'affinité poursuit une cible mouvante — voir `crate::collab`.
const COLLAB_EVERY_SWEEPS: u32 = 15;

pub fn spawn(recommender: Arc<RecommenderService>, cache: CacheManager) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
        // Le premier tick d'un interval() se déclenche immédiatement : on le
        // consomme pour ne pas balayer avant que le service ait servi quoi que
        // ce soit.
        ticker.tick().await;

        let mut last_saved_at = recommender.ctr_stats().0;
        let mut last_dwell_saved_at = recommender.dwell_stats().0;
        let mut last_objectives_saved_at = recommender.objective_samples();
        // Reconstruit une première fois dès le démarrage : sans ça, le trait
        // d'affinité resterait neutre pendant le premier quart d'heure de vie
        // du service, à chaque déploiement.
        recommender.refresh_collab_space().await;
        let mut sweeps: u32 = 0;

        loop {
            ticker.tick().await;
            sweeps = sweeps.wrapping_add(1);
            if sweeps % COLLAB_EVERY_SWEEPS == 0 {
                recommender.refresh_collab_space().await;
            }

            let expired = cache.drain_expired_impressions(MAX_PER_SWEEP).await;
            let n = expired.len();
            for (_user_id, _tweet_id, features) in expired {
                recommender.record_ctr_event(&features, false);
                // Un tweet montré que personne n'a relayé n'a pas été relayé,
                // et qu'aucun lecteur n'a signalé n'a pas été signalé : le
                // balayage est la source principale de négatifs des DEUX
                // têtes multi-objectifs, exactement comme pour le CTR.
                recommender.record_objective_ignored(&features);
            }

            let (samples, global_ctr) = recommender.ctr_stats();
            if n > 0 {
                let pending = cache.pending_impressions().await;
                debug!(
                    negatives = n,
                    samples, global_ctr, pending, "Balayage CTR : impressions sans engagement"
                );
            }

            if samples >= last_saved_at + SAVE_EVERY_SAMPLES {
                recommender.persist_ctr_model().await;
                last_saved_at = samples;
                info!(samples, global_ctr, "Modèle CTR persisté");
            }

            let objective_samples = recommender.objective_samples();
            if objective_samples >= last_objectives_saved_at + SAVE_EVERY_SAMPLES {
                recommender.persist_objective_models().await;
                last_objectives_saved_at = objective_samples;
                let ((amplify_n, amplify_rate), (reject_n, reject_rate)) =
                    recommender.objective_stats();
                info!(
                    amplify_n,
                    amplify_rate, reject_n, reject_rate, "Têtes multi-objectifs persistées"
                );
            }

            let (dwell_samples, mean_weight) = recommender.dwell_stats();
            if dwell_samples >= last_dwell_saved_at + SAVE_EVERY_SAMPLES {
                recommender.persist_dwell_model().await;
                last_dwell_saved_at = dwell_samples;
                info!(
                    samples = dwell_samples,
                    mean_weight, "Modèle de dwell persisté"
                );
            }
        }
    });
}
