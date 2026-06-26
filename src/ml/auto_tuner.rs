/// Auto-tuner — Adapte les poids des 8 dimensions en temps réel.
///
/// Mécanisme :
///   - Le CtrModel apprend en continu (SGD online) depuis les interactions
///   - Ses poids [0..8] représentent l'importance de D1-D8 pour le CTR
///   - Quand ≥ 500 samples : on normalise ces poids → nouveaux poids de scoring
///   - Les poids admin (override manuel) prennent toujours la priorité
///
/// Résultat : l'algo ajuste ses poids automatiquement selon ce qui fait cliquer
/// les utilisateurs, sans intervention humaine.
use std::sync::{Arc, RwLock};

use tracing::{debug, info};

use crate::admin::AlgoWeights;
use crate::ml::CtrPredictor;

const MIN_SAMPLES_FOR_TUNING: u64 = 500;

#[derive(Clone)]
pub struct AutoTuner {
    /// Poids courants (calculés depuis CTR model, ou defaults si pas assez de données)
    current: Arc<RwLock<TunerState>>,
}

#[derive(Debug, Clone)]
struct TunerState {
    weights: AlgoWeights,
    auto_tuned: bool,
    last_tune_at: u64,
}

impl Default for TunerState {
    fn default() -> Self {
        Self {
            weights: AlgoWeights::default(),
            auto_tuned: false,
            last_tune_at: 0,
        }
    }
}

impl AutoTuner {
    pub fn new() -> Self {
        Self { current: Arc::new(RwLock::new(TunerState::default())) }
    }

    /// Retente une mise à jour des poids depuis le CTR model.
    /// À appeler après chaque interaction enregistrée.
    pub fn maybe_update(&self, ctr: &CtrPredictor, admin_override: Option<&AlgoWeights>) {
        // Si override admin actif, on ne touche pas aux poids
        if admin_override.is_some() {
            return;
        }

        let (samples, _) = ctr.stats();

        // Vérifier si on a assez de données ET si on a de nouvelles données depuis le dernier tune
        let last_tune = self.current.read().unwrap().last_tune_at;
        if samples < MIN_SAMPLES_FOR_TUNING || samples < last_tune + 100 {
            return;
        }

        let learned = ctr.extract_dimension_weights();
        if let Some(weights) = learned {
            let mut state = self.current.write().unwrap();
            state.weights = weights;
            state.auto_tuned = true;
            state.last_tune_at = samples;
            drop(state);
            info!(samples, "AutoTuner: dimension weights updated from CTR model");
        }
    }

    /// Poids actifs (admin override > auto-tuned > defaults)
    pub fn active_weights(&self, admin_override: Option<&AlgoWeights>) -> AlgoWeights {
        if let Some(ow) = admin_override {
            return ow.clone();
        }
        self.current.read().unwrap().weights.clone()
    }

    pub fn is_auto_tuned(&self) -> bool {
        self.current.read().unwrap().auto_tuned
    }

    pub fn last_tune_at(&self) -> u64 {
        self.current.read().unwrap().last_tune_at
    }
}

impl Default for AutoTuner {
    fn default() -> Self { Self::new() }
}

// ─── Extension du CtrPredictor pour extraire les poids dimensionnels ─────────

impl CtrPredictor {
    /// Extrait et normalise les poids D1-D8 appris par le modèle CTR.
    /// Retourne None si pas assez de données ou poids incohérents.
    pub fn extract_dimension_weights(&self) -> Option<AlgoWeights> {
        let (samples, _) = self.stats();
        if samples < MIN_SAMPLES_FOR_TUNING {
            return None;
        }

        // Récupérer les 8 premiers poids via snapshot du modèle
        let raw_weights = self.dimension_weights_snapshot();

        // Forcer les poids positifs (contribution au CTR)
        let clipped: Vec<f64> = raw_weights.iter().map(|&w| w.max(0.001)).collect();
        let sum: f64 = clipped.iter().sum();
        if sum <= 0.0 {
            return None;
        }

        // Normaliser pour que la somme = 1.0
        let norm: Vec<f64> = clipped.iter().map(|w| w / sum).collect();

        debug!(
            d1 = norm[0], d2 = norm[1], d3 = norm[2], d4 = norm[3],
            d5 = norm[4], d6 = norm[5], d7 = norm[6], d8 = norm[7],
            samples,
            "CTR-derived dimension weights"
        );

        Some(AlgoWeights {
            d1_engagement_velocity:  norm[0],
            d2_content_intelligence: norm[1],
            d3_social_graph:         norm[2],
            d4_temporal:             norm[3],
            d5_behavioral:           norm[4],
            d6_diversity:            norm[5],
            d7_viral:                norm[6],
            d8_personalization:      norm[7],
        })
    }
}
