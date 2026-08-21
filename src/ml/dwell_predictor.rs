/// Phase 4 — ML Dwell Predictor
///
/// Régression logistique avec SGD en ligne, même mécanique que `CtrPredictor`
/// (voir `ctr_predictor.rs`) : mêmes 15 features, même mise à jour par erreur
/// de gradient. Ce qui change, c'est la CIBLE — pas un clic (0/1), mais le
/// poids de temps de lecture déjà calculé par `algorithm::dwell::dwell_weight`
/// une fois l'interaction observée, ramené sur [0, 1] pour tenir dans la même
/// mécanique sigmoïde, puis reprojeté sur son échelle d'origine à la lecture.
///
/// ⚠ Ce module PRÉDIT un temps de lecture attendu avant que le lecteur ait vu
/// le tweet — ce que `dwell.rs` ne fait pas : lui ne fait que NORMALISER un
/// temps déjà observé. Les deux se complètent : celui-ci sert au classement,
/// celui-là sert à noter ce qui vient d'être observé pour entraîner celui-ci.
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::algorithm::dwell::{MAX_BONUS, SKIP_PENALTY};
use crate::ml::ctr_predictor::N_FEATURES;

const MODEL_PATH: &str = "data/dwell_model.json";

/// Étendue du poids de dwell (voir `algorithm::dwell`) — la mécanique sigmoïde
/// prédit dans [0, 1], reprojeté sur cette plage à la lecture.
const RANGE: f64 = MAX_BONUS - SKIP_PENALTY;

/// Poids de dwell neutre (ni consommé ni survolé) une fois ramené sur [0, 1] :
/// `(0.0 - SKIP_PENALTY) / RANGE`. Sert de calibrage de départ pour le biais,
/// même raisonnement que `PRIOR_CTR` dans `ctr_predictor.rs` — mieux vaut
/// démarrer neutre que de deviner un biais arbitraire.
const NEUTRAL01: f64 = -SKIP_PENALTY / RANGE;

/// Voir `ctr_predictor::BIAS_LR_MULTIPLIER` — même raison : encaisser le
/// recalibrage initial sur un seul paramètre plutôt que sur les 15 poids.
const BIAS_LR_MULTIPLIER: f64 = 8.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwellModel {
    pub weights: [f64; N_FEATURES],
    pub bias: f64,
    pub learning_rate: f64,
    pub samples_seen: u64,
    /// Moyenne glissante (non pondérée) du poids de dwell observé, sur
    /// l'échelle d'origine — pour diagnostic (`/admin/algo/stats`), jamais
    /// utilisée dans la prédiction elle-même.
    pub running_mean_weight: f64,
}

impl Default for DwellModel {
    fn default() -> Self {
        Self {
            // Prior FAIBLE et uniforme, contrairement au CTR : on n'a pas
            // d'historique calibré dimension par dimension pour « ce qui
            // retient l'attention », le modèle l'apprend en ligne. Seuls
            // les traits ayant un lien évident avec la longueur/nature du
            // contenu partent non nuls.
            weights: [
                0.05, // d1 engagement velocity
                0.10, // d2 content intelligence — contenu jugé riche, plausible que ça retienne
                0.03, // d3 social graph
                0.02, // d4 temporal
                0.05, // d5 behavioral
                0.02, // d6 diversity
                0.03, // d7 viral
                0.04, // d8 personalization
                0.0,  // age_h normalisé
                0.02, // is_trending
                0.12, // has_media — image/vidéo demande un temps de consommation non nul par construction
                0.02, // log(author_followers)
                0.0,  // is_recent
                0.02, // engagement_acceleration
                0.03, // activité du LECTEUR
            ],
            bias: logit(NEUTRAL01),
            learning_rate: 0.01,
            samples_seen: 0,
            running_mean_weight: 0.0,
        }
    }
}

impl DwellModel {
    /// Prédit un poids de dwell dans [SKIP_PENALTY, MAX_BONUS] — l'échelle
    /// déjà utilisée partout ailleurs (`dwell_weight`, `dwell_bonus`).
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        SKIP_PENALTY + self.predict01(features) * RANGE
    }

    /// Sortie brute du modèle, avant reprojection — c'est celle-là qui entre
    /// dans le calcul de gradient.
    fn predict01(&self, features: &[f64; N_FEATURES]) -> f64 {
        let z: f64 = self.bias
            + features
                .iter()
                .zip(self.weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f64>();
        sigmoid(z)
    }

    /// Met à jour le modèle sur un poids de dwell RÉELLEMENT observé
    /// (`algorithm::dwell::dwell_weight`, pas la valeur brute en ms).
    pub fn update(&mut self, features: &[f64; N_FEATURES], observed_weight: f64) {
        let label01 = ((observed_weight.clamp(SKIP_PENALTY, MAX_BONUS)) - SKIP_PENALTY) / RANGE;
        let pred01 = self.predict01(features);
        let error = label01 - pred01;

        let lr = self.learning_rate / (1.0 + 0.001 * self.samples_seen as f64).sqrt();

        self.bias += lr * BIAS_LR_MULTIPLIER * error;
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += lr * error * f;
            *w *= 1.0 - lr * 0.0001; // L2 légère, même valeur que le CTR
        }

        self.samples_seen += 1;
        let n = self.samples_seen as f64;
        self.running_mean_weight += (observed_weight - self.running_mean_weight) / n;
    }
}

#[derive(Clone)]
pub struct DwellPredictor(Arc<RwLock<DwellModel>>, crate::eval::OnlineEval);

impl DwellPredictor {
    pub fn new() -> Self {
        Self(
            Arc::new(RwLock::new(DwellModel::default())),
            crate::eval::OnlineEval::new(),
        )
    }

    /// Qualité mesurée sur la fenêtre glissante récente — voir `crate::eval`.
    ///
    /// La cible est CONTINUE ici : le rapport porte une RMSE, pas une AUC.
    /// Sortir une AUC d'un temps de lecture reviendrait à inventer un seuil, et
    /// le seuil choisi déciderait du résultat.
    pub fn eval_report(&self) -> crate::eval::EvalReport {
        self.1.report()
    }

    pub async fn load_or_default() -> Self {
        if Path::new(MODEL_PATH).exists() {
            match fs::read_to_string(MODEL_PATH).await {
                Ok(json) => match serde_json::from_str::<DwellModel>(&json) {
                    Ok(model) => {
                        info!(
                            samples = model.samples_seen,
                            mean_weight = model.running_mean_weight,
                            "Dwell model loaded from disk"
                        );
                        return Self(
                            Arc::new(RwLock::new(model)),
                            crate::eval::OnlineEval::new(),
                        );
                    }
                    Err(e) => warn!("Failed to parse dwell model: {e}, using default"),
                },
                Err(e) => warn!("Failed to read dwell model: {e}, using default"),
            }
        }
        info!("Starting with default dwell model (no training data yet)");
        Self::new()
    }

    pub fn predict_dwell(&self, features: &[f64; N_FEATURES]) -> f64 {
        self.0.read().unwrap().predict(features)
    }

    pub fn record_interaction(&self, features: [f64; N_FEATURES], observed_weight: f64) {
        let mut model = self.0.write().unwrap();
        // Validation progressive — voir `crate::eval`. Les deux valeurs sont
        // enregistrées sur l'échelle [0,1] interne au modèle, pas sur celle du
        // poids de dwell : une RMSE n'a de sens que si prédiction et vérité
        // vivent sur la même échelle.
        let prior_prediction01 = model.predict01(&features);
        let truth01 = ((observed_weight.clamp(SKIP_PENALTY, MAX_BONUS)) - SKIP_PENALTY) / RANGE;
        model.update(&features, observed_weight);
        let samples = model.samples_seen;
        let mean_weight = model.running_mean_weight;
        drop(model);
        self.1.record(prior_prediction01, truth01);
        debug!(samples, mean_weight, observed_weight, "Dwell model updated");
    }

    pub async fn save(&self) {
        let model = self.0.read().unwrap().clone();
        match serde_json::to_string_pretty(&model) {
            Ok(json) => {
                let _ = fs::create_dir_all("data").await;
                match fs::write(MODEL_PATH, json).await {
                    Ok(_) => info!(samples = model.samples_seen, "Dwell model saved"),
                    Err(e) => warn!("Failed to save dwell model: {e}"),
                }
            }
            Err(e) => warn!("Failed to serialize dwell model: {e}"),
        }
    }

    pub fn stats(&self) -> (u64, f64) {
        let m = self.0.read().unwrap();
        (m.samples_seen, m.running_mean_weight)
    }
}

impl Default for DwellPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

#[inline]
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::ctr_predictor::extract_features;

    fn sample_features() -> [f64; N_FEATURES] {
        extract_features(0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3, 0.5, false, true, 5000, 0.5, 0.5)
    }

    #[test]
    fn predict_reste_dans_la_plage_du_poids_de_dwell() {
        let model = DwellModel::default();
        let p = model.predict(&sample_features());
        assert!(
            p >= SKIP_PENALTY && p <= MAX_BONUS,
            "prédiction hors plage: {p}"
        );
    }

    #[test]
    fn apprend_a_predire_un_temps_consomme_en_entier() {
        let mut model = DwellModel::default();
        let features = sample_features();
        let before = model.predict(&features);
        for _ in 0..200 {
            model.update(&features, MAX_BONUS);
        }
        let after = model.predict(&features);
        assert!(
            after > before,
            "le modèle devrait apprendre à prédire un dwell plus élevé: avant={before} après={after}"
        );
    }

    #[test]
    fn apprend_a_predire_un_survol() {
        let mut model = DwellModel::default();
        let features = sample_features();
        let before = model.predict(&features);
        for _ in 0..200 {
            model.update(&features, SKIP_PENALTY);
        }
        let after = model.predict(&features);
        assert!(
            after < before,
            "le modèle devrait apprendre à prédire un dwell plus bas: avant={before} après={after}"
        );
    }

    #[test]
    fn le_biais_de_depart_predit_un_dwell_neutre() {
        // Vecteur de features nul : seul le biais parle, il doit retomber
        // près de 0.0 (ni consommé ni survolé) sur l'échelle d'origine.
        let model = DwellModel::default();
        let zero = [0.0; N_FEATURES];
        let p = model.predict(&zero);
        assert!((p - 0.0).abs() < 0.05, "biais de départ mal calibré: {p}");
    }

    #[test]
    fn running_mean_suit_les_observations() {
        let mut model = DwellModel::default();
        let features = sample_features();
        model.update(&features, MAX_BONUS);
        model.update(&features, SKIP_PENALTY);
        // Moyenne de deux observations opposées : proche de zéro.
        assert!(model.running_mean_weight.abs() < 0.3);
    }
}
