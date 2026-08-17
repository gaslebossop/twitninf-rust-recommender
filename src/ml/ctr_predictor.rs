/// Phase 2 — ML CTR Predictor
///
/// Logistic regression avec SGD online learning.
/// Input: 15 features (scores D1-D8 + contexte tweet + activité du lecteur)
/// Output: CTR probability [0, 1]
///
/// Entraînement continu depuis les interactions utilisateurs.
/// Convergence typique : ~500 interactions, gain CTR : +1.5-2%
///
/// ⚠ Changer `N_FEATURES` change la forme du tableau persisté
/// (`data/ctr_model.json`). `load_or_default` migre automatiquement un modèle
/// dont `weights` est plus court (voir `migrate_legacy_weights`) : les poids
/// appris et `samples_seen` sont conservés, seul(s) le/les nouveau(x) poids
/// sont seedé(s) à leur valeur par défaut. Un tableau plus LONG que
/// `N_FEATURES` (retour en arrière du code) reste, lui, un modèle neuf — pas
/// de sens à deviner quelle feature a disparu.
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::fs;
use tracing::{debug, info, warn};

pub const N_FEATURES: usize = 15;
const MODEL_PATH: &str = "data/ctr_model.json";

/// CTR de référence avant tout entraînement — cohérent avec le prior de
/// `global_ctr()` ci-dessous.
const PRIOR_CTR: f64 = 0.07;

/// Multiplicateur de taux d'apprentissage appliqué au seul biais.
///
/// Avec un CTR réel de l'ordre de 1-2 %, la correction de calibration à froid
/// est massive : partie d'un biais mal calé, elle doit être encaissée par un
/// seul paramètre plutôt que diffusée sur les 8 poids de dimension. Sans ce
/// multiplicateur, observé en prod après 10k samples : les 8 poids D1-D8
/// finissent tous négatifs (le biais seul n'avait pas assez convergé), et
/// l'auto-tuner (voir `extract_dimension_weights`) n'a alors plus aucun signal
/// de dimension à extraire.
const BIAS_LR_MULTIPLIER: f64 = 8.0;

/// Modèle logistique entraînable en ligne (SGD)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtrModel {
    /// Poids des 14 features [D1..D8, age, trending, media, followers_log, recency, accel]
    pub weights: [f64; N_FEATURES],
    pub bias: f64,
    pub learning_rate: f64,
    pub samples_seen: u64,
    pub total_clicks: u64,
    pub total_views: u64,
}

impl Default for CtrModel {
    fn default() -> Self {
        Self {
            weights: [
                0.35,  // d1 engagement velocity — meilleur prédicteur CTR
                0.15,  // d2 content intelligence
                0.12,  // d3 social graph
                0.08,  // d4 temporal
                0.08,  // d5 behavioral
                0.05,  // d6 diversity
                0.10,  // d7 viral
                0.04,  // d8 personalization
                -0.05, // age_h normalisé (plus vieux = moins de CTR)
                0.10,  // is_trending (0 ou 1)
                0.08,  // has_media (0 ou 1)
                0.03,  // log(author_followers) / 20
                0.05,  // is_recent (< 2h)
                0.07,  // engagement_acceleration
                0.03,  // activité du LECTEUR (engagement_velocity/20, clampé) —
                       // seule feature qui décrit qui regarde, pas ce qui est
                       // regardé ; prior faible tant qu'elle n'a pas appris
            ],
            bias: -2.5867, // logit(PRIOR_CTR) — au lieu d'un -0.5 arbitraire qui prédisait ~38 % de CTR à froid
            learning_rate: 0.01,
            samples_seen: 0,
            total_clicks: 0,
            total_views: 0,
        }
    }
}

impl CtrModel {
    /// Prédit la probabilité de CTR (sigmoid)
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        let z: f64 = self.bias
            + features
                .iter()
                .zip(self.weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f64>();
        sigmoid(z)
    }

    /// Mise à jour SGD sur un événement click/skip
    /// Learning rate décroissant pour stabilité
    pub fn update(&mut self, features: &[f64; N_FEATURES], clicked: bool) {
        let pred = self.predict(features);
        let label = if clicked { 1.0 } else { 0.0 };
        let error = label - pred;

        // Learning rate avec décroissance inverse de racine (Robbins-Monro)
        let lr = self.learning_rate / (1.0 + 0.001 * self.samples_seen as f64).sqrt();

        self.bias += lr * BIAS_LR_MULTIPLIER * error;
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += lr * error * f;
            // L2 regularisation légère (λ=0.0001) pour éviter overfitting
            *w *= 1.0 - lr * 0.0001;
        }

        self.samples_seen += 1;
        if clicked {
            self.total_clicks += 1;
        }
        self.total_views += 1;
    }

    pub fn global_ctr(&self) -> f64 {
        if self.total_views == 0 {
            return PRIOR_CTR;
        }
        self.total_clicks as f64 / self.total_views as f64
    }
}

/// Extrait le vecteur de features depuis les scores et métadonnées du tweet.
///
/// `reader_engagement` est la seule feature qui décrit qui regarde plutôt que
/// ce qui est regardé — jusqu'ici ce modèle apprenait « quel tweet clique en
/// général », un même prior de clic pour tout le monde, quel que soit le
/// lecteur. Normalisée en amont (voir `crate::algorithm::scoring::ctr_feature_vector`).
#[allow(clippy::too_many_arguments)]
pub fn extract_features(
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
    d6: f64,
    d7: f64,
    d8: f64,
    age_h: f64,
    is_trending: bool,
    has_media: bool,
    author_followers: i64,
    acceleration: f64,
    reader_engagement: f64,
) -> [f64; N_FEATURES] {
    [
        d1,
        d2,
        d3,
        d4,
        d5,
        d6,
        d7,
        d8,
        (age_h / 24.0).min(1.0), // âge normalisé [0,1]
        if is_trending { 1.0 } else { 0.0 },
        if has_media { 1.0 } else { 0.0 },
        (author_followers as f64 + 1.0).ln() / 20.0, // log-followers normalisé
        if age_h < 2.0 { 1.0 } else { 0.0 },         // is_recent
        acceleration.clamp(0.0, 1.0),
        reader_engagement.clamp(0.0, 1.0),
    ]
}

/// Service thread-safe pour le modèle CTR partagé entre requêtes
#[derive(Clone)]
pub struct CtrPredictor(Arc<RwLock<CtrModel>>);

impl CtrPredictor {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(CtrModel::default())))
    }

    pub async fn load_or_default() -> Self {
        if Path::new(MODEL_PATH).exists() {
            match fs::read_to_string(MODEL_PATH).await {
                Ok(json) => match serde_json::from_str::<CtrModel>(&json) {
                    Ok(model) => {
                        info!(
                            samples = model.samples_seen,
                            global_ctr = model.global_ctr(),
                            "CTR model loaded from disk"
                        );
                        return Self(Arc::new(RwLock::new(model)));
                    }
                    Err(e) => match migrate_legacy_weights(&json) {
                        Some(model) => {
                            info!(
                                samples = model.samples_seen,
                                global_ctr = model.global_ctr(),
                                "CTR model migrated from a shorter feature vector — \
                                 learned weights kept, new feature(s) seeded at their default"
                            );
                            return Self(Arc::new(RwLock::new(model)));
                        }
                        None => warn!("Failed to parse CTR model: {e}, using default"),
                    },
                },
                Err(e) => warn!("Failed to read CTR model: {e}, using default"),
            }
        }
        info!("Starting with default CTR model (no training data yet)");
        Self::new()
    }

    /// Retourne le score CTR prédit pour un tweet scoré
    pub fn predict_ctr(&self, features: &[f64; N_FEATURES]) -> f64 {
        self.0.read().unwrap().predict(features)
    }

    /// Met à jour le modèle sur un event click/skip
    pub fn record_interaction(&self, features: [f64; N_FEATURES], clicked: bool) {
        let mut model = self.0.write().unwrap();
        model.update(&features, clicked);
        let samples = model.samples_seen;
        let global_ctr = model.global_ctr();
        drop(model);
        debug!(samples, global_ctr, clicked, "CTR model updated");
    }

    /// Sauvegarde le modèle sur disque (à appeler périodiquement)
    pub async fn save(&self) {
        let model = self.0.read().unwrap().clone();
        drop(model);
        let model = self.0.read().unwrap().clone();
        match serde_json::to_string_pretty(&model) {
            Ok(json) => {
                let _ = fs::create_dir_all("data").await;
                match fs::write(MODEL_PATH, json).await {
                    Ok(_) => info!(samples = model.samples_seen, "CTR model saved"),
                    Err(e) => warn!("Failed to save CTR model: {e}"),
                }
            }
            Err(e) => warn!("Failed to serialize CTR model: {e}"),
        }
    }

    pub fn stats(&self) -> (u64, f64) {
        let m = self.0.read().unwrap();
        (m.samples_seen, m.global_ctr())
    }

    /// Retourne les 8 premiers poids (D1-D8) pour l'auto-tuner
    pub fn dimension_weights_snapshot(&self) -> [f64; 8] {
        let m = self.0.read().unwrap();
        let w = &m.weights;
        [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

/// Complète un modèle persisté dont le vecteur `weights` est plus court que
/// `N_FEATURES` (schéma enrichi depuis la sauvegarde), au lieu de le jeter.
///
/// `bias`, `samples_seen`, `total_clicks`, `total_views` et les poids déjà
/// appris restent intacts ; seuls les poids manquants sont seedés à leur
/// valeur par défaut. Sans ça, chaque élargissement du vecteur de features
/// remettrait l'entraînement à zéro — ici ça representait 43 751 samples
/// perdus au moment où `reader_engagement` (feature #15) a été ajoutée.
fn migrate_legacy_weights(json: &str) -> Option<CtrModel> {
    let mut value: serde_json::Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object_mut()?;
    let old_weights = obj.get("weights")?.as_array()?.clone();
    if old_weights.is_empty() || old_weights.len() >= N_FEATURES {
        return None;
    }
    let defaults = CtrModel::default().weights;
    let mut padded = old_weights;
    for w in defaults.iter().skip(padded.len()) {
        padded.push(serde_json::json!(w));
    }
    obj.insert("weights".to_string(), serde_json::Value::Array(padded));
    serde_json::from_value(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_range() {
        let model = CtrModel::default();
        let features = extract_features(
            0.8, 0.7, 0.6, 0.5, 0.5, 0.7, 0.6, 0.4, 1.0, true, true, 10000, 0.8, 0.5,
        );
        let p = model.predict(&features);
        assert!(p >= 0.0 && p <= 1.0, "CTR prediction out of range: {p}");
    }

    #[test]
    fn test_learning_improves_high_engagement() {
        let mut model = CtrModel::default();
        let high_eng = extract_features(
            0.9, 0.8, 0.7, 0.6, 0.6, 0.8, 0.7, 0.5, 0.5, true, true, 50000, 0.9, 0.5,
        );
        let initial_pred = model.predict(&high_eng);

        // Simulate 100 clicks on high-engagement tweets
        for _ in 0..100 {
            model.update(&high_eng, true);
        }
        let trained_pred = model.predict(&high_eng);
        assert!(
            trained_pred > initial_pred,
            "Model should learn to predict high CTR"
        );
    }

    #[test]
    fn migration_garde_les_poids_appris_et_seed_le_reste_au_defaut() {
        let legacy = serde_json::json!({
            "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "bias": -1.2345,
            "learning_rate": 0.01,
            "samples_seen": 43751,
            "total_clicks": 900,
            "total_views": 43751,
        });
        let migrated =
            migrate_legacy_weights(&legacy.to_string()).expect("14 poids doit migrer vers 15");
        assert_eq!(migrated.weights.len(), N_FEATURES);
        assert_eq!(
            &migrated.weights[0..14],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        );
        assert_eq!(migrated.weights[14], CtrModel::default().weights[14]);
        assert_eq!(migrated.samples_seen, 43751);
        assert_eq!(migrated.bias, -1.2345);
    }

    #[test]
    fn migration_refuse_un_vecteur_deja_a_la_bonne_taille() {
        let current = serde_json::to_string(&CtrModel::default()).unwrap();
        assert!(migrate_legacy_weights(&current).is_none());
    }

    #[test]
    fn test_sgd_update_direction() {
        let mut model = CtrModel::default();
        let features = extract_features(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, false, false, 1000, 0.5, 0.5,
        );
        let before = model.predict(&features);
        model.update(&features, true); // clicked → should increase
        let after = model.predict(&features);
        assert!(after >= before, "Click should push prediction up");
    }
}
