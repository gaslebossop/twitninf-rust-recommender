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

pub const N_FEATURES: usize = 22;

/// Indice du premier CROISEMENT de traits.
///
/// ── Ce que ces cinq traits achetent ────────────────────────────────────────
/// Une regression logistique est ADDITIVE : elle ne peut pas apprendre « le
/// graphe social compte davantage pour un lecteur tres actif ». Elle apprend un
/// poids pour le graphe social et un poids pour l'activite du lecteur, et les
/// additionne — l'interaction entre les deux lui est structurellement
/// inaccessible.
///
/// C'est precisement ce qu'un reseau de neurones apprendrait a notre place. Or
/// avec 16 traits DENSES (pas des identifiants a forte cardinalite), le nombre
/// d'interactions utiles est petit et connu : on peut les ecrire. C'est le pont
/// classique entre regression logistique et reseau — la machine a factorisation
/// fait la meme chose en apprenant toutes les paires, ce qui coute `k·n`
/// parametres la ou nous en depensons cinq.
///
/// ── Pourquoi APRES la feature de position ──────────────────────────────────
/// `POSITION_FEATURE` vaut 15 et est lu par `record_impressions`. Les nouveaux
/// traits sont donc ajoutes A LA FIN : la migration des modeles persistes
/// (`migrate_legacy_weights`) complete un vecteur plus court par les defauts,
/// donc les poids deja appris sur les 16 premiers traits survivent intacts. Ne
/// jamais inserer un trait AVANT l'indice 15.
pub const CROSS_BASE: usize = 16;

/// Indice de la feature « rang auquel ce tweet a ete SERVI ».
///
/// ── Le biais que ca corrige ─────────────────────────────────────────────────
/// Un tweet en tete de page est clique bien plus qu'un tweet en position 40, a
/// qualite strictement egale : c'est un effet du RANG, pas du contenu. Sans
/// cette feature, le modele attribue tout cet ecart au contenu, apprend « les
/// tweets comme celui-la marchent bien », les remonte encore, et le fil se
/// fige sur ce qui etait deja en tete. C'est la boucle de retroaction la plus
/// classique d'un systeme de recommandation, et le moteur n'avait rien contre
/// elle.
///
/// ── La correction, et pourquoi elle ne coute rien au client ─────────────────
/// Recette dite de la « tour peu profonde » (YouTube) : on ENTRAINE avec le
/// rang reel, et on PREDIT avec un rang fixe. Le poids appris sur cette
/// feature absorbe l'effet de rang ; au moment de classer, tous les candidats
/// recoivent la meme valeur, donc ce terme s'annule dans la comparaison et il
/// ne reste que le contenu.
///
/// Le rang reel n'a pas besoin de venir du client : le serveur SAIT a quelle
/// place il a servi chaque tweet. Il est donc ecrit dans le vecteur au moment
/// ou l'impression est memorisee (`record_impressions`), pas au moment du
/// scoring — ou la pagination n'a pas encore eu lieu.
pub const POSITION_FEATURE: usize = 15;

/// Escompte de position : 1/log2(rang + 2).
///
/// Meme forme que l'escompte du NDCG, et pour la meme raison : la perte
/// d'attention est brutale en haut de page et lente ensuite. Un ecart lineaire
/// en rang decrirait mal ce que fait un lecteur qui fait defiler.
/// Vaut ~0,63 en tete, 0,5 a la 3e place, 0,25 a la 15e.
pub fn position_discount(rank: usize) -> f64 {
    1.0 / ((rank + 2) as f64).log2()
}

/// Valeur de rang utilisee AU MOMENT DE CLASSER.
///
/// Identique pour tous les candidats — c'est ce qui fait que le terme de
/// position s'annule dans la comparaison. On prend la tete de page : le
/// classement repond alors a « ce tweet marcherait-il s'il etait montre en
/// premier ? », qui est exactement la question a poser pour decider de l'ordre.
pub fn serving_position() -> f64 {
    position_discount(0)
}
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
    /// Moyenne courante des predictions — denominateur de `lift`.
    ///
    /// Voir `ml::objectives::Head::lift` pour le raisonnement complet. En deux
    /// mots : `blend_positive` est une moyenne PONDEREE, et une tete qui predit
    /// autour de 0,05 face a un score de regles qui balaie 0,2–0,8 n'y apporte
    /// presque aucune variance — elle abaisse tous les scores d'a peu pres
    /// autant, ce qui ne change aucun ordre. Diviser par cette moyenne rend a
    /// la tete la plage qui lui est due.
    ///
    /// `serde(default)` : le modele deja persiste en production ne porte pas ce
    /// champ. Il repart de 0,0, et `lift` retombe sur son plancher le temps du
    /// premier echantillon.
    #[serde(default)]
    pub pred_mean: f64,
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
                0.20,  // escompte de position — voir `POSITION_FEATURE`. Prior
                       // POSITIF et net : plus le rang est haut, plus l'escompte
                       // est grand, plus le clic est probable. C'est le seul
                       // prior qu'on connaisse avec certitude avant tout
                       // apprentissage.
                // ── Croisements (voir `CROSS_BASE`) ────────────────────────
                // Priors FAIBLES et tous positifs : on affirme seulement que
                // ces produits vont dans le meme sens que leurs facteurs, ce
                // qui est le minimum defendable. Leur amplitude est de toute
                // facon petite — un produit de deux valeurs de [0,1] vaut en
                // moyenne bien moins que chacune d'elles.
                0.04, // d3 x activite du lecteur
                0.03, // d8 x has_media
                0.06, // d1 x is_recent — le croisement « ca decolle »
                0.03, // d2 x activite du lecteur
                0.03, // d5 x d8
                // Prior NET : si deux lecteurs qui se ressemblent aiment le
                // meme auteur, c'est le signal le plus direct qui existe.
                // C'est aussi le seul trait dont on sait d'avance qu'il
                // n'apporte rien quand il vaut 0,5 (voir `crate::collab`).
                0.15, // affinite collaborative
            ],
            bias: -2.5867, // logit(PRIOR_CTR) — au lieu d'un -0.5 arbitraire qui prédisait ~38 % de CTR à froid
            learning_rate: 0.01,
            samples_seen: 0,
            total_clicks: 0,
            total_views: 0,
            pred_mean: PRIOR_CTR,
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
        // Entretenue ici et pas dans `predict` : `predict` ne prend qu'un
        // verrou de LECTURE et tourne sur chaque candidat de chaque fil.
        let n = self.samples_seen as f64;
        self.pred_mean += (pred - self.pred_mean) / n;
    }

    /// Prediction ramenee sur l'echelle commune du melange — voir `pred_mean`
    /// et `ml::objectives::Head::lift`. Centree sur 0,5 quel que soit le taux
    /// de base observe.
    pub fn lift(&self, features: &[f64; N_FEATURES]) -> f64 {
        let p = self.predict(features);
        let mean = self.pred_mean.max(1e-4);
        let l = p / mean;
        l / (l + 1.0)
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
    collab_affinity: f64,
) -> [f64; N_FEATURES] {
    extract_features_at(
        d1,
        d2,
        d3,
        d4,
        d5,
        d6,
        d7,
        d8,
        age_h,
        is_trending,
        has_media,
        author_followers,
        acceleration,
        reader_engagement,
        serving_position(),
        collab_affinity,
    )
}

/// Même vecteur, avec un escompte de position explicite.
///
/// Utilisé à l'ENTRAÎNEMENT, où l'on connaît le rang réellement servi. Le
/// chemin de classement passe par `extract_features` ci-dessus, qui fixe la
/// position à la tête de page pour tout le monde — voir `POSITION_FEATURE`.
#[allow(clippy::too_many_arguments)]
pub fn extract_features_at(
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
    position: f64,
    collab_affinity: f64,
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
        position.clamp(0.0, 1.0),
        // ── Croisements — voir `CROSS_BASE` ─────────────────────────────
        // 16 · graphe social x activite du lecteur. Un lecteur qui consomme
        //      beaucoup a un graphe qui veut dire quelque chose ; celui d'un
        //      compte inactif est une liste morte.
        d3 * reader_engagement.clamp(0.0, 1.0),
        // 17 · personnalisation x media. Le meme sujet ne se consomme pas de
        //      la meme facon en texte et en video.
        d8 * if has_media { 1.0 } else { 0.0 },
        // 18 · velocite x fraicheur. Vingt j'aime en une heure et vingt
        //      j'aime en trois jours sont le meme D1 et pas du tout le meme
        //      evenement — c'est le croisement qui distingue un contenu qui
        //      DECOLLE d'un contenu qui a simplement vecu.
        d1 * if age_h < 2.0 { 1.0 } else { 0.0 },
        // 19 · contenu x activite du lecteur. Un gros consommateur devient
        //      plus exigeant sur la qualite ; un lecteur occasionnel prend ce
        //      qu'on lui donne.
        d2 * reader_engagement.clamp(0.0, 1.0),
        // 20 · comportement x personnalisation. Les deux dimensions qui
        //      decrivent le lecteur : leur produit dit « ce tweet lui
        //      ressemble ET correspond a ses habitudes », ce que la somme des
        //      deux ne distingue pas d'« il excelle sur une seule des deux ».
        d5 * d8,
        // 21 · AFFINITE COLLABORATIVE — voir `crate::collab`.
        //
        //      Le seul trait qui ne se deduit ni du tweet ni du profil : il
        //      vient de la position du lecteur ET de celle de l'auteur dans un
        //      espace factorise a partir du graphe de co-appreciation. C'est le
        //      pendant du produit scalaire « interested in » x « known for » de
        //      SimClusters.
        //
        //      0,5 quand l'un des deux n'est pas placable : la valeur d'un
        //      cosinus nul, c'est-a-dire « aucun rapport constate », qui est
        //      bien ce qu'on sait dans ce cas.
        collab_affinity.clamp(0.0, 1.0),
    ]
}

/// Service thread-safe pour le modèle CTR partagé entre requêtes
#[derive(Clone)]
pub struct CtrPredictor(Arc<RwLock<CtrModel>>, crate::eval::OnlineEval);

impl CtrPredictor {
    pub fn new() -> Self {
        Self(
            Arc::new(RwLock::new(CtrModel::default())),
            crate::eval::OnlineEval::new(),
        )
    }

    /// Qualité mesurée du modèle sur la fenêtre glissante récente — voir
    /// `crate::eval`. `samples_seen` dit combien le modèle a vu, ceci dit s'il
    /// en a tiré quoi que ce soit.
    pub fn eval_report(&self) -> crate::eval::EvalReport {
        self.1.report()
    }

    /// Ce qu'une recalibration rattraperait — mesure seulement, rien n'est
    /// appliqué. Voir `crate::ml::calibrator`.
    pub fn calibration_gain(&self) -> Option<crate::ml::CalibrationGain> {
        self.1.calibration_gain()
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
                        return Self(
                            Arc::new(RwLock::new(model)),
                            crate::eval::OnlineEval::new(),
                        );
                    }
                    Err(e) => match migrate_legacy_weights(&json) {
                        Some(model) => {
                            info!(
                                samples = model.samples_seen,
                                global_ctr = model.global_ctr(),
                                "CTR model migrated from a shorter feature vector — \
                                 learned weights kept, new feature(s) seeded at their default"
                            );
                            return Self(
                                Arc::new(RwLock::new(model)),
                                crate::eval::OnlineEval::new(),
                            );
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

    /// Ce que le CLASSEMENT consomme — voir `CtrModel::lift`. `predict_ctr`
    /// reste la probabilite brute, qui est ce qu'il faut pour mesurer une AUC
    /// ou une calibration.
    pub fn ctr_lift(&self, features: &[f64; N_FEATURES]) -> f64 {
        self.0.read().unwrap().lift(features)
    }

    /// Met à jour le modèle sur un event click/skip
    pub fn record_interaction(&self, features: [f64; N_FEATURES], clicked: bool) {
        let mut model = self.0.write().unwrap();
        // Validation progressive : la prédiction est prise AVANT la mise à
        // jour, donc sur un exemple que le modèle n'a jamais vu. Prise après,
        // elle décrirait un modèle qui a déjà lu la réponse — et l'AUC qui en
        // sortirait serait une flatterie, pas un diagnostic. Voir `crate::eval`.
        let prior_prediction = model.predict(&features);
        model.update(&features, clicked);
        let samples = model.samples_seen;
        let global_ctr = model.global_ctr();
        drop(model);
        self.1
            .record(prior_prediction, if clicked { 1.0 } else { 0.0 });
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
            0.5,
        );
        let p = model.predict(&features);
        assert!(p >= 0.0 && p <= 1.0, "CTR prediction out of range: {p}");
    }

    #[test]
    fn test_learning_improves_high_engagement() {
        let mut model = CtrModel::default();
        let high_eng = extract_features(
            0.9, 0.8, 0.7, 0.6, 0.6, 0.8, 0.7, 0.5, 0.5, true, true, 50000, 0.9, 0.5,
            0.5,
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




    // ─── Correction du biais de position ────────────────────────────────────

    #[test]
    fn l_escompte_de_position_decroit_avec_le_rang() {
        let tete = position_discount(0);
        let milieu = position_discount(10);
        let bas = position_discount(45);
        assert!(tete > milieu && milieu > bas, "{tete} {milieu} {bas}");
        // Reste une valeur exploitable par un modele lineaire : borne, jamais
        // nulle, jamais superieure a 1.
        for rang in [0, 1, 5, 49, 500, 100_000] {
            let d = position_discount(rang);
            assert!(d > 0.0 && d <= 1.0, "rang={rang} escompte={d}");
        }
    }

    /// La perte doit etre BRUTALE en haut de page et lente ensuite : passer de
    /// la place 1 a la place 4 coute bien plus que de passer de la 41 a la 44.
    /// Un escompte lineaire en rang dirait le contraire.
    #[test]
    fn la_perte_est_concentree_en_haut_de_page() {
        let haut = position_discount(0) - position_discount(3);
        let bas = position_discount(40) - position_discount(43);
        assert!(haut > bas * 3.0, "haut={haut} bas={bas}");
    }

    /// Le coeur de la correction : au moment de CLASSER, tous les candidats
    /// recoivent la meme valeur de position. Le terme s'annule donc dans la
    /// comparaison, et il ne reste que le contenu.
    #[test]
    fn le_classement_utilise_la_meme_position_pour_tout_le_monde() {
        let a = extract_features(
            0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, true, true, 100, 0.5, 0.5,
            0.5,
        );
        let b = extract_features(
            0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 9.0, false, false, 5, 0.1, 0.2,
            0.5,
        );
        assert_eq!(a[POSITION_FEATURE], b[POSITION_FEATURE]);
        assert_eq!(a[POSITION_FEATURE], serving_position());
    }

    /// Et a l'ENTRAINEMENT, la position reelle passe bien dans le vecteur —
    /// c'est ce qui permet au poids d'absorber l'effet de rang.
    #[test]
    fn l_entrainement_recoit_la_position_reelle() {
        let tete = extract_features_at(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, false, false, 10, 0.5, 0.5,
            position_discount(0),
            0.5,
        );
        let bas = extract_features_at(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, false, false, 10, 0.5, 0.5,
            position_discount(40),
            0.5,
        );
        assert!(tete[POSITION_FEATURE] > bas[POSITION_FEATURE]);
        // Tout le reste du vecteur est identique : seule la position change.
        assert_eq!(&tete[..POSITION_FEATURE], &bas[..POSITION_FEATURE]);
    }

    /// Un modele nourri de clics en haut de page et de non-clics en bas doit
    /// apprendre un poids de position POSITIF — c'est exactement l'effet de
    /// rang, et c'est ce poids qui l'absorbe au lieu de le laisser contaminer
    /// les dimensions de contenu.
    #[test]
    fn le_modele_apprend_a_absorber_l_effet_de_rang() {
        let mut model = CtrModel::default();
        let contenu = |position: f64| {
            extract_features_at(
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, false, false, 100, 0.5, 0.5, position,
                0.5,
            )
        };
        let mut rng = 0xDEADBEEFCAFEBABEu64;
        for _ in 0..3_000 {
            // MEME contenu des deux cotes : seul le rang differe. Tout ecart
            // appris ne peut donc venir que du rang.
            if melange(&mut rng) {
                model.update(&contenu(position_discount(0)), true);
            } else {
                model.update(&contenu(position_discount(45)), false);
            }
        }
        assert!(
            model.weights[POSITION_FEATURE] > 0.0,
            "poids de position appris = {}",
            model.weights[POSITION_FEATURE]
        );
        // Et a rang egal, le modele ne distingue plus les deux : l'ecart a
        // bien ete impute au rang, pas au contenu.
        let p_tete = model.predict(&contenu(serving_position()));
        let p_bas = model.predict(&contenu(serving_position()));
        assert!((p_tete - p_bas).abs() < 1e-12);
    }

    /// Tirage a pile ou face deterministe (generateur congruentiel simple).
    ///
    /// Indispensable ici, et pas par gout du realisme : une suite d'etiquettes
    /// strictement ALTERNEE fait osciller un apprenant en ligne en phase avec
    /// elle. Chaque prediction, prise avant la mise a jour, herite du biais
    /// deplace par l'echantillon precedent — donc elle est systematiquement
    /// haute juste avant un negatif et basse juste avant un positif. La mesure
    /// progressive y lit une anti-correlation parfaite : sur du bruit pur en
    /// alternance stricte, l'AUC tombe a 0,06 au lieu de 0,50.
    ///
    /// Ce n'est pas un defaut du calcul, c'est la limite du protocole, notee
    /// dans `crate::eval` : la validation progressive suppose que l'ordre
    /// d'arrivee n'est pas correle a l'etiquette. Le trafic reel ne l'est pas ;
    /// un test qui alterne, si.
    fn melange(state: &mut u64) -> bool {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) & 1 == 1
    }

    /// Preuve de bout en bout que la mesure fonctionne — et qu'elle est
    /// honnête.
    ///
    /// On alimente le modèle en ligne avec un signal REELLEMENT separable (les
    /// tweets a forte velocite sont cliques, les autres non), et on verifie
    /// que l'AUC mesuree en validation progressive monte nettement au-dessus
    /// du hasard. Chaque prediction comptee a ete faite AVANT que le modele ne
    /// voie l'echantillon correspondant : si le protocole etait casse (mesure
    /// prise apres la mise a jour), ce test passerait aussi — c'est le test
    /// jumeau ci-dessous qui ferme cette porte.
    #[test]
    fn la_mesure_progressive_detecte_un_modele_qui_apprend() {
        let predictor = CtrPredictor::new();
        let fort = extract_features(
            0.95, 0.8, 0.7, 0.6, 0.6, 0.5, 0.8, 0.5, 0.5, true, true, 50_000, 0.9, 0.6,
            0.5,
        );
        let faible = extract_features(
            0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 20.0, false, false, 3, 0.0, 0.05,
            0.5,
        );

        // Ordre pseudo-aleatoire, pas alterne : voir `melange` plus bas.
        let mut rng = 0x2545F4914F6CDD1Du64;
        for _ in 0..2_000 {
            if melange(&mut rng) {
                predictor.record_interaction(fort, true);
            } else {
                predictor.record_interaction(faible, false);
            }
        }

        let report = predictor.eval_report();
        assert!(report.samples >= crate::eval::MIN_SAMPLES_FOR_METRICS);
        let auc = report.auc.expect("les deux classes sont presentes");
        assert!(
            auc > 0.90,
            "un signal parfaitement separable doit ressortir : auc={auc}"
        );
        // Et la calibration doit suivre : le taux reel est de 50 %.
        assert!(
            (report.positive_rate - 0.5).abs() < 0.02,
            "taux observe={}",
            report.positive_rate
        );
    }

    /// Le pendant : sur un signal qui ne contient AUCUNE information (meme
    /// vecteur de features, etiquette tiree a pile ou face), l'AUC mesuree doit
    /// rester au niveau du hasard. Un harnais qui trouverait du signal ici
    /// serait un harnais qui ment.
    #[test]
    fn la_mesure_progressive_ne_trouve_rien_dans_du_bruit() {
        let predictor = CtrPredictor::new();
        let f = extract_features(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, false, true, 1_000, 0.5, 0.5,
            0.5,
        );
        let mut rng = 0x9E3779B97F4A7C15u64;
        for _ in 0..2_000 {
            let label = melange(&mut rng);
            predictor.record_interaction(f, label);
        }
        let auc = predictor.eval_report().auc.expect("deux classes");
        assert!(
            (auc - 0.5).abs() < 0.05,
            "aucune information a extraire, l'AUC doit rester au hasard : auc={auc}"
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
            0.5,
        );
        let before = model.predict(&features);
        model.update(&features, true); // clicked → should increase
        let after = model.predict(&features);
        assert!(after >= before, "Click should push prediction up");
    }
}
