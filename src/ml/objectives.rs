//! Têtes de prédiction multi-objectifs.
//!
//! ── Le trou que ce module comble ────────────────────────────────────────────
//! Le moteur ne prédisait qu'UNE chose : « ce tweet sera-t-il engagé ? »
//! (`ml::ctr_predictor`), plus le temps de lecture attendu
//! (`ml::dwell_predictor`). Et cette unique tête d'engagement écrase toutes les
//! réactions dans un seul booléen : d'après `InteractionType::ctr_label`, un
//! like, un retweet, un marque-page et une visite de profil produisent le MÊME
//! exemple positif ; un survol, un signalement et un blocage le même exemple
//! négatif.
//!
//! Autrement dit, le modèle ne pouvait pas distinguer « ce tweet sera aimé » de
//! « ce tweet sera partagé », ni « ce tweet sera ignoré » de « ce tweet sera
//! signalé ». Or ces distinctions sont exactement ce qui sépare un classement
//! ordinaire d'un classement de niveau industriel : les moteurs de X et de
//! TikTok prédisent plusieurs probabilités séparées et les combinent par une
//! somme pondérée où certains termes sont NÉGATIFS. C'est ce qui permet de
//! rétrograder un contenu qui fera réagir — mais mal.
//!
//! ── Les deux têtes ajoutées ─────────────────────────────────────────────────
//! * **Amplification** — p(retweet / partage / marque-page / commentaire). Le
//!   geste le plus coûteux pour le lecteur, donc le plus informatif : il engage
//!   sa propre audience. Un like coûte un pouce, un retweet coûte une
//!   réputation. Un like SANS amplification est un exemple négatif de cette
//!   tête : le lecteur a vu, apprécié, et n'a pas relayé.
//! * **Rejet** — p(signalement / blocage / « ça ne m'intéresse pas »). Elle
//!   entre au classement avec un signe NÉGATIF. C'est la tête qui manquait le
//!   plus : sans elle, le seul moyen de rétrograder un contenu problématique
//!   était de le repérer après coup (signalements déjà reçus, étiquette de
//!   toxicité du LLM). Ici on prédit le rejet AVANT de montrer le tweet.
//!
//! ── Ce qui est délibérément réutilisé ───────────────────────────────────────
//! Les mêmes 15 features que le CTR et le dwell, et la même impression
//! mémorisée dans Redis. Aucune nouvelle plomberie de collecte : ces têtes
//! apprennent sur ce qui est DÉJÀ enregistré. C'est la raison pour laquelle
//! elles sont ajoutables à cette échelle de trafic — ce n'est pas trois fois
//! plus de données à récolter, c'est trois lectures différentes des mêmes
//! données.
//!
//! ── Démarrage à froid ───────────────────────────────────────────────────────
//! Chaque tête est gardée par son PROPRE compteur d'échantillons. Tant qu'elle
//! n'a pas atteint `MIN_SAMPLES`, elle ne pèse rien dans le classement : le
//! comportement est exactement celui d'avant ce module. Une tête qui apprend
//! plus lentement que l'autre (le rejet est rare par construction) n'entraîne
//! pas l'autre avec elle.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::ml::ctr_predictor::N_FEATURES;
use crate::models::InteractionType;

const MODEL_PATH: &str = "data/objective_models.json";

/// Échantillons avant qu'une tête ne pèse dans le classement. Même seuil que
/// le CTR et le dwell : un modèle tout juste initialisé ne doit jamais peser.
pub const MIN_SAMPLES: u64 = 200;

/// Voir `ctr_predictor::BIAS_LR_MULTIPLIER` — même raison : encaisser le
/// recalibrage initial sur un seul paramètre plutôt que sur les 15 poids.
const BIAS_LR_MULTIPLIER: f64 = 8.0;

/// Taux de base supposé avant tout apprentissage.
///
/// L'amplification est rare (quelques pour cent des impressions), le rejet
/// explicite l'est encore plus. Partir d'un biais qui prédit ~50 % ferait
/// démarrer les deux têtes avec une erreur énorme, encaissée pendant des
/// milliers d'échantillons — c'est le défaut que `PRIOR_CTR` corrige déjà
/// pour la tête d'engagement.
const PRIOR_AMPLIFY: f64 = 0.02;
const PRIOR_REJECT: f64 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════
// Objectifs
// ═══════════════════════════════════════════════════════════════════════════

/// Ce qu'une tête cherche à prédire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Le lecteur va-t-il RELAYER ce tweet ?
    Amplify,
    /// Le lecteur va-t-il REFUSER explicitement ce tweet ?
    Reject,
}

/// Étiquette d'entraînement d'une interaction pour un objectif donné.
///
/// `None` = cette interaction ne tranche pas pour cet objectif. Un `None` est
/// une réponse à part entière, pas un oubli : étiqueter au hasard ce qu'on ne
/// sait pas est précisément ce qui a cassé l'étiquetage du CTR par le passé
/// (une `View` comptée comme un clic, donc « tout est un clic »).
pub fn label_for(interaction: InteractionType, objective: Objective) -> Option<bool> {
    use InteractionType as I;
    match objective {
        // Relayer, c'est engager sa propre audience. Le like en est le
        // contre-exemple le plus utile : vu, apprécié, PAS relayé.
        Objective::Amplify => match interaction {
            I::Retweet | I::Share | I::Bookmark | I::Comment => Some(true),
            I::Like
            | I::Interested
            | I::ProfileView
            | I::Skip
            | I::Report
            | I::Block
            | I::NotInterested
            | I::Unretweet => Some(false),
            // Une vue ouvre la fenêtre d'attribution, elle ne conclut pas —
            // le balayage la comptera en négatif si rien ne suit.
            I::View => None,
            // Défaire un like dit quelque chose du like, pas du relais.
            I::Unlike => None,
        },
        // Refus EXPLICITE seulement. Un survol n'en est pas un : c'est
        // justement toute la différence entre « je passe » et « ceci n'aurait
        // pas dû m'être montré ». Les confondre redonnerait la tête unique
        // qu'on cherche à remplacer.
        Objective::Reject => match interaction {
            I::Report | I::Block | I::NotInterested => Some(true),
            I::Like
            | I::Comment
            | I::Retweet
            | I::Share
            | I::Bookmark
            | I::Interested
            | I::ProfileView => Some(false),
            I::View | I::Skip | I::Unlike | I::Unretweet => None,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tête logistique
// ═══════════════════════════════════════════════════════════════════════════

/// Régression logistique entraînée en ligne (SGD), même mécanique que
/// `ctr_predictor::CtrModel` — mais paramétrable par son prior, parce que les
/// taux de base des objectifs diffèrent de deux ordres de grandeur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Head {
    pub weights: [f64; N_FEATURES],
    pub bias: f64,
    pub learning_rate: f64,
    pub samples_seen: u64,
    pub positives: u64,
}

impl Head {
    fn with_prior(prior: f64) -> Self {
        Self {
            // Poids nuls, contrairement au CTR : on n'a AUCUN historique
            // calibré dimension par dimension pour « ce qui se relaie » ni
            // pour « ce qui se signale ». Inventer un prior par dimension
            // reviendrait à imposer une intuition non vérifiée au modèle ;
            // partir de zéro le laisse l'apprendre, et le biais porte seul
            // le taux de base.
            weights: [0.0; N_FEATURES],
            bias: logit(prior),
            learning_rate: 0.01,
            samples_seen: 0,
            positives: 0,
        }
    }

    pub fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        let z: f64 = self.bias
            + features
                .iter()
                .zip(self.weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f64>();
        sigmoid(z)
    }

    pub fn update(&mut self, features: &[f64; N_FEATURES], positive: bool) {
        let pred = self.predict(features);
        let error = if positive { 1.0 } else { 0.0 } - pred;

        // Décroissance en racine inverse (Robbins-Monro), comme les deux
        // autres modèles.
        let lr = self.learning_rate / (1.0 + 0.001 * self.samples_seen as f64).sqrt();

        self.bias += lr * BIAS_LR_MULTIPLIER * error;
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += lr * error * f;
            *w *= 1.0 - lr * 0.0001; // L2 légère, même valeur que le CTR
        }

        self.samples_seen += 1;
        if positive {
            self.positives += 1;
        }
    }

    /// Taux de base observé — pour le diagnostic admin, jamais pour la
    /// prédiction.
    pub fn base_rate(&self) -> f64 {
        if self.samples_seen == 0 {
            return 0.0;
        }
        self.positives as f64 / self.samples_seen as f64
    }

    /// Cette tête a-t-elle assez appris pour peser dans le classement ?
    pub fn is_ready(&self) -> bool {
        self.samples_seen >= MIN_SAMPLES
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Jeu de têtes partagé
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveModels {
    pub amplify: Head,
    pub reject: Head,
}

impl Default for ObjectiveModels {
    fn default() -> Self {
        Self {
            amplify: Head::with_prior(PRIOR_AMPLIFY),
            reject: Head::with_prior(PRIOR_REJECT),
        }
    }
}

/// Prédictions prêtes à entrer dans le classement. `None` = tête pas encore
/// mûre, le classement doit se comporter comme si elle n'existait pas.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectivePredictions {
    pub amplify: Option<f64>,
    pub reject: Option<f64>,
}

#[derive(Clone)]
pub struct ObjectivePredictor {
    models: Arc<RwLock<ObjectiveModels>>,
    /// Une fenêtre d'évaluation par tête : le rejet est rare par construction
    /// et apprend beaucoup plus lentement que l'amplification. Les mélanger
    /// masquerait exactement ce qu'on cherche à voir.
    eval_amplify: crate::eval::OnlineEval,
    eval_reject: crate::eval::OnlineEval,
}

impl ObjectivePredictor {
    pub fn new() -> Self {
        Self::from_models(ObjectiveModels::default())
    }

    fn from_models(models: ObjectiveModels) -> Self {
        Self {
            models: Arc::new(RwLock::new(models)),
            eval_amplify: crate::eval::OnlineEval::new(),
            eval_reject: crate::eval::OnlineEval::new(),
        }
    }

    /// Qualité mesurée de chaque tête — (amplification, rejet). Voir
    /// `crate::eval`.
    pub fn eval_reports(&self) -> (crate::eval::EvalReport, crate::eval::EvalReport) {
        (self.eval_amplify.report(), self.eval_reject.report())
    }

    pub async fn load_or_default() -> Self {
        if Path::new(MODEL_PATH).exists() {
            match fs::read_to_string(MODEL_PATH).await {
                Ok(json) => match serde_json::from_str::<ObjectiveModels>(&json) {
                    Ok(models) => {
                        info!(
                            amplify_samples = models.amplify.samples_seen,
                            amplify_rate = models.amplify.base_rate(),
                            reject_samples = models.reject.samples_seen,
                            reject_rate = models.reject.base_rate(),
                            "Têtes multi-objectifs chargées depuis le disque"
                        );
                        return Self::from_models(models);
                    }
                    Err(e) => warn!("Têtes multi-objectifs illisibles ({e}), repart des défauts"),
                },
                Err(e) => warn!("Têtes multi-objectifs non lues ({e}), repart des défauts"),
            }
        }
        info!("Têtes multi-objectifs neuves (aucune donnée d'entraînement)");
        Self::new()
    }

    /// Prédit les deux objectifs. Une tête pas encore mûre renvoie `None` —
    /// c'est au classement de décider quoi en faire, pas à elle de renvoyer
    /// une valeur neutre qui aurait l'air d'une prédiction.
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> ObjectivePredictions {
        let m = self.models.read().unwrap();
        ObjectivePredictions {
            amplify: m.amplify.is_ready().then(|| m.amplify.predict(features)),
            reject: m.reject.is_ready().then(|| m.reject.predict(features)),
        }
    }

    /// Entraîne les têtes concernées par cette interaction. Une tête dont
    /// l'étiquette est `None` n'est pas touchée — et surtout, son compteur
    /// d'échantillons n'avance pas : elle ne doit pas devenir « mûre » sur des
    /// interactions qui ne lui apprennent rien.
    pub fn record_interaction(
        &self,
        features: &[f64; N_FEATURES],
        interaction: InteractionType,
    ) -> bool {
        let amplify = label_for(interaction, Objective::Amplify);
        let reject = label_for(interaction, Objective::Reject);
        if amplify.is_none() && reject.is_none() {
            return false;
        }
        let mut m = self.models.write().unwrap();
        // Validation progressive : prédire AVANT d'apprendre — voir
        // `crate::eval`.
        let mut pending: Vec<(&crate::eval::OnlineEval, f64, f64)> = Vec::with_capacity(2);
        if let Some(positive) = amplify {
            let prior = m.amplify.predict(features);
            m.amplify.update(features, positive);
            pending.push((&self.eval_amplify, prior, positive as u8 as f64));
        }
        if let Some(positive) = reject {
            let prior = m.reject.predict(features);
            m.reject.update(features, positive);
            pending.push((&self.eval_reject, prior, positive as u8 as f64));
        }
        let stats = (
            m.amplify.samples_seen,
            m.amplify.base_rate(),
            m.reject.samples_seen,
            m.reject.base_rate(),
        );
        drop(m);
        for (window, prediction, truth) in pending {
            window.record(prediction, truth);
        }
        debug!(
            interaction = ?interaction,
            amplify_samples = stats.0, amplify_rate = stats.1,
            reject_samples = stats.2, reject_rate = stats.3,
            "Têtes multi-objectifs mises à jour"
        );
        true
    }

    /// Impression expirée sans la moindre réaction — voir `ml::ctr_sweeper`.
    ///
    /// C'est un négatif pour LES DEUX têtes, et de loin leur source principale
    /// d'exemples négatifs : un tweet montré que personne n'a relayé n'a pas
    /// été relayé, et un tweet montré que personne n'a signalé n'a pas été
    /// signalé.
    pub fn record_ignored(&self, features: &[f64; N_FEATURES]) {
        let mut m = self.models.write().unwrap();
        let prior_amplify = m.amplify.predict(features);
        let prior_reject = m.reject.predict(features);
        m.amplify.update(features, false);
        m.reject.update(features, false);
        drop(m);
        self.eval_amplify.record(prior_amplify, 0.0);
        self.eval_reject.record(prior_reject, 0.0);
    }

    /// (échantillons, taux de base) pour chaque tête, dans l'ordre
    /// (amplification, rejet).
    pub fn stats(&self) -> ((u64, f64), (u64, f64)) {
        let m = self.models.read().unwrap();
        (
            (m.amplify.samples_seen, m.amplify.base_rate()),
            (m.reject.samples_seen, m.reject.base_rate()),
        )
    }

    pub fn total_samples(&self) -> u64 {
        let m = self.models.read().unwrap();
        m.amplify.samples_seen.max(m.reject.samples_seen)
    }

    pub async fn save(&self) {
        let models = self.models.read().unwrap().clone();
        match serde_json::to_string_pretty(&models) {
            Ok(json) => {
                let _ = fs::create_dir_all("data").await;
                match fs::write(MODEL_PATH, json).await {
                    Ok(_) => info!(
                        amplify_samples = models.amplify.samples_seen,
                        reject_samples = models.reject.samples_seen,
                        "Têtes multi-objectifs persistées"
                    ),
                    Err(e) => warn!("Écriture des têtes multi-objectifs impossible : {e}"),
                }
            }
            Err(e) => warn!("Sérialisation des têtes multi-objectifs impossible : {e}"),
        }
    }
}

impl Default for ObjectivePredictor {
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
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::ctr_predictor::extract_features;

    fn features() -> [f64; N_FEATURES] {
        extract_features(
            0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3, 0.5, false, true, 5000, 0.5, 0.5,
        )
    }

    // ─── Étiquetage ─────────────────────────────────────────────────────────

    /// Le point de tout le module : un like et un retweet ne doivent PLUS
    /// produire la même étiquette. Sous la tête unique, les deux valaient
    /// `Some(true)` et le modèle ne pouvait pas les distinguer.
    #[test]
    fn un_like_et_un_retweet_ne_disent_pas_la_meme_chose() {
        assert_eq!(
            label_for(InteractionType::Retweet, Objective::Amplify),
            Some(true)
        );
        assert_eq!(
            label_for(InteractionType::Like, Objective::Amplify),
            Some(false),
            "vu, apprécié, PAS relayé — c'est le contre-exemple le plus utile"
        );
        // Alors qu'ils restent identiques pour la tête d'engagement.
        assert_eq!(InteractionType::Retweet.ctr_label(), Some(true));
        assert_eq!(InteractionType::Like.ctr_label(), Some(true));
    }

    /// Même chose côté négatif : un survol et un signalement valaient tous
    /// deux `Some(false)` pour la tête unique.
    #[test]
    fn un_survol_et_un_signalement_ne_disent_pas_la_meme_chose() {
        assert_eq!(
            label_for(InteractionType::Report, Objective::Reject),
            Some(true)
        );
        assert_eq!(
            label_for(InteractionType::Skip, Objective::Reject),
            None,
            "passer son chemin n'est pas déclarer qu'un contenu n'aurait pas dû être montré"
        );
        assert_eq!(InteractionType::Report.ctr_label(), Some(false));
        assert_eq!(InteractionType::Skip.ctr_label(), Some(false));
    }

    /// Une vue n'a jamais d'étiquette : elle ouvre la fenêtre d'attribution,
    /// c'est le balayage qui conclut. L'étiqueter ici, c'est le bug historique
    /// du CTR (« tout est un clic ») rejoué sur les nouvelles têtes.
    #[test]
    fn une_vue_ne_tranche_jamais() {
        assert_eq!(label_for(InteractionType::View, Objective::Amplify), None);
        assert_eq!(label_for(InteractionType::View, Objective::Reject), None);
    }

    /// Les trois refus explicites sont positifs pour la tête de rejet, et
    /// négatifs pour l'amplification : cohérence entre les deux lectures.
    #[test]
    fn les_refus_explicites_sont_coherents_entre_les_deux_tetes() {
        for refus in [
            InteractionType::Report,
            InteractionType::Block,
            InteractionType::NotInterested,
        ] {
            assert_eq!(label_for(refus, Objective::Reject), Some(true), "{refus:?}");
            assert_eq!(
                label_for(refus, Objective::Amplify),
                Some(false),
                "{refus:?}"
            );
        }
    }

    // ─── Apprentissage ──────────────────────────────────────────────────────

    #[test]
    fn le_biais_de_depart_predit_le_taux_de_base() {
        let m = ObjectiveModels::default();
        let zero = [0.0; N_FEATURES];
        assert!((m.amplify.predict(&zero) - PRIOR_AMPLIFY).abs() < 1e-6);
        assert!((m.reject.predict(&zero) - PRIOR_REJECT).abs() < 1e-6);
    }

    #[test]
    fn une_tete_apprend_dans_la_bonne_direction() {
        let f = features();
        let mut head = Head::with_prior(PRIOR_AMPLIFY);
        let avant = head.predict(&f);
        for _ in 0..300 {
            head.update(&f, true);
        }
        assert!(head.predict(&f) > avant);

        let mut head = Head::with_prior(0.5);
        let avant = head.predict(&f);
        for _ in 0..300 {
            head.update(&f, false);
        }
        assert!(head.predict(&f) < avant);
    }

    #[test]
    fn une_prediction_reste_une_probabilite() {
        let mut head = Head::with_prior(PRIOR_REJECT);
        for _ in 0..500 {
            head.update(&features(), true);
        }
        let p = head.predict(&features());
        assert!((0.0..=1.0).contains(&p), "hors [0,1] : {p}");
    }

    // ─── Démarrage à froid ──────────────────────────────────────────────────

    /// Une tête froide ne doit PAS peser : le classement doit se comporter
    /// exactement comme avant l'existence de ce module tant qu'elle n'a rien
    /// appris.
    #[test]
    fn une_tete_froide_ne_predit_rien() {
        let p = ObjectivePredictor::new();
        let pred = p.predict(&features());
        assert!(pred.amplify.is_none() && pred.reject.is_none());
    }

    #[test]
    fn une_tete_devient_mure_apres_le_seuil() {
        let p = ObjectivePredictor::new();
        let f = features();
        for _ in 0..MIN_SAMPLES {
            // Un retweet étiquette LES DEUX têtes (relais = oui, rejet = non).
            assert!(p.record_interaction(&f, InteractionType::Retweet));
        }
        let pred = p.predict(&f);
        assert!(pred.amplify.is_some() && pred.reject.is_some());
    }

    /// Une interaction qui n'apprend rien à une tête ne doit pas faire avancer
    /// son compteur — sinon elle deviendrait « mûre » sans avoir rien appris.
    #[test]
    fn une_interaction_muette_ne_fait_pas_murir_les_tetes() {
        let p = ObjectivePredictor::new();
        let f = features();
        for _ in 0..(MIN_SAMPLES * 2) {
            assert!(
                !p.record_interaction(&f, InteractionType::View),
                "une vue ne doit entraîner aucune tête"
            );
        }
        let ((amplify_n, _), (reject_n, _)) = p.stats();
        assert_eq!(amplify_n, 0);
        assert_eq!(reject_n, 0);
        assert!(p.predict(&f).amplify.is_none());
    }

    /// `Unlike` n'apprend rien à la tête d'amplification (elle parle du like)
    /// et rien à la tête de rejet non plus : le compteur de la seule tête
    /// concernée doit avancer, pas celui de l'autre.
    #[test]
    fn seule_la_tete_concernee_avance() {
        let p = ObjectivePredictor::new();
        let f = features();
        // `Skip` : négatif pour l'amplification, muet pour le rejet.
        for _ in 0..50 {
            p.record_interaction(&f, InteractionType::Skip);
        }
        let ((amplify_n, _), (reject_n, _)) = p.stats();
        assert_eq!(amplify_n, 50);
        assert_eq!(reject_n, 0, "un survol n'apprend rien au rejet");
    }

    /// Le balayage est la source principale des négatifs : il doit nourrir les
    /// deux têtes à la fois.
    #[test]
    fn une_impression_ignoree_est_negative_pour_les_deux_tetes() {
        let p = ObjectivePredictor::new();
        for _ in 0..10 {
            p.record_ignored(&features());
        }
        let ((amplify_n, amplify_rate), (reject_n, reject_rate)) = p.stats();
        assert_eq!(amplify_n, 10);
        assert_eq!(reject_n, 10);
        assert_eq!(amplify_rate, 0.0);
        assert_eq!(reject_rate, 0.0);
    }
}
