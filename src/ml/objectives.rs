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
/// Le j'aime est de loin le geste positif le plus courant.
const PRIOR_FAV: f64 = 0.030;
/// Répondre coûte d'écrire : c'est rare, et c'est ce qui en fait un signal fort.
const PRIOR_REPLY: f64 = 0.004;
/// Aller voir le profil de l'auteur — entre les deux.
const PRIOR_PROFILE: f64 = 0.012;

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
    /// Le lecteur va-t-il AIMER ce tweet ?
    Fav,
    /// Le lecteur va-t-il RÉPONDRE à ce tweet ?
    Reply,
    /// Le lecteur va-t-il ouvrir le PROFIL de l'auteur ?
    Profile,
}

impl Objective {
    /// Toutes les têtes, dans un ordre stable — c'est cet ordre qui indexe les
    /// fenêtres d'évaluation, donc il ne doit jamais être réarrangé sans
    /// réinitialiser les fenêtres.
    pub const ALL: [Objective; 5] = [
        Objective::Amplify,
        Objective::Reject,
        Objective::Fav,
        Objective::Reply,
        Objective::Profile,
    ];

    pub const fn index(self) -> usize {
        match self {
            Objective::Amplify => 0,
            Objective::Reject => 1,
            Objective::Fav => 2,
            Objective::Reply => 3,
            Objective::Profile => 4,
        }
    }
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
            // Ouvrir, c'est consommer — pas relayer. Comme le like, c'est un
            // contre-exemple utile : engagement avéré SANS amplification.
            I::Open
            | I::Like
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
            I::Open
            | I::Like
            | I::Comment
            | I::Retweet
            | I::Share
            | I::Bookmark
            | I::Interested
            | I::ProfileView => Some(false),
            I::View | I::Skip | I::Unlike | I::Unretweet => None,
        },

        // ── Les trois têtes par ÉVÉNEMENT ────────────────────────────────
        //
        // Règle commune, et elle est prudente exprès : seul l'événement
        // lui-même est un positif, seuls les refus explicites sont des
        // négatifs, tout le reste est `None`.
        //
        // Pourquoi ne pas compter un retweet comme « pas un j'aime » : les
        // interactions nous arrivent UNE PAR UNE. Un lecteur qui aime ET
        // retweete produit deux événements ; étiqueter le retweet
        // « n'a pas aimé » serait faux une fois sur deux. X n'a pas ce
        // problème — il entraîne sur l'issue complète d'une impression.
        //
        // Le gros des négatifs vient donc d'ailleurs, et c'est très bien :
        // `record_ignored` (impression expirée sans réaction) en fournit des
        // dizaines pour un positif. C'est la même économie que le CTR.
        Objective::Fav => match interaction {
            I::Like => Some(true),
            I::Report | I::Block | I::NotInterested | I::Skip => Some(false),
            // Défaire un like dit que le like était regretté : c'est le
            // négatif le plus informatif qui existe pour cette tête.
            I::Unlike => Some(false),
            _ => None,
        },
        Objective::Reply => match interaction {
            I::Comment => Some(true),
            I::Report | I::Block | I::NotInterested | I::Skip => Some(false),
            _ => None,
        },
        Objective::Profile => match interaction {
            I::ProfileView => Some(true),
            I::Report | I::Block | I::NotInterested | I::Skip => Some(false),
            _ => None,
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
    /// Moyenne courante des prédictions de cette tête.
    ///
    /// Ce n'est pas du diagnostic : c'est le dénominateur de `lift`, et donc
    /// une pièce du classement. Voir `lift` pour pourquoi.
    ///
    /// `serde(default)` : les modèles déjà persistés en production n'ont pas
    /// ce champ. Ils repartent de 0,0 et le champ se remplit au premier
    /// échantillon — `lift` retombe entre-temps sur le prior de la tête.
    #[serde(default)]
    pub pred_mean: f64,
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
            pred_mean: prior,
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
        // Moyenne courante des prédictions, entretenue ici et pas dans
        // `predict` : `predict` ne prend qu'un verrou de LECTURE et tourne sur
        // chaque candidat de chaque fil, des milliers de fois par seconde. La
        // mettre à jour là demanderait un verrou d'écriture au classement.
        let n = self.samples_seen as f64;
        self.pred_mean += (pred - self.pred_mean) / n;
    }

    /// Prédiction ramenée sur une échelle COMMUNE à toutes les têtes.
    ///
    /// ── Le défaut que ça corrige ────────────────────────────────────────
    /// `blend_positive` fait une moyenne PONDÉRÉE de valeurs dans [0,1]. Or
    /// les têtes ne vivent pas du tout sur la même plage : le score de règles
    /// balaie 0,2 à 0,8, tandis qu'une probabilité de réponse balaie 0,002 à
    /// 0,01. Injectée telle quelle, une tête rare n'apporte quasiment aucune
    /// variance au classement — elle abaisse tous les scores d'à peu près la
    /// même quantité, ce qui ne change aucun ordre. Le mélange annoncé
    /// « moitié règles / moitié modèles » était donc, en pratique, presque
    /// entièrement piloté par les règles.
    ///
    /// ── Le principe repris ──────────────────────────────────────────────
    /// X documente que les poids de son classeur lourd ont été choisis pour
    /// que « chaque probabilité d'engagement pondérée contribue en moyenne à
    /// peu près autant au score ». Chez eux c'est une somme, et le réglage
    /// tient dans les poids. Chez nous c'est une MOYENNE pondérée, donc le
    /// réglage doit être dans la valeur : on divise par la moyenne courante de
    /// la tête pour obtenir un `lift` (« combien de fois plus probable que
    /// d'habitude »), puis on l'écrase dans [0,1] par `l / (l + 1)`.
    ///
    /// Cette transformation envoie la moyenne sur 0,5 exactement, quelle que
    /// soit la rareté de l'événement : deux fois la moyenne donne 0,667, la
    /// moitié donne 0,333. Toutes les têtes contribuent alors la même plage,
    /// et les poids de mélange redeviennent ce qu'ils prétendent être.
    pub fn lift(&self, features: &[f64; N_FEATURES]) -> f64 {
        let p = self.predict(features);
        // Plancher : une tête dont la moyenne est tombée à zéro (aucun positif
        // encore vu) donnerait une division explosive.
        let mean = self.pred_mean.max(1e-4);
        let l = p / mean;
        l / (l + 1.0)
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
    /// Les trois tetes par evenement, ajoutees apres coup.
    ///
    /// `serde(default = ...)` sur chacune : le fichier deja persiste en
    /// production ne contient que `amplify` et `reject`. Sans ces defauts, tout
    /// le jeu redeviendrait illisible et les deux tetes deja entrainees
    /// repartiraient de zero en silence — exactement le piege que la migration
    /// du modele de dwell existe pour eviter.
    #[serde(default = "head_fav")]
    pub fav: Head,
    #[serde(default = "head_reply")]
    pub reply: Head,
    #[serde(default = "head_profile")]
    pub profile: Head,
}

fn head_fav() -> Head {
    Head::with_prior(PRIOR_FAV)
}
fn head_reply() -> Head {
    Head::with_prior(PRIOR_REPLY)
}
fn head_profile() -> Head {
    Head::with_prior(PRIOR_PROFILE)
}

impl Default for ObjectiveModels {
    fn default() -> Self {
        Self {
            amplify: Head::with_prior(PRIOR_AMPLIFY),
            reject: Head::with_prior(PRIOR_REJECT),
            fav: head_fav(),
            reply: head_reply(),
            profile: head_profile(),
        }
    }
}

impl ObjectiveModels {
    pub fn head(&self, objective: Objective) -> &Head {
        match objective {
            Objective::Amplify => &self.amplify,
            Objective::Reject => &self.reject,
            Objective::Fav => &self.fav,
            Objective::Reply => &self.reply,
            Objective::Profile => &self.profile,
        }
    }

    fn head_mut(&mut self, objective: Objective) -> &mut Head {
        match objective {
            Objective::Amplify => &mut self.amplify,
            Objective::Reject => &mut self.reject,
            Objective::Fav => &mut self.fav,
            Objective::Reply => &mut self.reply,
            Objective::Profile => &mut self.profile,
        }
    }
}

/// Predictions pretes a entrer dans le classement. `None` = tete pas encore
/// mure, le classement doit se comporter comme si elle n'existait pas.
///
/// ⚠ **Deux echelles differentes, et c'est voulu.** Les quatre champs positifs
/// portent un `lift` (voir `Head::lift`) : une valeur centree sur 0,5, faite
/// pour entrer dans une moyenne ponderee aux cotes du score de regles. Le champ
/// `reject_p`, lui, porte une VRAIE probabilite, parce qu'il entre en
/// multiplicateur (`1 − k·p`) et qu'un lift y vaudrait 0,5 pour un tweet
/// parfaitement ordinaire — ce qui penaliserait tout le corpus de 30 %.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectivePredictions {
    pub amplify_lift: Option<f64>,
    pub fav_lift: Option<f64>,
    pub reply_lift: Option<f64>,
    pub profile_lift: Option<f64>,
    pub reject_p: Option<f64>,
}

#[derive(Clone)]
pub struct ObjectivePredictor {
    models: Arc<RwLock<ObjectiveModels>>,
    /// Une fenetre d'evaluation par tete, indexee par `Objective::index()` : le
    /// rejet est rare par construction et apprend beaucoup plus lentement que
    /// l'amplification. Les melanger masquerait ce qu'on cherche a voir.
    evals: [crate::eval::OnlineEval; 5],
}

impl ObjectivePredictor {
    pub fn new() -> Self {
        Self::from_models(ObjectiveModels::default())
    }

    fn from_models(models: ObjectiveModels) -> Self {
        Self {
            models: Arc::new(RwLock::new(models)),
            evals: std::array::from_fn(|_| crate::eval::OnlineEval::new()),
        }
    }

    fn eval(&self, objective: Objective) -> &crate::eval::OnlineEval {
        &self.evals[objective.index()]
    }

    /// Qualite mesuree de chaque tete — (amplification, rejet). Voir
    /// `crate::eval`. Les trois tetes par evenement se lisent par
    /// `eval_report_of`.
    pub fn eval_reports(&self) -> (crate::eval::EvalReport, crate::eval::EvalReport) {
        (
            self.eval(Objective::Amplify).report(),
            self.eval(Objective::Reject).report(),
        )
    }

    pub fn eval_report_of(&self, objective: Objective) -> crate::eval::EvalReport {
        self.eval(objective).report()
    }

    /// Ce qu'une recalibration rattraperait sur chaque tete — (amplification,
    /// rejet). Mesure seulement. Voir `crate::ml::calibrator`.
    pub fn calibration_gains(
        &self,
    ) -> (Option<crate::ml::CalibrationGain>, Option<crate::ml::CalibrationGain>) {
        (
            self.eval(Objective::Amplify).calibration_gain(),
            self.eval(Objective::Reject).calibration_gain(),
        )
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
                            fav_samples = models.fav.samples_seen,
                            reply_samples = models.reply.samples_seen,
                            profile_samples = models.profile.samples_seen,
                            "Tetes multi-objectifs chargees depuis le disque"
                        );
                        return Self::from_models(models);
                    }
                    // Un vecteur de poids plus COURT que `N_FEATURES` vient
                    // d'un schema enrichi depuis la sauvegarde, pas d'un
                    // fichier corrompu. Sans cette branche, elargir le vecteur
                    // de traits remettrait a zero, EN SILENCE, toutes les tetes
                    // deja entrainees en production — le CTR et le dwell ont
                    // chacun leur migration depuis longtemps, celle-ci
                    // n'existait pas.
                    Err(e) => match migrate_legacy_models(&json) {
                        Some(models) => {
                            info!(
                                amplify_samples = models.amplify.samples_seen,
                                reject_samples = models.reject.samples_seen,
                                "Tetes multi-objectifs migrees depuis un vecteur plus court — \
                                 poids appris conserves, nouveaux traits au defaut"
                            );
                            return Self::from_models(models);
                        }
                        None => warn!("Tetes multi-objectifs illisibles ({e}), repart des defauts"),
                    },
                },
                Err(e) => warn!("Tetes multi-objectifs non lues ({e}), repart des defauts"),
            }
        }
        info!("Tetes multi-objectifs neuves (aucune donnee d'entrainement)");
        Self::new()
    }

    /// Predit tous les objectifs. Une tete pas encore mure renvoie `None` —
    /// c'est au classement de decider quoi en faire, pas a elle de renvoyer une
    /// valeur neutre qui aurait l'air d'une prediction.
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> ObjectivePredictions {
        let m = self.models.read().unwrap();
        let lift = |o: Objective| {
            let h = m.head(o);
            h.is_ready().then(|| h.lift(features))
        };
        ObjectivePredictions {
            amplify_lift: lift(Objective::Amplify),
            fav_lift: lift(Objective::Fav),
            reply_lift: lift(Objective::Reply),
            profile_lift: lift(Objective::Profile),
            // Probabilite brute, pas un lift — voir `ObjectivePredictions`.
            reject_p: m.reject.is_ready().then(|| m.reject.predict(features)),
        }
    }

    /// Entraine les tetes concernees par cette interaction. Une tete dont
    /// l'etiquette est `None` n'est pas touchee — et surtout, son compteur
    /// d'echantillons n'avance pas : elle ne doit pas devenir « mure » sur des
    /// interactions qui ne lui apprennent rien.
    pub fn record_interaction(
        &self,
        features: &[f64; N_FEATURES],
        interaction: InteractionType,
    ) -> bool {
        let labels: Vec<(Objective, bool)> = Objective::ALL
            .iter()
            .filter_map(|&o| label_for(interaction, o).map(|l| (o, l)))
            .collect();
        if labels.is_empty() {
            return false;
        }

        let mut m = self.models.write().unwrap();
        // Validation progressive : predire AVANT d'apprendre — voir
        // `crate::eval`. La prediction enregistree est la PROBABILITE, pas le
        // lift : une AUC se mesure sur ce que la tete predit, pas sur la
        // transformation qu'on lui applique ensuite pour le classement.
        let mut pending: Vec<(Objective, f64, f64)> = Vec::with_capacity(labels.len());
        for (objective, positive) in labels {
            let head = m.head_mut(objective);
            let prior = head.predict(features);
            head.update(features, positive);
            pending.push((objective, prior, positive as u8 as f64));
        }
        let stats = (
            m.amplify.samples_seen,
            m.amplify.base_rate(),
            m.reject.samples_seen,
            m.reject.base_rate(),
        );
        drop(m);

        for (objective, prediction, truth) in pending {
            self.eval(objective).record(prediction, truth);
        }
        debug!(
            interaction = ?interaction,
            amplify_samples = stats.0, amplify_rate = stats.1,
            reject_samples = stats.2, reject_rate = stats.3,
            "Tetes multi-objectifs mises a jour"
        );
        true
    }

    /// Impression expiree sans la moindre reaction — voir `ml::ctr_sweeper`.
    ///
    /// C'est un negatif pour TOUTES les tetes, et de loin leur source
    /// principale d'exemples negatifs : un tweet montre que personne n'a relaye
    /// n'a pas ete relaye, un tweet montre que personne n'a aime n'a pas ete
    /// aime, et ainsi de suite. C'est ce qui rend les tetes rares (reponse,
    /// visite de profil) entrainables du tout : leurs positifs se comptent en
    /// dizaines, leurs negatifs en dizaines de milliers.
    pub fn record_ignored(&self, features: &[f64; N_FEATURES]) {
        let mut m = self.models.write().unwrap();
        let mut pending: Vec<(Objective, f64)> = Vec::with_capacity(Objective::ALL.len());
        for &objective in Objective::ALL.iter() {
            let head = m.head_mut(objective);
            let prior = head.predict(features);
            head.update(features, false);
            pending.push((objective, prior));
        }
        drop(m);
        for (objective, prediction) in pending {
            self.eval(objective).record(prediction, 0.0);
        }
    }

    /// (echantillons, taux de base) pour chaque tete, dans l'ordre
    /// (amplification, rejet).
    pub fn stats(&self) -> ((u64, f64), (u64, f64)) {
        let m = self.models.read().unwrap();
        (
            (m.amplify.samples_seen, m.amplify.base_rate()),
            (m.reject.samples_seen, m.reject.base_rate()),
        )
    }

    /// (echantillons, taux de base, mure ?) pour chacune des cinq tetes, dans
    /// l'ordre de `Objective::ALL`.
    pub fn stats_all(&self) -> [(u64, f64, bool); 5] {
        let m = self.models.read().unwrap();
        std::array::from_fn(|i| {
            let h = m.head(Objective::ALL[i]);
            (h.samples_seen, h.base_rate(), h.is_ready())
        })
    }

    pub fn total_samples(&self) -> u64 {
        let m = self.models.read().unwrap();
        Objective::ALL
            .iter()
            .map(|&o| m.head(o).samples_seen)
            .max()
            .unwrap_or(0)
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
                        fav_samples = models.fav.samples_seen,
                        reply_samples = models.reply.samples_seen,
                        profile_samples = models.profile.samples_seen,
                        "Tetes multi-objectifs persistees"
                    ),
                    Err(e) => warn!("Ecriture des tetes multi-objectifs impossible : {e}"),
                }
            }
            Err(e) => warn!("Serialisation des tetes multi-objectifs impossible : {e}"),
        }
    }
}

impl Default for ObjectivePredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete les tetes d'un fichier dont les vecteurs `weights` sont plus courts
/// que `N_FEATURES`.
///
/// Meme regle que `ctr_predictor::migrate_legacy_weights` : un tableau plus
/// LONG reste un modele neuf, il n'y a pas de facon honnete de deviner quel
/// trait a disparu. Les tetes absentes du fichier (les trois tetes par
/// evenement, sur une sauvegarde anterieure) sont laissees a serde, qui les
/// remplira par leurs `#[serde(default)]`.
fn migrate_legacy_models(json: &str) -> Option<ObjectiveModels> {
    let mut value: serde_json::Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object_mut()?;
    let defaults = ObjectiveModels::default();
    let mut touched = false;

    for (name, fallback) in [
        ("amplify", &defaults.amplify),
        ("reject", &defaults.reject),
        ("fav", &defaults.fav),
        ("reply", &defaults.reply),
        ("profile", &defaults.profile),
    ] {
        let Some(head) = obj.get_mut(name).and_then(|h| h.as_object_mut()) else {
            continue;
        };
        let Some(old) = head.get("weights").and_then(|w| w.as_array()).cloned() else {
            continue;
        };
        if old.is_empty() || old.len() >= N_FEATURES {
            continue;
        }
        let mut padded = old;
        for w in fallback.weights.iter().skip(padded.len()) {
            padded.push(serde_json::json!(w));
        }
        head.insert("weights".to_string(), serde_json::Value::Array(padded));
        touched = true;
    }

    if !touched {
        return None;
    }
    serde_json::from_value(value).ok()
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
            0.5,
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
        assert!(pred.amplify_lift.is_none() && pred.reject_p.is_none());
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
        assert!(pred.amplify_lift.is_some() && pred.reject_p.is_some());
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
        assert!(p.predict(&f).amplify_lift.is_none());
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

    // ─── Migration ──────────────────────────────────────────────────────────

    /// Le piege qui aurait coute des dizaines de milliers d'echantillons :
    /// elargir le vecteur de traits ne doit PAS remettre a zero les tetes deja
    /// entrainees en production.
    #[test]
    fn un_vecteur_plus_court_est_migre_sans_perdre_l_appris() {
        // Un fichier tel qu'il existait AVANT les croisements : deux tetes, des
        // vecteurs de 16 poids, et 5 000 echantillons chacune.
        let seize: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
        let ancien = serde_json::json!({
            "amplify": {
                "weights": seize, "bias": -1.5, "learning_rate": 0.01,
                "samples_seen": 5000, "positives": 100
            },
            "reject": {
                "weights": seize, "bias": -3.0, "learning_rate": 0.01,
                "samples_seen": 5000, "positives": 12
            }
        })
        .to_string();

        let migre = migrate_legacy_models(&ancien).expect("la migration doit aboutir");
        assert_eq!(migre.amplify.samples_seen, 5000, "l'appris est conserve");
        assert_eq!(migre.amplify.weights.len(), N_FEATURES);
        // Les 16 premiers poids sont ceux du fichier.
        for i in 0..16 {
            assert!((migre.amplify.weights[i] - i as f64 * 0.01).abs() < 1e-12);
        }
        // Les tetes absentes du fichier arrivent neuves, pas corrompues.
        assert_eq!(migre.fav.samples_seen, 0);
        assert!(!migre.fav.is_ready());
    }

    /// Un vecteur DEJA a la bonne taille n'a rien a migrer : la fonction doit
    /// rendre `None` pour que le chemin normal de serde s'applique.
    #[test]
    fn un_vecteur_a_jour_n_est_pas_migre() {
        let json = serde_json::to_string(&ObjectiveModels::default()).unwrap();
        assert!(migrate_legacy_models(&json).is_none());
    }

    // ─── Normalisation en lift ──────────────────────────────────────────────

    /// Le coeur de la correction, teste sur son INVARIANT plutot que par un
    /// detour d'entrainement : `lift` ne depend que du rapport entre la
    /// prediction et la moyenne de la tete. Deux tetes dont la raretee differe
    /// de deux ordres de grandeur, mais qui predisent chacune le double de leur
    /// propre moyenne, doivent rendre EXACTEMENT la meme valeur.
    ///
    /// C'est ce qui rend les poids de melange honnetes : avant, la tete rare
    /// entrait dans la moyenne ponderee avec une plage cent fois plus etroite
    /// que sa voisine, et son poids nominal ne voulait rien dire.
    #[test]
    fn deux_tetes_de_raretes_differentes_rendent_le_meme_lift() {
        let f = [0.0; N_FEATURES];

        // Tete frequente : moyenne 0,30, et on la fait predire 0,60.
        let mut frequente = Head::with_prior(0.60);
        frequente.pred_mean = 0.30;
        // Tete rare : moyenne 0,003, et on la fait predire 0,006.
        let mut rare = Head::with_prior(0.006);
        rare.pred_mean = 0.003;

        let l_frequente = frequente.lift(&f);
        let l_rare = rare.lift(&f);
        assert!(
            (l_frequente - l_rare).abs() < 1e-9,
            "meme rapport, donc meme lift : {l_frequente} vs {l_rare}"
        );
        // Le double de la moyenne tombe sur 2/3, par construction.
        assert!((l_frequente - 2.0 / 3.0).abs() < 1e-9, "{l_frequente}");

        // Alors que sur les probabilites brutes, les deux sont incomparables.
        let ecart_brut = (frequente.predict(&f) - rare.predict(&f)).abs();
        assert!(ecart_brut > 0.5, "ecart brut attendu enorme : {ecart_brut}");
    }

    /// Une tete qui predit exactement sa propre moyenne est NEUTRE : elle ne
    /// deplace pas le score de regles dans le melange.
    #[test]
    fn une_tete_a_sa_moyenne_vaut_un_demi() {
        let f = [0.0; N_FEATURES];
        for prior in [0.5, 0.05, 0.002] {
            let mut head = Head::with_prior(prior);
            head.pred_mean = prior;
            assert!(
                (head.lift(&f) - 0.5).abs() < 1e-9,
                "prior={prior} lift={}",
                head.lift(&f)
            );
        }
    }

    /// Un lift reste borne, quelle que soit la moyenne de la tete.
    #[test]
    fn un_lift_reste_dans_zero_un() {
        let f = features();
        for prior in [0.5, 0.05, 0.001] {
            let mut head = Head::with_prior(prior);
            for _ in 0..200 {
                head.update(&f, true);
            }
            let l = head.lift(&f);
            assert!((0.0..=1.0).contains(&l), "prior={prior} lift={l}");
        }
    }

    // ─── Les trois tetes par evenement ──────────────────────────────────────

    /// Un j'aime n'est plus la meme chose qu'une reponse — c'est tout l'objet
    /// des tetes par evenement.
    #[test]
    fn chaque_geste_entraine_sa_propre_tete() {
        assert_eq!(label_for(InteractionType::Like, Objective::Fav), Some(true));
        assert_eq!(label_for(InteractionType::Like, Objective::Reply), None);
        assert_eq!(
            label_for(InteractionType::Comment, Objective::Reply),
            Some(true)
        );
        assert_eq!(
            label_for(InteractionType::ProfileView, Objective::Profile),
            Some(true)
        );
    }

    /// Un retweet ne doit PAS etre etiquete « n'a pas aime » : les interactions
    /// arrivent une par une, et un lecteur qui aime ET retweete en produit
    /// deux. C'est le piege le plus facile a poser sur des tetes par evenement.
    #[test]
    fn un_retweet_ne_dit_rien_du_jaime() {
        assert_eq!(
            label_for(InteractionType::Retweet, Objective::Fav),
            None,
            "aimer et retweeter ne s'excluent pas — l'ignorance est la bonne reponse"
        );
        assert_eq!(label_for(InteractionType::Comment, Objective::Fav), None);
    }

    /// Defaire un like est le negatif le plus informatif de la tete `Fav`.
    #[test]
    fn defaire_un_like_est_un_negatif_du_jaime() {
        assert_eq!(label_for(InteractionType::Unlike, Objective::Fav), Some(false));
    }

    /// Une impression ignoree nourrit les CINQ tetes — c'est ce qui rend les
    /// tetes rares entrainables du tout.
    #[test]
    fn une_impression_ignoree_nourrit_les_cinq_tetes() {
        let p = ObjectivePredictor::new();
        for _ in 0..10 {
            p.record_ignored(&features());
        }
        for (i, (samples, rate, _)) in p.stats_all().iter().enumerate() {
            assert_eq!(*samples, 10, "tete {i}");
            assert_eq!(*rate, 0.0, "tete {i}");
        }
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
