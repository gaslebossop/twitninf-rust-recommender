//! Mesure de la qualité des modèles — AUC, log-loss, calibration, NDCG.
//!
//! ── Pourquoi ce fichier existe ──────────────────────────────────────────────
//! Le moteur portait trois modèles entraînés en ligne (CTR, temps de lecture,
//! et maintenant les têtes multi-objectifs), un auto-tuner qui réécrit les
//! poids de scoring à partir de ce qu'ils ont appris, et un bandit — et AUCUNE
//! mesure de leur qualité. Les seuls indicateurs exposés étaient le nombre
//! d'échantillons vus et le taux de base observé. Ni l'un ni l'autre ne dit si
//! un modèle prédit quoi que ce soit d'utile : un modèle qui renvoie la même
//! valeur pour tous les tweets affiche exactement les mêmes deux chiffres
//! qu'un modèle parfait.
//!
//! Sans mesure, aucune modification du classement n'est démontrable. On peut
//! croire qu'un réglage a amélioré le fil, on ne peut pas le savoir — et
//! l'auto-tuner, lui, propage ses poids appris à tout le monde sans que rien
//! ne vérifie qu'ils valent mieux que les précédents.
//!
//! ── La méthode : validation progressive (prequential) ───────────────────────
//! Un modèle entraîné EN LIGNE ne peut pas être évalué par un découpage
//! apprentissage/test classique : il n'y a pas de jeu figé, les données
//! arrivent une par une et le modèle change à chaque fois. Le protocole
//! standard dans ce cas est la validation progressive : pour CHAQUE
//! échantillon, on prédit d'abord, on enregistre le couple (prédiction,
//! vérité), et seulement ensuite on met à jour le modèle.
//!
//! La prédiction enregistrée est donc toujours faite sur un exemple que le
//! modèle n'a jamais vu. C'est de la mesure hors-échantillon honnête, sans
//! avoir à mettre des données de côté — ce qui compte quand le corpus est
//! petit et qu'on ne peut pas se permettre d'en sacrifier une part.
//!
//! On garde une fenêtre GLISSANTE des derniers couples plutôt que tout
//! l'historique : un modèle en ligne dérive volontairement, et une moyenne
//! depuis le premier jour finirait par ne plus décrire ce qu'il fait
//! aujourd'hui.
//!
//! ⚠ **La limite du protocole, à connaître avant de lire un chiffre.** La
//! validation progressive suppose que l'ORDRE D'ARRIVÉE des échantillons n'est
//! pas corrélé à leur étiquette. Quand il l'est, un apprenant en ligne oscille
//! en phase avec la séquence : chaque prédiction, prise avant la mise à jour,
//! hérite du biais déplacé par l'échantillon précédent, donc elle est haute
//! juste avant un négatif et basse juste avant un positif. Mesuré ici sur du
//! bruit pur en alternance stricte positif/négatif : l'AUC tombe à 0,06 au lieu
//! de 0,50 — une anti-corrélation parfaite, entièrement fabriquée par l'ordre.
//!
//! Le trafic réel n'a pas cette structure (les interactions arrivent dans un
//! ordre qui ne dépend pas de leur issue). Mais si l'AUC d'un modèle plongeait
//! nettement SOUS 0,5, c'est la première chose à suspecter — un signal inversé
//! est rare, un ordre d'arrivée structuré ne l'est pas. Le balayage
//! d'attribution (`ml::ctr_sweeper`) est le candidat à surveiller : il produit
//! ses négatifs par paquets de 500, toutes les 60 secondes.
//!
//! ── Ce que chaque métrique répond ───────────────────────────────────────────
//! * **AUC** — « le modèle sait-il ORDONNER ? ». 0,5 = tirage à pile ou face,
//!   1,0 = ordre parfait. C'est la seule qui compte vraiment pour un
//!   classement : elle ne dépend d'aucun seuil et ne bouge pas si toutes les
//!   probabilités sont décalées d'un même facteur.
//! * **Log-loss** — « le modèle est-il sûr à bon escient ? ». Punit la
//!   confiance mal placée, contrairement à l'AUC.
//! * **ECE** (erreur de calibration attendue) — « quand le modèle dit 30 %,
//!   est-ce que ça arrive 30 % du temps ? ». C'est ce que le dépôt n'avait
//!   nulle part : `calibration.rs` ne calibre pas des probabilités, c'est le
//!   parcours de recalibration des goûts vu par l'utilisateur. Un modèle mal
//!   calibré reste utilisable pour ordonner, mais toute somme pondérée qui le
//!   mélange à d'autres signaux (c'est exactement ce que fait le classement)
//!   lui donne un poids faussé.
//! * **NDCG@k** — « les meilleurs items sont-ils EN HAUT ? ». Contrairement à
//!   l'AUC, elle pèse les positions : se tromper à la place 2 coûte plus cher
//!   qu'à la place 40.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use serde::Serialize;

/// Taille de la fenêtre glissante d'évaluation.
///
/// Assez grand pour que l'AUC soit stable (quelques milliers de couples),
/// assez petit pour rester représentatif du modèle ACTUEL et pour que le
/// coût mémoire reste négligeable (~80 Ko par fenêtre).
pub const WINDOW: usize = 5_000;

/// Nombre de tranches de la courbe de fiabilité.
const RELIABILITY_BINS: usize = 10;

/// En dessous, aucune métrique n'est publiée : sur trente couples, une AUC ne
/// mesure que le hasard. Mieux vaut ne rien afficher qu'un chiffre qui invite
/// à conclure.
pub const MIN_SAMPLES_FOR_METRICS: usize = 200;

// ═══════════════════════════════════════════════════════════════════════════
// Métriques — fonctions pures
// ═══════════════════════════════════════════════════════════════════════════

/// Aire sous la courbe ROC, calculée par les RANGS (statistique de
/// Mann-Whitney), pas par intégration d'une courbe.
///
/// Le calcul par rangs traite correctement les ex æquo : deux items de même
/// score se partagent le rang moyen, ce qui compte l'égalité pour une
/// demi-victoire. C'est décisif ici, parce qu'un modèle qui n'a rien appris
/// renvoie exactement la même probabilité pour tout le monde — un calcul naïf
/// lui donnerait 0,0 ou 1,0 selon l'ordre d'arrivée, alors que la bonne
/// réponse est 0,5.
///
/// `None` quand une des deux classes est absente : l'AUC n'est pas définie.
pub fn roc_auc(samples: &[(f64, bool)]) -> Option<f64> {
    let positives = samples.iter().filter(|(_, label)| *label).count();
    let negatives = samples.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }

    let mut ordered: Vec<(f64, bool)> = samples.to_vec();
    ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Rangs moyens sur les paquets d'ex æquo.
    let mut rank_sum_positives = 0.0_f64;
    let mut i = 0usize;
    while i < ordered.len() {
        let mut j = i;
        while j + 1 < ordered.len() && ordered[j + 1].0 == ordered[i].0 {
            j += 1;
        }
        // Rangs 1-indexés de i..=j, donc rang moyen = (i+1 + j+1) / 2.
        let average_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for (_, label) in &ordered[i..=j] {
            if *label {
                rank_sum_positives += average_rank;
            }
        }
        i = j + 1;
    }

    let p = positives as f64;
    let n = negatives as f64;
    let u = rank_sum_positives - p * (p + 1.0) / 2.0;
    Some((u / (p * n)).clamp(0.0, 1.0))
}

/// Log-loss binaire (entropie croisée) moyenne.
///
/// Les probabilités sont bornées loin de 0 et de 1 : sans ça, une seule
/// prédiction certaine et fausse rend la moyenne infinie et emporte toute
/// l'information des autres.
pub fn log_loss(samples: &[(f64, bool)]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    const EPS: f64 = 1e-9;
    let total: f64 = samples
        .iter()
        .map(|(p, label)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            if *label {
                -p.ln()
            } else {
                -(1.0 - p).ln()
            }
        })
        .sum();
    Some(total / samples.len() as f64)
}

/// Une tranche de la courbe de fiabilité : ce que le modèle a annoncé, et ce
/// qui est réellement arrivé.
#[derive(Debug, Clone, Serialize)]
pub struct ReliabilityBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    /// Probabilité moyenne annoncée par le modèle dans cette tranche.
    pub predicted: f64,
    /// Fréquence réellement observée.
    pub observed: f64,
}

/// Courbe de fiabilité : les prédictions rangées par tranches, avec la
/// fréquence réellement observée dans chacune. C'est le diagnostic brut dont
/// l'ECE n'est que le résumé en un chiffre — il dit COMBIEN on se trompe,
/// elle dit OÙ.
pub fn reliability_curve(samples: &[(f64, bool)], bins: usize) -> Vec<ReliabilityBin> {
    let bins = bins.max(1);
    let mut acc = vec![(0usize, 0.0_f64, 0usize); bins];
    for (p, label) in samples {
        let idx = ((p.clamp(0.0, 1.0) * bins as f64) as usize).min(bins - 1);
        acc[idx].0 += 1;
        acc[idx].1 += p;
        if *label {
            acc[idx].2 += 1;
        }
    }
    acc.into_iter()
        .enumerate()
        .map(|(i, (count, sum_p, positives))| ReliabilityBin {
            lower: i as f64 / bins as f64,
            upper: (i + 1) as f64 / bins as f64,
            count,
            predicted: if count == 0 {
                0.0
            } else {
                sum_p / count as f64
            },
            observed: if count == 0 {
                0.0
            } else {
                positives as f64 / count as f64
            },
        })
        .collect()
}

/// Erreur de calibration attendue : écart moyen entre annoncé et observé,
/// pondéré par le nombre d'échantillons de chaque tranche.
///
/// 0 = parfaitement calibré. Une valeur de 0,20 veut dire que le modèle se
/// trompe en moyenne de 20 points de probabilité — il peut encore très bien
/// ORDONNER (bonne AUC), mais ses valeurs ne veulent rien dire, et toute somme
/// pondérée qui le mélange à d'autres signaux lui donne un poids faussé.
pub fn expected_calibration_error(samples: &[(f64, bool)], bins: usize) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let total = samples.len() as f64;
    let ece = reliability_curve(samples, bins)
        .iter()
        .filter(|b| b.count > 0)
        .map(|b| (b.count as f64 / total) * (b.predicted - b.observed).abs())
        .sum();
    Some(ece)
}

/// NDCG@k — gain cumulé actualisé, normalisé.
///
/// `gains` est la liste des gains DANS L'ORDRE OÙ ILS ONT ÉTÉ SERVIS. Le
/// résultat vaut 1,0 quand cet ordre est le meilleur possible pour ces mêmes
/// gains, et baisse à mesure que ce qui a de la valeur descend dans la page.
///
/// C'est la métrique qui manque à l'AUC : l'AUC dit si le modèle sait
/// ordonner, le NDCG dit si les bons items sont EN HAUT. Se tromper à la
/// place 2 coûte beaucoup plus cher qu'à la place 40, ce que l'AUC ignore
/// complètement.
///
/// `None` quand rien n'a de valeur dans la page : il n'y a alors pas d'ordre
/// idéal auquel se comparer, et rendre 0 laisserait croire à un échec de
/// classement alors qu'il n'y avait rien à classer.
pub fn ndcg_at_k(gains: &[f64], k: usize) -> Option<f64> {
    if gains.is_empty() || k == 0 {
        return None;
    }
    let dcg = |values: &[f64]| -> f64 {
        values
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, g)| g / ((i + 2) as f64).log2())
            .sum()
    };

    let mut ideal: Vec<f64> = gains.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg(&ideal);
    if idcg <= 0.0 {
        return None;
    }
    Some((dcg(gains) / idcg).clamp(0.0, 1.0))
}

// ═══════════════════════════════════════════════════════════════════════════
// Fenêtre glissante d'évaluation progressive
// ═══════════════════════════════════════════════════════════════════════════

/// Rapport publiable pour un modèle.
///
/// Les métriques sont toutes optionnelles à dessein : une AUC sur une classe
/// unique, ou sur trente couples, n'existe pas. Renvoyer 0 dans ces cas-là
/// invite à conclure ; renvoyer `null` dit ce qui est vrai — on ne sait pas
/// encore.
#[derive(Debug, Clone, Serialize, Default)]
pub struct EvalReport {
    /// Couples (prédiction, vérité) dans la fenêtre courante.
    pub samples: usize,
    /// Le modèle sait-il ORDONNER ? 0,5 = hasard.
    pub auc: Option<f64>,
    /// Le modèle est-il sûr à bon escient ?
    pub log_loss: Option<f64>,
    /// Quand il dit 30 %, est-ce que ça arrive 30 % du temps ?
    pub calibration_error: Option<f64>,
    /// Erreur quadratique moyenne — la métrique utile quand la cible est
    /// continue (temps de lecture) plutôt que binaire.
    pub rmse: Option<f64>,
    /// Fréquence réelle des positifs dans la fenêtre. À surveiller : un taux
    /// nul après des milliers d'échantillons signale un étiquetage cassé ou un
    /// signal qui n'arrive pas, pas un public sans réaction.
    pub positive_rate: f64,
    /// Prédiction moyenne. Loin du taux réel = modèle décalé.
    pub mean_prediction: f64,
    pub reliability: Vec<ReliabilityBin>,
}

/// Fenêtre glissante des derniers couples (prédiction, vérité), partagée entre
/// les requêtes.
#[derive(Clone)]
pub struct OnlineEval(Arc<RwLock<VecDeque<(f32, f32)>>>);

impl OnlineEval {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(VecDeque::with_capacity(WINDOW))))
    }

    /// Enregistre un couple.
    ///
    /// ⚠ `prediction` DOIT avoir été calculée AVANT la mise à jour du modèle
    /// sur ce même échantillon. C'est toute la validité de la mesure : une
    /// prédiction faite après coup décrit un modèle qui a déjà vu la réponse,
    /// et l'AUC qui en sort est une flatterie, pas un diagnostic.
    pub fn record(&self, prediction: f64, truth: f64) {
        if !prediction.is_finite() || !truth.is_finite() {
            return;
        }
        let mut w = self.0.write().unwrap();
        if w.len() == WINDOW {
            w.pop_front();
        }
        w.push_back((prediction as f32, truth as f32));
    }

    pub fn len(&self) -> usize {
        self.0.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rapport sur la fenêtre courante.
    ///
    /// Les métriques binaires ne sont calculées que si la cible EST binaire.
    /// Le modèle de temps de lecture apprend sur une valeur continue : lui
    /// sortir une AUC reviendrait à inventer un seuil, et le seuil choisi
    /// déciderait du résultat.
    pub fn report(&self) -> EvalReport {
        let window: Vec<(f64, f64)> = {
            let w = self.0.read().unwrap();
            w.iter().map(|(p, t)| (*p as f64, *t as f64)).collect()
        };
        let n = window.len();
        if n == 0 {
            return EvalReport::default();
        }

        let mean_prediction = window.iter().map(|(p, _)| p).sum::<f64>() / n as f64;
        let positive_rate = window.iter().filter(|(_, t)| *t > 0.5).count() as f64 / n as f64;
        let rmse = Some(
            (window.iter().map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64).sqrt(),
        );

        if n < MIN_SAMPLES_FOR_METRICS {
            return EvalReport {
                samples: n,
                positive_rate,
                mean_prediction,
                rmse,
                ..Default::default()
            };
        }

        let is_binary = window.iter().all(|(_, t)| *t == 0.0 || *t == 1.0);
        if !is_binary {
            return EvalReport {
                samples: n,
                positive_rate,
                mean_prediction,
                rmse,
                ..Default::default()
            };
        }

        let binary: Vec<(f64, bool)> = window.iter().map(|(p, t)| (*p, *t > 0.5)).collect();
        EvalReport {
            samples: n,
            auc: roc_auc(&binary),
            log_loss: log_loss(&binary),
            calibration_error: expected_calibration_error(&binary, RELIABILITY_BINS),
            rmse,
            positive_rate,
            mean_prediction,
            reliability: reliability_curve(&binary, RELIABILITY_BINS),
        }
    }
}

impl Default for OnlineEval {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── AUC ────────────────────────────────────────────────────────────────

    #[test]
    fn une_separation_parfaite_donne_une_auc_de_un() {
        let s = vec![(0.9, true), (0.8, true), (0.2, false), (0.1, false)];
        assert_eq!(roc_auc(&s), Some(1.0));
    }

    #[test]
    fn un_ordre_exactement_inverse_donne_zero() {
        let s = vec![(0.1, true), (0.2, true), (0.8, false), (0.9, false)];
        assert_eq!(roc_auc(&s), Some(0.0));
    }

    /// Le cas qui compte le plus en pratique : un modèle qui n'a rien appris
    /// renvoie la MÊME probabilité pour tout le monde. La bonne réponse est
    /// 0,5 (aucun pouvoir d'ordonnancement), pas 0 ou 1 selon l'ordre
    /// d'arrivée des lignes — c'est exactement ce qu'un calcul naïf des ex
    /// æquo se trompe à produire.
    #[test]
    fn un_modele_qui_predit_toujours_pareil_vaut_le_hasard() {
        let s = vec![(0.3, true), (0.3, false), (0.3, true), (0.3, false)];
        assert_eq!(roc_auc(&s), Some(0.5));
    }

    #[test]
    fn les_ex_aequo_partiels_comptent_pour_une_demi_victoire() {
        // Un positif au-dessus d'un négatif (victoire), un ex æquo (demie).
        let s = vec![(0.9, true), (0.5, true), (0.5, false), (0.1, false)];
        // Paires : (0.9,0.5N)=1, (0.9,0.1)=1, (0.5P,0.5N)=0.5, (0.5P,0.1)=1
        // → 3.5 / 4 = 0.875
        assert_eq!(roc_auc(&s), Some(0.875));
    }

    #[test]
    fn une_classe_absente_n_a_pas_d_auc() {
        assert!(roc_auc(&[(0.9, true), (0.8, true)]).is_none());
        assert!(roc_auc(&[(0.9, false)]).is_none());
        assert!(roc_auc(&[]).is_none());
    }

    /// L'AUC ne doit pas bouger si on applique une transformation monotone
    /// aux scores : c'est ce qui la rend indépendante de la calibration.
    #[test]
    fn l_auc_est_insensible_a_une_transformation_monotone() {
        let s = vec![(0.9, true), (0.4, false), (0.7, true), (0.1, false)];
        let ecrase: Vec<(f64, bool)> = s.iter().map(|(p, l)| (p * 0.01 + 0.5, *l)).collect();
        assert_eq!(roc_auc(&s), roc_auc(&ecrase));
    }

    // ─── Log-loss ───────────────────────────────────────────────────────────

    #[test]
    fn la_log_loss_punit_la_confiance_mal_placee() {
        let sur_et_juste = log_loss(&[(0.99, true), (0.01, false)]).unwrap();
        let hesitant = log_loss(&[(0.55, true), (0.45, false)]).unwrap();
        let sur_et_faux = log_loss(&[(0.01, true), (0.99, false)]).unwrap();
        assert!(sur_et_juste < hesitant, "{sur_et_juste} < {hesitant}");
        assert!(hesitant < sur_et_faux, "{hesitant} < {sur_et_faux}");
    }

    #[test]
    fn une_prediction_certaine_et_fausse_ne_rend_pas_la_log_loss_infinie() {
        let l = log_loss(&[(0.0, true), (0.9, true)]).unwrap();
        assert!(l.is_finite(), "log-loss non finie : {l}");
    }

    // ─── Calibration ────────────────────────────────────────────────────────

    #[test]
    fn un_modele_parfaitement_calibre_a_une_ece_nulle() {
        // Tranche à 0,5 annoncée, exactement la moitié de positifs observés.
        let s: Vec<(f64, bool)> = (0..100).map(|i| (0.5, i % 2 == 0)).collect();
        let ece = expected_calibration_error(&s, 10).unwrap();
        assert!(ece < 1e-9, "ece={ece}");
    }

    /// Le cœur du diagnostic : un modèle peut ORDONNER parfaitement et être
    /// complètement décalé. L'AUC dit 1,0, l'ECE dit qu'il n'y a rien à croire
    /// dans ses valeurs. Les deux métriques ne sont pas redondantes.
    #[test]
    fn ordonner_parfaitement_n_empeche_pas_d_etre_mal_calibre() {
        // Ordre PARFAIT — les scores les plus hauts sont exactement les
        // positifs — mais tous annoncés à ~0,99 alors que seule la moitié se
        // réalise.
        let mut s: Vec<(f64, bool)> = Vec::new();
        for i in 0..100 {
            s.push((0.99 - i as f64 * 0.0001, i < 50));
        }
        assert_eq!(roc_auc(&s), Some(1.0), "l'ordre est parfait");
        let ece = expected_calibration_error(&s, 10).unwrap();
        assert!(ece > 0.4, "et pourtant totalement décalé : ece={ece}");
    }

    #[test]
    fn la_courbe_de_fiabilite_couvre_tout_l_intervalle() {
        let curve = reliability_curve(&[(0.05, true), (0.95, false)], 10);
        assert_eq!(curve.len(), 10);
        assert_eq!(curve[0].count, 1);
        assert_eq!(curve[9].count, 1);
        assert_eq!(curve.iter().map(|b| b.count).sum::<usize>(), 2);
        // Une prédiction pile à 1,0 doit tomber dans la dernière tranche, pas
        // déborder.
        let bord = reliability_curve(&[(1.0, true)], 10);
        assert_eq!(bord[9].count, 1);
    }

    // ─── NDCG ───────────────────────────────────────────────────────────────

    #[test]
    fn un_classement_ideal_donne_un_ndcg_de_un() {
        assert_eq!(ndcg_at_k(&[3.0, 2.0, 1.0, 0.0], 4), Some(1.0));
    }

    /// La différence avec l'AUC : le NDCG pèse les POSITIONS. Descendre le
    /// meilleur item de la place 1 à la place 4 doit coûter plus cher que de
    /// l'échanger avec son voisin immédiat.
    #[test]
    fn descendre_le_meilleur_item_coute_plus_cher_que_le_deplacer_d_un_cran() {
        let ideal = ndcg_at_k(&[3.0, 2.0, 1.0, 0.0], 4).unwrap();
        let un_cran = ndcg_at_k(&[2.0, 3.0, 1.0, 0.0], 4).unwrap();
        let tout_en_bas = ndcg_at_k(&[2.0, 1.0, 0.0, 3.0], 4).unwrap();
        assert!(un_cran < ideal, "{un_cran} < {ideal}");
        assert!(tout_en_bas < un_cran, "{tout_en_bas} < {un_cran}");
    }

    #[test]
    fn le_ndcg_ne_regarde_pas_au_dela_de_k() {
        // Ce qui se passe après la place 2 ne doit pas changer NDCG@2.
        let a = ndcg_at_k(&[3.0, 3.0, 0.0, 1.0], 2);
        let b = ndcg_at_k(&[3.0, 3.0, 1.0, 0.0], 2);
        assert_eq!(a, b);
    }

    #[test]
    fn une_page_sans_valeur_n_a_pas_de_ndcg() {
        // Aucun ordre idéal auquel se comparer : `None`, pas 0 — sinon on
        // laisse croire à un échec de classement là où il n'y avait rien à
        // classer.
        assert!(ndcg_at_k(&[0.0, 0.0, 0.0], 3).is_none());
        assert!(ndcg_at_k(&[], 3).is_none());
        assert!(ndcg_at_k(&[1.0], 0).is_none());
    }

    // ─── Fenêtre glissante ──────────────────────────────────────────────────

    #[test]
    fn la_fenetre_ne_depasse_jamais_sa_taille() {
        let e = OnlineEval::new();
        for i in 0..(WINDOW + 500) {
            e.record((i % 100) as f64 / 100.0, (i % 2) as f64);
        }
        assert_eq!(e.len(), WINDOW);
    }

    #[test]
    fn aucune_metrique_publiee_sous_le_seuil() {
        let e = OnlineEval::new();
        for i in 0..(MIN_SAMPLES_FOR_METRICS - 1) {
            e.record(0.5, (i % 2) as f64);
        }
        let r = e.report();
        assert_eq!(r.samples, MIN_SAMPLES_FOR_METRICS - 1);
        assert!(
            r.auc.is_none(),
            "sur si peu de couples, une AUC ne mesure que le hasard"
        );
        // Les indicateurs bruts, eux, restent disponibles tout de suite.
        assert!(r.rmse.is_some());
        assert!((r.positive_rate - 0.5).abs() < 0.02);
    }

    #[test]
    fn une_fenetre_discriminante_ressort_avec_une_bonne_auc() {
        let e = OnlineEval::new();
        for i in 0..MIN_SAMPLES_FOR_METRICS {
            if i % 2 == 0 {
                e.record(0.80, 1.0);
            } else {
                e.record(0.20, 0.0);
            }
        }
        let r = e.report();
        assert_eq!(r.auc, Some(1.0));
        assert!(r.log_loss.unwrap() > 0.0);
        assert_eq!(r.reliability.len(), RELIABILITY_BINS);
    }

    /// Cible continue (temps de lecture) : pas d'AUC, mais une RMSE. Lui
    /// sortir une AUC reviendrait à inventer un seuil, et le seuil choisi
    /// déciderait du résultat.
    #[test]
    fn une_cible_continue_ne_produit_pas_d_auc() {
        let e = OnlineEval::new();
        for i in 0..MIN_SAMPLES_FOR_METRICS {
            e.record(0.5, (i % 7) as f64 / 7.0);
        }
        let r = e.report();
        assert!(r.auc.is_none());
        assert!(r.log_loss.is_none());
        assert!(r.rmse.is_some());
    }

    #[test]
    fn une_valeur_non_finie_est_ignoree() {
        let e = OnlineEval::new();
        e.record(f64::NAN, 1.0);
        e.record(0.5, f64::INFINITY);
        assert!(e.is_empty());
    }
}
