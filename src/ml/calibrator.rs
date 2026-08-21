//! Recalibration des probabilités prédites — mise à l'échelle de Platt.
//!
//! ⚠ **À ne pas confondre avec `crate::calibration`**, qui est le parcours de
//! recalibration des GOÛTS vu par l'utilisateur à l'inscription (des tours de
//! tweets à aimer). Ici il s'agit des probabilités que crachent les têtes ML.
//!
//! ── Le problème ─────────────────────────────────────────────────────────────
//! `crate::eval` sait dire depuis peu qu'une tête est mal calibrée (ECE, courbe
//! de fiabilité) : quand elle annonce 30 %, l'événement arrive peut-être 8 % du
//! temps. Savoir le mesurer ne le corrige pas.
//!
//! Tant qu'une seule tête décide, ça n'a aucune importance : la mise à l'échelle
//! de Platt est **monotone**, elle ne change donc jamais l'ordre d'une liste
//! triée sur une seule tête. Ce qui change tout, c'est que le classement mélange
//! désormais QUATRE têtes par somme pondérée (`blend_positive`). Une somme
//! pondérée additionne des grandeurs supposées comparables : si `ctr` annonce
//! systématiquement 0,30 là où la vérité est 0,08, et que `amplify` est juste,
//! alors le poids réellement appliqué à `ctr` n'est plus `W_CTR` — il est
//! silencieusement multiplié par le décalage. Les poids écrits dans le barème
//! cessent de décrire ce que fait le moteur.
//!
//! C'est exactement l'avertissement que porte déjà `expected_calibration_error` :
//! « il peut encore très bien ORDONNER, mais ses valeurs ne veulent rien dire, et
//! toute somme pondérée qui le mélange à d'autres signaux lui donne un poids
//! faussé. » Ce module est la moitié qui manquait.
//!
//! ── La méthode ──────────────────────────────────────────────────────────────
//! Platt (1999) : une régression logistique à une seule entrée, le **logit** de
//! la probabilité annoncée.
//!
//! ```text
//!     p_calibré = σ(a · logit(p) + b)
//! ```
//!
//! Deux paramètres, pas davantage — et c'est délibéré. La régression isotonique
//! corrige des déformations plus riches, mais elle a besoin de beaucoup plus de
//! données pour ne pas simplement mémoriser le bruit, et la volumétrie ici est
//! petite. À `a = 1, b = 0` la transformation est l'identité : un correcteur non
//! ajusté ne fait donc rien du tout, ce qui est la bonne valeur par défaut.
//!
//! Travailler sur le logit et non sur `p` directement importe : c'est ce qui
//! rend la correction affine dans l'espace où le modèle raisonne, et ce qui lui
//! permet de corriger un décalage global (`b`) séparément d'un excès de
//! confiance (`a < 1`, les prédictions sont tassées vers le centre) ou d'un
//! manque de confiance (`a > 1`).

use serde::Serialize;

/// Bornes du logit.
///
/// Une probabilité de 0 ou de 1 donne un logit infini, et un seul échantillon
/// suffirait alors à faire diverger l'ajustement. `1e-6` place la borne à
/// ±13,8 : largement au-delà de ce qu'un modèle en ligne produit, assez près de
/// zéro pour ne rien déformer d'utile.
const EPS: f64 = 1e-6;

/// Échantillons minimum avant d'ajuster.
///
/// Même ordre de grandeur que le seuil de démarrage à froid des têtes (200) :
/// en dessous, la correction décrirait le bruit de la fenêtre plutôt que le
/// biais du modèle, et on remplacerait un défaut mesuré par un défaut inventé.
pub const MIN_SAMPLES_FOR_FIT: usize = 200;

/// Itérations de la descente. La surface est convexe et à deux paramètres :
/// elle converge bien avant, la borne n'est qu'un garde-fou.
const MAX_ITER: usize = 100;

/// Seuil d'arrêt sur le déplacement des paramètres.
const TOL: f64 = 1e-7;

fn logit(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Correcteur de Platt ajusté sur une fenêtre d'observations.
///
/// Non ajusté, il est l'identité — c'est ce qui permet de le poser partout sans
/// rien changer tant qu'il n'a pas vu assez de monde.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PlattCalibrator {
    /// Pente sur le logit. 1 = aucune correction d'échelle.
    pub a: f64,
    /// Décalage. 0 = aucune correction de biais.
    pub b: f64,
    /// Faux tant qu'aucun ajustement n'a réussi : `apply` est alors l'identité.
    pub fitted: bool,
    /// Nombre d'échantillons ayant servi au dernier ajustement.
    pub samples: usize,
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        Self { a: 1.0, b: 0.0, fitted: false, samples: 0 }
    }
}

impl PlattCalibrator {
    /// Ajuste sur des couples (probabilité annoncée, vérité).
    ///
    /// Rend `None` — et non un correcteur identité — quand l'ajustement n'a pas
    /// lieu d'être : trop peu d'échantillons, ou une seule classe présente. Le
    /// distinguer importe : « je n'ai pas pu » et « il n'y a rien à corriger »
    /// ne demandent pas la même conduite en amont.
    pub fn fit(samples: &[(f64, bool)]) -> Option<Self> {
        if samples.len() < MIN_SAMPLES_FOR_FIT {
            return None;
        }
        let n_pos = samples.iter().filter(|(_, t)| *t).count();
        let n_neg = samples.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return None;
        }

        // Cibles corrigées par le prior, recette de Platt. On n'ajuste PAS sur
        // 0 et 1 : la régression logistique pousserait alors ses paramètres vers
        // l'infini pour atteindre des cibles qu'une sigmoïde n'atteint jamais,
        // et le correcteur deviendrait une marche d'escalier — un classifieur
        // dur là où on voulait une probabilité. Les cibles amorties gardent la
        // correction douce, ce qui est tout l'intérêt d'avoir mesuré une ECE
        // plutôt qu'un taux d'erreur.
        let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let t_neg = 1.0 / (n_neg as f64 + 2.0);

        let points: Vec<(f64, f64)> = samples
            .iter()
            .map(|(p, truth)| (logit(*p), if *truth { t_pos } else { t_neg }))
            .collect();

        let (mut a, mut b) = (1.0_f64, 0.0_f64);
        for _ in 0..MAX_ITER {
            // Newton amorti sur la log-vraisemblance : gradient et hessienne se
            // calculent en une passe, et la surface étant convexe à deux
            // paramètres, aucune recherche linéaire n'est nécessaire.
            let (mut g_a, mut g_b) = (0.0, 0.0);
            let (mut h_aa, mut h_ab, mut h_bb) = (0.0, 0.0, 0.0);
            for (x, t) in &points {
                let p = sigmoid(a * x + b);
                let d = p - t;
                g_a += d * x;
                g_b += d;
                let w = p * (1.0 - p);
                h_aa += w * x * x;
                h_ab += w * x;
                h_bb += w;
            }
            // Régularisation de Tikhonov : sans elle, une fenêtre où tous les
            // logits se ressemblent rend la hessienne singulière et le pas part
            // à l'infini.
            h_aa += 1e-9;
            h_bb += 1e-9;

            let det = h_aa * h_bb - h_ab * h_ab;
            if det.abs() < 1e-12 {
                break;
            }
            let step_a = (h_bb * g_a - h_ab * g_b) / det;
            let step_b = (h_aa * g_b - h_ab * g_a) / det;
            a -= step_a;
            b -= step_b;
            if step_a.abs() < TOL && step_b.abs() < TOL {
                break;
            }
        }

        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        // ── Pente négative : on refuse ──────────────────────────────────────
        // `a <= 0` veut dire que sur cette fenêtre, plus la tête annonce une
        // probabilité haute, MOINS l'événement arrive. Le correcteur qui en
        // sort est parfaitement valide au sens de la vraisemblance — et il
        // INVERSE le classement, puisque `apply` devient décroissante.
        //
        // Ce n'est plus de la calibration : recalibrer suppose que le modèle
        // sait ordonner et se trompe seulement d'échelle. Une pente négative
        // dit qu'il ne sait pas ordonner du tout (tête anti-prédictive, ou
        // fenêtre sans aucun signal — c'est ce que produit un modèle qui rend
        // la même valeur pour tout le monde, où la pente n'est que du bruit).
        //
        // Dans les deux cas la conduite est la même : ne rien corriger et
        // laisser l'AUC du rapport d'éval dire ce qui ne va pas. Une AUC sous
        // 0,5 est le symptôme à regarder, pas quelque chose à rattraper ici.
        if a <= 0.0 {
            return None;
        }
        Some(Self { a, b, fitted: true, samples: samples.len() })
    }

    /// Applique la correction. Identité tant que le correcteur n'est pas ajusté.
    pub fn apply(&self, p: f64) -> f64 {
        if !self.fitted || !p.is_finite() {
            return p;
        }
        sigmoid(self.a * logit(p) + self.b).clamp(0.0, 1.0)
    }
}

/// Ce que la correction changerait, sans l'appliquer.
///
/// Sert à répondre « est-ce que ça vaut le coup ? » avant de brancher quoi que
/// ce soit : on ajuste sur la fenêtre, on remesure l'ECE sur cette même fenêtre,
/// et on compare. C'est optimiste par construction — l'ajustement a vu ces
/// données-là — donc un gain affiché ici est un PLAFOND, pas une promesse. Un
/// plafond faible est en revanche une réponse définitive : inutile de brancher.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationGain {
    pub calibrator: Option<PlattCalibrator>,
    /// ECE avant correction.
    pub ece_before: Option<f64>,
    /// ECE après correction, sur la même fenêtre (donc optimiste).
    pub ece_after: Option<f64>,
}

/// Mesure le gain potentiel d'une recalibration sur une fenêtre.
pub fn calibration_gain(samples: &[(f64, bool)], bins: usize) -> CalibrationGain {
    let before = crate::eval::expected_calibration_error(samples, bins);
    let Some(cal) = PlattCalibrator::fit(samples) else {
        return CalibrationGain { calibrator: None, ece_before: before, ece_after: None };
    };
    let corrected: Vec<(f64, bool)> =
        samples.iter().map(|(p, t)| (cal.apply(*p), *t)).collect();
    CalibrationGain {
        calibrator: Some(cal),
        ece_before: before,
        ece_after: crate::eval::expected_calibration_error(&corrected, bins),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fabrique une fenêtre où le modèle est SÛR DE LUI À TORT : il annonce
    /// `announced`, la vérité arrive avec la fréquence `actual`.
    fn window(announced: f64, actual: f64, n: usize) -> Vec<(f64, bool)> {
        (0..n)
            .map(|i| (announced, (i as f64 / n as f64) < actual))
            .collect()
    }

    #[test]
    fn un_correcteur_non_ajuste_est_l_identite() {
        let cal = PlattCalibrator::default();
        for p in [0.0, 0.01, 0.5, 0.9, 1.0] {
            assert_eq!(cal.apply(p), p);
        }
    }

    #[test]
    fn pas_d_ajustement_sous_le_seuil() {
        assert!(PlattCalibrator::fit(&window(0.3, 0.1, MIN_SAMPLES_FOR_FIT - 1)).is_none());
    }

    #[test]
    fn pas_d_ajustement_sur_une_seule_classe() {
        let tout_negatif: Vec<(f64, bool)> = (0..500).map(|_| (0.3, false)).collect();
        assert!(PlattCalibrator::fit(&tout_negatif).is_none());
    }

    #[test]
    fn la_correction_ramene_l_annonce_vers_la_verite() {
        // Le modèle annonce 30 %, l'événement arrive 8 % du temps.
        let w = window(0.30, 0.08, 2000);
        let cal = PlattCalibrator::fit(&w).expect("ajustable");
        let corrige = cal.apply(0.30);
        assert!(
            (corrige - 0.08).abs() < 0.02,
            "annonce corrigee {corrige:.4}, attendue proche de 0.08"
        );
    }

    #[test]
    fn la_correction_fait_baisser_l_ece() {
        let w = window(0.30, 0.08, 2000);
        let gain = calibration_gain(&w, 10);
        let avant = gain.ece_before.expect("mesurable");
        let apres = gain.ece_after.expect("mesurable");
        assert!(avant > 0.15, "le cas de test doit etre franchement decale (ECE {avant:.3})");
        assert!(apres < avant / 2.0, "ECE {avant:.3} -> {apres:.3} : correction insuffisante");
    }

    /// Fenêtre d'un modèle qui SAIT ordonner mais annonce trop haut : la
    /// probabilité de l'événement croît bien avec la valeur annoncée.
    fn window_avec_signal(n: usize) -> Vec<(f64, bool)> {
        (0..n)
            .map(|i| {
                let p = 0.05 + (i % 90) as f64 / 100.0;
                // Vérité corrélée à `p`, mais trois fois plus rare que ce que
                // le modèle annonce : il ordonne juste, il surestime.
                let vraie = p / 3.0;
                let tirage = ((i * 7919) % 1000) as f64 / 1000.0;
                (p, tirage < vraie)
            })
            .collect()
    }

    /// Le point qui justifie de ne calibrer que pour le MÉLANGE : sur une tête
    /// seule, la correction ne peut rien changer au classement.
    #[test]
    fn la_correction_ne_change_jamais_l_ordre() {
        let cal = PlattCalibrator::fit(&window_avec_signal(4000)).expect("ajustable");
        assert!(cal.a > 0.0, "pente {} : le cas de test doit porter du signal", cal.a);

        let mut scores: Vec<f64> = (1..=99).map(|i| i as f64 / 100.0).collect();
        let avant = scores.clone();
        scores.sort_by(|x, y| cal.apply(*x).partial_cmp(&cal.apply(*y)).unwrap());
        assert_eq!(scores, avant, "la mise a l'echelle de Platt doit etre monotone");
    }

    /// Une tête anti-prédictive ne se « recalibre » pas : le correcteur qui en
    /// sortirait inverserait le classement. On refuse plutôt que de corriger.
    #[test]
    fn pas_d_ajustement_sur_une_tete_anti_predictive() {
        let inverse: Vec<(f64, bool)> = window_avec_signal(4000)
            .into_iter()
            .map(|(p, t)| (1.0 - p, t))
            .collect();
        assert!(PlattCalibrator::fit(&inverse).is_none());
    }

    #[test]
    fn un_modele_deja_calibre_est_peu_touche() {
        // Annonce 20 %, arrive 20 % du temps : il n'y a rien à corriger.
        let w = window(0.20, 0.20, 2000);
        let cal = PlattCalibrator::fit(&w).expect("ajustable");
        assert!(
            (cal.apply(0.20) - 0.20).abs() < 0.02,
            "un modele juste ne doit pas etre deplace ({:.4})",
            cal.apply(0.20)
        );
    }
}
