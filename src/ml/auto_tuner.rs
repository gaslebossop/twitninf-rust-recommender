//! Auto-réglage des poids de dimensions depuis le modèle CTR.
//!
//! ── Ce qu'il faisait, et pourquoi c'était faux ──────────────────────────────
//! Le principe d'origine : le modèle CTR apprend en continu, ses coefficients
//! `[0..8]` « représentent l'importance de D1-D8 », on les normalise et ça
//! devient les poids de scoring.
//!
//! Relevé en production le 2026-08-31, après des semaines de fonctionnement
//! sans surveillance :
//!
//! ```text
//! d1 vélocité 0,232 · d2 contenu 0,183 · d3 graphe social 0,082
//! d4 temporel 0,039 · d5 comportemental 0,0008 · d6 diversité 0,0008
//! d7 viralité 0,189 · d8 personnalisation 0,172 · d9 LLM 0,100
//! ```
//!
//! **D5 et D6 valaient exactement le plancher** (`0.001` normalisé) : leurs
//! coefficients appris étaient NÉGATIFS, l'écrêtage les a transformés en une
//! part de 0,08 % — c'est-à-dire supprimés. Et D3, le graphe social du lecteur,
//! était tombé de 0,22 à 0,082 pendant que vélocité + viralité montaient à
//! 0,42.
//!
//! Autrement dit : **le réglage automatique dépersonnalisait le fil**, et la
//! boucle se referme sur elle-même — plus de contenu populaire servi, plus de
//! contenu populaire cliqué, plus de poids sur la popularité.
//!
//! ── Les trois corrections ───────────────────────────────────────────────────
//!
//! 1. **Un coefficient négatif ne devient plus un plancher.** Blanchir « cette
//!    dimension nuit au CTR » en « cette dimension vaut 0,08 % » jette
//!    l'information du signe et supprime la dimension. Une dimension dont le
//!    coefficient appris n'est pas positif garde désormais sa part PAR DÉFAUT :
//!    le modèle n'a rien de fiable à en dire, on ne le laisse pas décider.
//!
//! 2. **Rétrécissement vers les défauts.** Les coefficients d'une régression
//!    logistique sur des traits CORRÉLÉS ne sont pas des importances : D1 et D7
//!    mesurent tous deux la vélocité, D2 et D8 la correspondance au contenu.
//!    Entre deux traits colinéaires, le coefficient se répartit arbitrairement
//!    et peut changer de signe. On mélange donc l'estimation apprise au défaut
//!    au lieu de la laisser écraser — d'autant que les 9 219 échantillons de
//!    production viennent d'**onze** lecteurs réellement actifs, pas de neuf
//!    mille personnes indépendantes.
//!
//! 3. **Plancher de personnalisation.** Quoi qu'apprenne le modèle, les
//!    dimensions qui décrivent CE lecteur (D3 graphe, D5 comportement, D8
//!    affinités, D10 goût sémantique) gardent une part minimale du score. Sans
//!    ce plancher, optimiser le CTR global converge mécaniquement vers « ce que
//!    tout le monde clique », c'est-à-dire vers un fil identique pour tous.

use std::sync::{Arc, RwLock};

use tracing::{debug, info, warn};

use crate::admin::AlgoWeights;
use crate::ml::CtrPredictor;

const MIN_SAMPLES_FOR_TUNING: u64 = 500;

/// Bande de CTR global considérée comme plausible. En dehors, on refuse de
/// réécrire les poids de scoring.
///
/// L'auto-tuner écrase les poids qui pilotent tout le feed à partir de ce que
/// le modèle a appris. Si l'étiquetage est dégénéré — tous les samples positifs,
/// ou tous négatifs — les poids appris ne veulent rien dire et les propager
/// casserait le classement pour tout le monde. C'est exactement ce qui serait
/// arrivé avec l'ancien label (`weight > 0.0`, donc une vue comptait comme un
/// clic) : un CTR global de ~100 % et des poids de bruit poussés en production.
const MIN_PLAUSIBLE_CTR: f64 = 0.005;
const MAX_PLAUSIBLE_CTR: f64 = 0.80;

/// Nombre d'échantillons auquel l'estimation apprise pèse autant que le défaut.
///
/// `α = samples / (samples + DEMI_CONFIANCE)`. Volontairement élevé : le compte
/// d'échantillons SURESTIME largement l'information réellement disponible.
/// 9 219 impressions étiquetées venaient, en production, de **onze** lecteurs
/// actifs sur sept jours — ce sont onze avis, pas neuf mille. Un `K` bas
/// laisserait le bruit d'une poignée de sessions redéfinir le fil de tout le
/// monde.
const CONFIANCE_DEMI: f64 = 20_000.0;

/// Part minimale du score réservée aux dimensions qui décrivent CE lecteur.
///
/// Le défaut vaut 0,45 (voir `AlgoWeights::personalization_share`) : ce
/// plancher ne mord donc que si le réglage automatique pousse dans le sens de
/// la dépersonnalisation. Il n'impose rien tant que le modèle reste sage — il
/// borne jusqu'où il peut aller quand il ne l'est pas.
const PERSONALIZATION_FLOOR: f64 = 0.40;

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
        Self {
            current: Arc::new(RwLock::new(TunerState::default())),
        }
    }

    /// Retente une mise à jour des poids depuis le CTR model.
    /// À appeler après chaque interaction enregistrée.
    pub fn maybe_update(&self, ctr: &CtrPredictor, admin_override: Option<&AlgoWeights>) {
        // Si override admin actif, on ne touche pas aux poids
        if admin_override.is_some() {
            return;
        }

        let (samples, global_ctr) = ctr.stats();

        // Vérifier si on a assez de données ET si on a de nouvelles données depuis le dernier tune
        let last_tune = self.current.read().unwrap().last_tune_at;
        if samples < MIN_SAMPLES_FOR_TUNING || samples < last_tune + 100 {
            return;
        }

        // Garde-fou : un CTR global aberrant signale un étiquetage cassé, pas un
        // apprentissage réussi. On garde les poids en place plutôt que de
        // propager du bruit à l'ensemble du feed.
        if !(MIN_PLAUSIBLE_CTR..=MAX_PLAUSIBLE_CTR).contains(&global_ctr) {
            warn!(
                samples,
                global_ctr, "AutoTuner: CTR global hors bande plausible — poids inchangés"
            );
            return;
        }

        // ⚠ Le point de référence est le DÉFAUT, pas l'état courant.
        //
        // Rétrécir vers l'état courant ferait dériver les poids sans limite :
        // chaque passage tirerait un peu plus loin depuis l'endroit où le
        // passage précédent s'était arrêté, et le rétrécissement ne
        // retiendrait plus rien. C'est ainsi que D3 est descendu de 0,22 à
        // 0,082 par petits pas dont aucun n'était aberrant.
        let base = AlgoWeights::default();
        if let Some(weights) = ctr.extract_dimension_weights(&base) {
            let mut state = self.current.write().unwrap();
            state.weights = weights;
            state.auto_tuned = true;
            state.last_tune_at = samples;
            drop(state);
            info!(
                samples,
                "AutoTuner: dimension weights updated from CTR model"
            );
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
    fn default() -> Self {
        Self::new()
    }
}

/// Mélange deux jeux de poids : `(1-t)·a + t·b`.
///
/// Les deux somment à 1, donc le mélange aussi — aucune renormalisation à
/// faire, et l'échelle des scores ne dérive pas.
fn melanger(a: &AlgoWeights, b: &AlgoWeights, t: f64) -> AlgoWeights {
    let t = t.clamp(0.0, 1.0);
    let m = |x: f64, y: f64| x * (1.0 - t) + y * t;
    AlgoWeights {
        d1_engagement_velocity: m(a.d1_engagement_velocity, b.d1_engagement_velocity),
        d2_content_intelligence: m(a.d2_content_intelligence, b.d2_content_intelligence),
        d3_social_graph: m(a.d3_social_graph, b.d3_social_graph),
        d4_temporal: m(a.d4_temporal, b.d4_temporal),
        d5_behavioral: m(a.d5_behavioral, b.d5_behavioral),
        d6_diversity: m(a.d6_diversity, b.d6_diversity),
        d7_viral: m(a.d7_viral, b.d7_viral),
        d8_personalization: m(a.d8_personalization, b.d8_personalization),
        d9_llm_understanding: m(a.d9_llm_understanding, b.d9_llm_understanding),
        d10_taste_affinity: m(a.d10_taste_affinity, b.d10_taste_affinity),
    }
}

/// Ramène les poids vers les défauts juste assez pour que la part de
/// personnalisation atteigne le plancher.
///
/// Un seul levier, monotone, et la somme reste à 1,0 par construction puisque
/// les deux jeux de poids somment à 1. L'alternative — remonter D3, D5, D8 à la
/// main et reprendre ailleurs — demande de choisir *où* reprendre, et ce choix
/// n'a aucun fondement.
fn appliquer_plancher(weights: AlgoWeights, base: &AlgoWeights) -> AlgoWeights {
    let part = weights.personalization_share();
    if part >= PERSONALIZATION_FLOOR {
        return weights;
    }
    let part_defaut = base.personalization_share();
    if part_defaut <= part {
        // Le défaut ne ferait pas mieux : rien à corriger de ce côté.
        return weights;
    }
    let t = ((PERSONALIZATION_FLOOR - part) / (part_defaut - part)).clamp(0.0, 1.0);
    warn!(
        part_apprise = part,
        plancher = PERSONALIZATION_FLOOR,
        retour_vers_defaut = t,
        "AutoTuner: poids appris trop dépersonnalisants — ramenés au plancher"
    );
    melanger(&weights, base, t)
}

// ─── Extension du CtrPredictor pour extraire les poids dimensionnels ─────────

impl CtrPredictor {
    /// Poids de dimensions dérivés du modèle CTR, rétrécis vers `base`.
    ///
    /// `None` si le modèle n'a pas assez appris, ou si aucune dimension n'a de
    /// coefficient positif — dans ce cas il n'y a aucun signal à propager et
    /// maquiller ça en distribution uniforme serait pire que ne rien faire.
    ///
    /// ⚠ **D9 et D10 ne sont jamais réécrites.** Elles ne font pas partie du
    /// vecteur de traits du modèle CTR : il n'a rien appris à leur sujet et n'a
    /// donc aucune légitimité à décider de leur part. Les huit dimensions
    /// apprises se partagent le budget restant.
    pub fn extract_dimension_weights(&self, base: &AlgoWeights) -> Option<AlgoWeights> {
        let (samples, _) = self.stats();
        if samples < MIN_SAMPLES_FOR_TUNING {
            return None;
        }

        let raw_weights = self.dimension_weights_snapshot();

        // Aucune dimension positive : le modèle n'a extrait aucun signal
        // discriminant. On refuse la mise à jour plutôt que de la maquiller.
        if raw_weights.iter().all(|&w| w <= 0.0) {
            warn!("AutoTuner: 8/8 poids de dimension non-positifs — aucun signal exploitable, poids inchangés");
            return None;
        }

        // Budget des huit dimensions apprises : tout sauf D9 et D10, préservées.
        let budget = (1.0 - base.d9_llm_understanding - base.d10_taste_affinity).max(0.0);
        let defauts = [
            base.d1_engagement_velocity,
            base.d2_content_intelligence,
            base.d3_social_graph,
            base.d4_temporal,
            base.d5_behavioral,
            base.d6_diversity,
            base.d7_viral,
            base.d8_personalization,
        ];
        let budget_defaut: f64 = defauts.iter().sum();
        if budget_defaut <= 0.0 || budget <= 0.0 {
            return None;
        }

        // ── Répartition ─────────────────────────────────────────────────────
        //
        // Une dimension dont le coefficient appris n'est pas positif GARDE sa
        // part par défaut : le modèle dit qu'elle nuit au CTR, ce qui n'est pas
        // une raison de la supprimer du classement — le CTR n'est pas le seul
        // objectif du fil, et un coefficient négatif sur un trait corrélé à un
        // autre ne veut de toute façon rien dire de fiable.
        //
        // Les dimensions positives se partagent ce qui reste, au prorata de
        // leur coefficient.
        let mut parts = [0.0f64; 8];
        let mut part_conservee = 0.0f64;
        let mut somme_positive = 0.0f64;
        for i in 0..8 {
            if raw_weights[i] > 0.0 {
                somme_positive += raw_weights[i];
            } else {
                parts[i] = defauts[i] / budget_defaut;
                part_conservee += parts[i];
            }
        }
        let reste = (1.0 - part_conservee).max(0.0);
        if somme_positive <= 0.0 {
            return None;
        }
        for i in 0..8 {
            if raw_weights[i] > 0.0 {
                parts[i] = raw_weights[i] / somme_positive * reste;
            }
        }

        let apprises = AlgoWeights {
            d1_engagement_velocity: parts[0] * budget,
            d2_content_intelligence: parts[1] * budget,
            d3_social_graph: parts[2] * budget,
            d4_temporal: parts[3] * budget,
            d5_behavioral: parts[4] * budget,
            d6_diversity: parts[5] * budget,
            d7_viral: parts[6] * budget,
            d8_personalization: parts[7] * budget,
            d9_llm_understanding: base.d9_llm_understanding,
            d10_taste_affinity: base.d10_taste_affinity,
        };

        // ── Rétrécissement, puis plancher ───────────────────────────────────
        let alpha = samples as f64 / (samples as f64 + CONFIANCE_DEMI);
        let melange = melanger(base, &apprises, alpha);
        let finales = appliquer_plancher(melange, base);

        debug!(
            samples,
            alpha,
            d1 = finales.d1_engagement_velocity,
            d3 = finales.d3_social_graph,
            d5 = finales.d5_behavioral,
            d8 = finales.d8_personalization,
            d10 = finales.d10_taste_affinity,
            part_perso = finales.personalization_share(),
            "CTR-derived dimension weights"
        );

        Some(finales)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Les coefficients relevés en production le 2026-08-31, reconstruits
    /// depuis les poids servis : D5 et D6 exactement au plancher, donc appris
    /// négatifs ; vélocité et viralité dominantes.
    fn coefficients_de_production() -> [f64; 8] {
        [0.26, 0.20, 0.09, 0.043, -0.02, -0.05, 0.21, 0.19]
    }

    fn poids_appris(coeffs: [f64; 8], samples: u64) -> AlgoWeights {
        let base = AlgoWeights::default();
        // Reproduction directe du calcul, sans passer par un CtrPredictor : ce
        // sont la répartition et les garde-fous qu'on teste, pas le modèle.
        let budget = 1.0 - base.d9_llm_understanding - base.d10_taste_affinity;
        let defauts = [
            base.d1_engagement_velocity,
            base.d2_content_intelligence,
            base.d3_social_graph,
            base.d4_temporal,
            base.d5_behavioral,
            base.d6_diversity,
            base.d7_viral,
            base.d8_personalization,
        ];
        let budget_defaut: f64 = defauts.iter().sum();
        let mut parts = [0.0f64; 8];
        let mut conservee = 0.0;
        let mut positive = 0.0;
        for i in 0..8 {
            if coeffs[i] > 0.0 {
                positive += coeffs[i];
            } else {
                parts[i] = defauts[i] / budget_defaut;
                conservee += parts[i];
            }
        }
        let reste = (1.0 - conservee).max(0.0);
        for i in 0..8 {
            if coeffs[i] > 0.0 {
                parts[i] = coeffs[i] / positive * reste;
            }
        }
        let apprises = AlgoWeights {
            d1_engagement_velocity: parts[0] * budget,
            d2_content_intelligence: parts[1] * budget,
            d3_social_graph: parts[2] * budget,
            d4_temporal: parts[3] * budget,
            d5_behavioral: parts[4] * budget,
            d6_diversity: parts[5] * budget,
            d7_viral: parts[6] * budget,
            d8_personalization: parts[7] * budget,
            d9_llm_understanding: base.d9_llm_understanding,
            d10_taste_affinity: base.d10_taste_affinity,
        };
        let alpha = samples as f64 / (samples as f64 + CONFIANCE_DEMI);
        appliquer_plancher(melanger(&base, &apprises, alpha), &base)
    }

    #[test]
    fn la_somme_reste_a_un() {
        for samples in [500u64, 9_219, 100_000, 10_000_000] {
            let w = poids_appris(coefficients_de_production(), samples);
            assert!(
                (w.sum() - 1.0).abs() < 1e-9,
                "samples={samples} somme={}",
                w.sum()
            );
        }
    }

    /// Le défaut exact qu'on corrige : D5 et D6 avaient des coefficients
    /// négatifs et se retrouvaient à 0,0008 — supprimées du classement.
    #[test]
    fn une_dimension_de_coefficient_negatif_n_est_plus_supprimee() {
        let base = AlgoWeights::default();
        let w = poids_appris(coefficients_de_production(), 9_219);
        assert!(
            w.d5_behavioral > base.d5_behavioral * 0.5,
            "D5 écrasée : {} (défaut {})",
            w.d5_behavioral,
            base.d5_behavioral
        );
        assert!(
            w.d6_diversity > base.d6_diversity * 0.5,
            "D6 écrasée : {} (défaut {})",
            w.d6_diversity,
            base.d6_diversity
        );
    }

    #[test]
    fn le_plancher_de_personnalisation_tient() {
        // Des coefficients qui ne récompensent QUE la popularité.
        let tout_populaire = [1.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0, 0.0001];
        for samples in [500u64, 9_219, 1_000_000, 100_000_000] {
            let w = poids_appris(tout_populaire, samples);
            assert!(
                w.personalization_share() >= PERSONALIZATION_FLOOR - 1e-9,
                "samples={samples} part={}",
                w.personalization_share()
            );
            assert!((w.sum() - 1.0).abs() < 1e-9);
        }
    }

    /// D9 et D10 ne sont pas dans le vecteur de traits du modèle : il n'a rien
    /// à en dire, et sa mise à jour ne doit pas les déplacer.
    #[test]
    fn d9_et_d10_ne_bougent_pas() {
        let base = AlgoWeights::default();
        for samples in [500u64, 9_219, 10_000_000] {
            let w = poids_appris(coefficients_de_production(), samples);
            assert!((w.d9_llm_understanding - base.d9_llm_understanding).abs() < 1e-9);
            assert!((w.d10_taste_affinity - base.d10_taste_affinity).abs() < 1e-9);
        }
    }

    /// Peu de données ⇒ l'estimation apprise ne doit presque rien déplacer.
    #[test]
    fn le_retrecissement_protege_les_petits_echantillons() {
        let base = AlgoWeights::default();
        let w = poids_appris(coefficients_de_production(), 500);
        let ecart = (w.d1_engagement_velocity - base.d1_engagement_velocity).abs();
        assert!(
            ecart < 0.02,
            "500 échantillons ne doivent presque rien changer, écart={ecart}"
        );
    }

    /// Beaucoup de données ⇒ le modèle doit tout de même pouvoir parler.
    ///
    /// Coefficients choisis pour ne PAS dépersonnaliser (D3 reste fort) : on
    /// teste ici le rétrécissement seul. Le plancher a son propre test, et son
    /// interaction avec celui-ci est vérifiée juste en dessous.
    #[test]
    fn avec_beaucoup_de_donnees_le_modele_pese() {
        let base = AlgoWeights::default();
        let coeffs = [1.0, 0.05, 0.60, 0.05, 0.20, 0.05, 0.05, 0.15];
        let w = poids_appris(coeffs, 500_000);
        assert!(
            w.d1_engagement_velocity > base.d1_engagement_velocity * 1.5,
            "le modèle doit pouvoir monter D1 : {} contre {}",
            w.d1_engagement_velocity,
            base.d1_engagement_velocity
        );
    }

    /// Le plancher BRIDE le modèle, et c'est voulu : des coefficients qui
    /// écrasent la personnalisation n'obtiennent qu'une partie de ce qu'ils
    /// demandent, même avec un demi-million d'échantillons.
    ///
    /// Ce test existe pour que l'arbitrage soit explicite plutôt que découvert
    /// un jour comme une anomalie : oui, le modèle est empêché d'aller au bout
    /// de ce qu'il a appris quand ce qu'il a appris est « montre à tout le
    /// monde la même chose ».
    #[test]
    fn le_plancher_bride_un_modele_depersonnalisant() {
        let base = AlgoWeights::default();
        let depersonnalisant = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        let bride = poids_appris(depersonnalisant, 500_000);
        let libre = poids_appris([1.0, 0.05, 0.60, 0.05, 0.20, 0.05, 0.05, 0.15], 500_000);

        assert!(
            bride.d1_engagement_velocity > base.d1_engagement_velocity,
            "D1 doit tout de même monter"
        );
        assert!(
            bride.d1_engagement_velocity < libre.d1_engagement_velocity,
            "le plancher doit limiter la montée : bridé={} libre={}",
            bride.d1_engagement_velocity,
            libre.d1_engagement_velocity
        );
        assert!(bride.personalization_share() >= PERSONALIZATION_FLOOR - 1e-9);
    }

    #[test]
    fn tous_les_poids_restent_positifs() {
        for coeffs in [
            coefficients_de_production(),
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ] {
            let w = poids_appris(coeffs, 50_000);
            for (i, v) in w.as_array().iter().enumerate() {
                assert!(*v >= 0.0, "dimension {} négative : {v}", i + 1);
            }
            assert!((w.sum() - 1.0).abs() < 1e-9);
        }
    }
}
