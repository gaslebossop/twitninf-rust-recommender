use serde::{Deserialize, Serialize};

use crate::shadowban::{ShadowbanLevel, StrikePolicy};

// ─── Requêtes admin ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SetShadowbanRequest {
    pub user_id: String,
    pub level: ShadowbanLevel,
    pub reason: Option<String>,
    /// Durée de la décision, en jours. Absent = sans terme.
    ///
    /// Une restriction sans terme est presque toujours une restriction qu'on
    /// oubliera de lever : les clés `admin:shadowban:*` n'avaient aucune
    /// expiration, donc chaque décision prise ici durait indéfiniment. Préférer
    /// une durée explicite, ou mieux, un avertissement (`/admin/strike`) qui
    /// expire seul au bout de 90 jours.
    #[serde(default)]
    pub expires_in_days: Option<u32>,
}

/// Émission d'un avertissement daté.
#[derive(Debug, Deserialize)]
pub struct IssueStrikeRequest {
    pub user_id: String,
    pub policy: StrikePolicy,
    /// Tweet à l'origine de l'avertissement — nécessaire pour qu'un recours
    /// puisse le révoquer précisément.
    #[serde(default)]
    pub tweet_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Recours accepté : on retire les avertissements liés à un tweet, ou tous.
#[derive(Debug, Deserialize)]
pub struct RevokeStrikeRequest {
    pub user_id: String,
    /// Absent = on vide entièrement le registre du compte.
    #[serde(default)]
    pub tweet_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BanRequest {
    pub user_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UnbanRequest {
    pub user_id: String,
}

/// `apply: false` (défaut) — reconstruit et rapporte, n'écrit rien sur disque.
/// `since_days` par défaut à 14 : la fenêtre convenue pour le rattrapage.
#[derive(Debug, Deserialize)]
pub struct BackfillCtrRequest {
    pub since_days: Option<i32>,
    pub apply: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SetWeightsRequest {
    pub d1: Option<f64>,
    pub d2: Option<f64>,
    pub d3: Option<f64>,
    pub d4: Option<f64>,
    pub d5: Option<f64>,
    pub d6: Option<f64>,
    pub d7: Option<f64>,
    pub d8: Option<f64>,
    pub d9: Option<f64>,
}

// ─── Réponses admin ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct AdminActionResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ShadowbannedUser {
    pub user_id: String,
    pub level: String,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct BannedUser {
    pub user_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FiltersResponse {
    pub shadowbanned: Vec<ShadowbannedUser>,
    pub hard_banned: Vec<BannedUser>,
    pub total_shadowbanned: usize,
    pub total_hard_banned: usize,
}

#[derive(Debug, Serialize, Clone, Deserialize)]
pub struct AlgoWeights {
    pub d1_engagement_velocity: f64,
    pub d2_content_intelligence: f64,
    pub d3_social_graph: f64,
    pub d4_temporal: f64,
    pub d5_behavioral: f64,
    pub d6_diversity: f64,
    pub d7_viral: f64,
    pub d8_personalization: f64,
    /// D9 — compréhension du contenu par le LLM annotateur.
    /// Absent des poids persistés avant son introduction : `serde(default)`
    /// évite qu'une config déjà enregistrée en Redis devienne illisible.
    #[serde(default = "default_d9")]
    pub d9_llm_understanding: f64,
}

fn default_d9() -> f64 {
    0.10
}

impl Default for AlgoWeights {
    fn default() -> Self {
        // D9 est financée par un prélèvement sur D1/D2 plutôt que par une
        // inflation du total : la somme reste à 1.0, sinon tous les scores
        // dérivent vers le haut et les seuils calibrés ailleurs sautent.
        // D3 passe de 0,15 à 0,22, prélevés sur D1, D2, D4 et D7 : la somme
        // reste à 1,0. Un abonnement pesait environ +0,05 sur un score borné à
        // 1 — invisible face à D1. Le multiplicateur FOLLOW_FEED_BOOST fait le
        // reste du travail au moment du classement.
        Self {
            d1_engagement_velocity: 0.24,
            d2_content_intelligence: 0.12,
            d3_social_graph: 0.22,
            d4_temporal: 0.09,
            d5_behavioral: 0.08,
            d6_diversity: 0.06,
            d7_viral: 0.06,
            d8_personalization: 0.03,
            d9_llm_understanding: 0.10,
        }
    }
}

impl AlgoWeights {
    pub fn as_array(&self) -> [f64; 9] {
        [
            self.d1_engagement_velocity,
            self.d2_content_intelligence,
            self.d3_social_graph,
            self.d4_temporal,
            self.d5_behavioral,
            self.d6_diversity,
            self.d7_viral,
            self.d8_personalization,
            self.d9_llm_understanding,
        ]
    }
}

#[derive(Debug, Serialize)]
pub struct AlgoWeightsResponse {
    pub weights: AlgoWeights,
    pub auto_tuned: bool,
    pub ctr_samples: u64,
    pub global_ctr: f64,
}

#[derive(Debug, Serialize)]
pub struct AlgoStatsResponse {
    pub ctr_samples: u64,
    pub global_ctr: f64,
    pub weights: AlgoWeights,
    pub auto_tuned: bool,
    pub ml_active: bool,
    /// Échantillons appris par `ml::dwell_predictor` et poids de dwell moyen
    /// observé sur son échelle d'origine (voir `algorithm::dwell`) — même
    /// paire (samples, moyenne) que `ctr_samples`/`global_ctr`, pour l'autre
    /// modèle.
    pub dwell_samples: u64,
    pub dwell_mean_weight: f64,
    pub dwell_active: bool,
    pub algorithm_version: &'static str,
}
