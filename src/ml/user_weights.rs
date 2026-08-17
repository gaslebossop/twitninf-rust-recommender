/// Phase 2 — User-Specific Dimension Weights
///
/// Chaque segment utilisateur a des poids optimisés pour maximiser son CTR.
/// PowerUser : veut du viral + découverte
/// Regular   : balance engagement + social
/// Casual    : contenu frais + réseau proche
use serde::{Deserialize, Serialize};
use tracing::trace;

use crate::models::{PersonalityType, UserProfile, UserType};

/// Poids par dimension pour un segment utilisateur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDimensionWeights {
    pub d1: f64, // Engagement Velocity
    pub d2: f64, // Content Intelligence
    pub d3: f64, // Social Graph
    pub d4: f64, // Temporal
    pub d5: f64, // Behavioral
    pub d6: f64, // Diversity
    pub d7: f64, // Viral
    pub d8: f64, // Personalization
}

impl UserDimensionWeights {
    /// Poids calibrés par segment selon analyse CTR historique
    pub fn for_profile(profile: &UserProfile) -> Self {
        let base = Self::for_user_type(&profile.user_type);
        // Ajustement personnalité : Curious → plus de diversité, Enthusiastic → plus viral
        Self::adjust_for_personality(base, &profile.personality_type)
    }

    fn for_user_type(user_type: &UserType) -> Self {
        match user_type {
            // PowerUser: fort engagement, découverte, viral (+CTR sur contenus trending)
            UserType::PowerUser => Self {
                d1: 0.35,
                d2: 0.14,
                d3: 0.12,
                d4: 0.10,
                d5: 0.09,
                d6: 0.05,
                d7: 0.11,
                d8: 0.04,
            },
            // Regular: poids Phase 1 par défaut (bien calibrés)
            UserType::Regular => Self {
                d1: 0.32,
                d2: 0.18,
                d3: 0.15,
                d4: 0.10,
                d5: 0.08,
                d6: 0.06,
                d7: 0.07,
                d8: 0.04,
            },
            // Casual: réseau social fort + fraîcheur (moins de pression virale)
            UserType::Casual => Self {
                d1: 0.25,
                d2: 0.20,
                d3: 0.20,
                d4: 0.13,
                d5: 0.08,
                d6: 0.06,
                d7: 0.05,
                d8: 0.03,
            },
        }
    }

    fn adjust_for_personality(mut w: Self, personality: &PersonalityType) -> Self {
        match personality {
            PersonalityType::Enthusiastic => {
                // Aime le viral et l'engagement fort
                w.d7 = (w.d7 + 0.02).min(0.15);
                w.d6 = (w.d6 - 0.01).max(0.03);
                w.normalize();
            }
            PersonalityType::Curious => {
                // Aime découvrir du contenu nouveau
                w.d6 = (w.d6 + 0.02).min(0.10);
                w.d8 = (w.d8 + 0.01).min(0.08);
                w.d1 = (w.d1 - 0.03).max(0.20);
                w.normalize();
            }
            PersonalityType::Thoughtful => {
                // Préfère contenu de qualité et réseau de confiance
                w.d2 = (w.d2 + 0.02).min(0.25);
                w.d3 = (w.d3 + 0.01).min(0.20);
                w.d7 = (w.d7 - 0.03).max(0.03);
                w.normalize();
            }
            PersonalityType::Balanced => {} // pas d'ajustement
        }
        w
    }

    /// Normalise pour que la somme = 1.0
    fn normalize(&mut self) {
        let sum = self.d1 + self.d2 + self.d3 + self.d4 + self.d5 + self.d6 + self.d7 + self.d8;
        if sum > 0.0 {
            self.d1 /= sum;
            self.d2 /= sum;
            self.d3 /= sum;
            self.d4 /= sum;
            self.d5 /= sum;
            self.d6 /= sum;
            self.d7 /= sum;
            self.d8 /= sum;
        }
    }

    /// Applique les poids personnalisés aux scores de dimensions
    pub fn apply(
        &self,
        d1: f64,
        d2: f64,
        d3: f64,
        d4: f64,
        d5: f64,
        d6: f64,
        d7: f64,
        d8: f64,
    ) -> f64 {
        let score = d1 * self.d1
            + d2 * self.d2
            + d3 * self.d3
            + d4 * self.d4
            + d5 * self.d5
            + d6 * self.d6
            + d7 * self.d7
            + d8 * self.d8;
        trace!(
            score,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            w1 = self.d1,
            w2 = self.d2,
            w3 = self.d3,
            w4 = self.d4,
            w5 = self.d5,
            w6 = self.d6,
            w7 = self.d7,
            w8 = self.d8,
            "User-specific weighted score"
        );
        score.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_sum_to_one() {
        for ut in [UserType::PowerUser, UserType::Regular, UserType::Casual] {
            let w = UserDimensionWeights::for_user_type(&ut);
            let sum = w.d1 + w.d2 + w.d3 + w.d4 + w.d5 + w.d6 + w.d7 + w.d8;
            assert!((sum - 1.0).abs() < 0.001, "{ut:?} weights sum {sum} ≠ 1.0");
        }
    }

    #[test]
    fn test_poweruser_has_higher_d1() {
        let power = UserDimensionWeights::for_user_type(&UserType::PowerUser);
        let casual = UserDimensionWeights::for_user_type(&UserType::Casual);
        assert!(
            power.d1 > casual.d1,
            "PowerUser should weight D1 more than Casual"
        );
    }

    #[test]
    fn test_casual_has_higher_d3() {
        let power = UserDimensionWeights::for_user_type(&UserType::PowerUser);
        let casual = UserDimensionWeights::for_user_type(&UserType::Casual);
        assert!(
            casual.d3 > power.d3,
            "Casual should weight social graph more"
        );
    }
}
