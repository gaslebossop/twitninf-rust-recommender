// Mathematical utility functions for the NeuralRank algorithm

/// Sigmoid activation function: σ(x) = 1 / (1 + e^-x)
/// Maps any real number to (0, 1) range, centered at 0.5
#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Gaussian/Normal distribution: exp(-0.5 * ((x - μ) / σ)²)
/// Returns highest value (1.0) at center μ, falls off with width σ
#[inline(always)]
pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp()
}

/// Exponential decay with half-life
/// decay_rate = ln(2) / half_life
/// After half_life hours, result = 0.5
#[inline(always)]
pub fn exponential_decay(time: f64, decay_rate: f64) -> f64 {
    (-decay_rate * time).exp()
}

/// Linear interpolation between a and b
#[inline(always)]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Smooth step function (Hermite interpolation)
/// Returns 0 at x=0, 1 at x=1, smooth curve between
#[inline(always)]
pub fn smoothstep(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Clamp value to [min, max] range
#[inline(always)]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Natural logarithm with safety floor to avoid -inf
#[inline(always)]
pub fn safe_ln(x: f64) -> f64 {
    x.ln().max(0.0)
}

/// Weighted average with validation
pub fn weighted_avg(values: &[(f64, f64)]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let sum_weighted: f64 = values.iter().map(|(v, w)| v * w).sum();
    let sum_weights: f64 = values.iter().map(|(_, w)| w).sum();

    if sum_weights == 0.0 {
        0.0
    } else {
        sum_weighted / sum_weights
    }
}

/// Normalize values to [0, 1] range
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
    values.iter().map(|v| v / max_val).collect()
}

/// Ratio with safe division (prevents divide by zero)
#[inline(always)]
pub fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, f64::INFINITY)
    }
}

/// Calculate sigmoid with custom scale parameter
#[inline(always)]
pub fn sigmoid_scaled(x: f64, scale: f64) -> f64 {
    sigmoid(x / scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gaussian() {
        assert!((gaussian(0.0, 0.0, 1.0) - 1.0).abs() < 1e-10); // peak at center
        assert!(gaussian(1.0, 0.0, 1.0) < gaussian(0.5, 0.0, 1.0)); // decreases away
    }

    #[test]
    fn test_exponential_decay() {
        assert!((exponential_decay(0.0, 0.115) - 1.0).abs() < 1e-10); // 1.0 at t=0
        assert!(exponential_decay(6.0, 0.115) - 0.5 < 0.01); // ~0.5 at half-life
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_weighted_avg() {
        let values = vec![(10.0, 1.0), (20.0, 1.0)];
        assert!((weighted_avg(&values) - 15.0).abs() < 1e-10);

        let weighted = vec![(10.0, 1.0), (20.0, 3.0)];
        assert!((weighted_avg(&weighted) - 17.5).abs() < 1e-10);
    }
}
