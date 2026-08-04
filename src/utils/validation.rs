use crate::constants::*;
use crate::error::{AppError, AppResult};
use uuid::Uuid;

/// Validate user ID format (must be valid UUID)
pub fn validate_user_id(user_id: &str) -> AppResult<()> {
    if user_id.len() < MIN_USER_ID_LENGTH {
        return Err(AppError::InvalidInput(
            "User ID too short".to_string(),
        ));
    }

    Uuid::parse_str(user_id).map_err(|_| {
        AppError::InvalidInput("Invalid user ID format (must be UUID)".to_string())
    })?;

    Ok(())
}

/// Validate pagination parameters.
/// `limit` and `offset` are i64 (signed) pour permettre la détection de valeurs négatives
/// reçues d'un appelant externe avant conversion.
pub fn validate_pagination(limit: Option<i64>, offset: Option<i64>) -> AppResult<(usize, usize)> {
    let limit  = limit.unwrap_or(API_DEFAULT_LIMIT as i64);
    let offset = offset.unwrap_or(API_DEFAULT_OFFSET as i64);

    if offset < 0 {
        return Err(AppError::InvalidInput("Offset cannot be negative".to_string()));
    }

    // Une limite trop haute est plafonnée, pas rejetée — c'est déjà ce que fait
    // `recommend()` (`.clamp(1, 200)`) et ce que ce test attend. La version
    // précédente renvoyait une erreur : les deux comportements coexistaient
    // pour la même règle métier.
    if limit < API_MIN_LIMIT as i64 {
        return Err(AppError::InvalidInput(
            format!("Limit must be at least {}", API_MIN_LIMIT),
        ));
    }
    let limit_usize = (limit as usize).min(API_MAX_LIMIT);

    Ok((limit_usize, offset as usize))
}

/// Validate score is in valid range [0, 1]
pub fn validate_score(score: f64) -> AppResult<f64> {
    if !score.is_finite() {
        return Err(AppError::InvalidInput("Score is not a valid number".to_string()));
    }

    if score < MIN_SCORE || score > MAX_SCORE {
        return Err(AppError::InvalidInput(
            format!("Score must be between {} and {}", MIN_SCORE, MAX_SCORE),
        ));
    }

    Ok(score)
}

/// Validate content length
pub fn validate_content_length(length: usize) -> AppResult<()> {
    if length > MAX_CONTENT_LENGTH {
        return Err(AppError::InvalidInput(
            format!("Content too long. Max {} characters", MAX_CONTENT_LENGTH),
        ));
    }
    Ok(())
}

/// Validate engagement counts (should be non-negative)
pub fn validate_engagement_count(count: i64, field_name: &str) -> AppResult<()> {
    if count < 0 {
        return Err(AppError::InvalidInput(
            format!("{} cannot be negative", field_name),
        ));
    }
    Ok(())
}

/// Check if all required fields are present
pub fn validate_required_fields(fields: &[(&str, bool)]) -> AppResult<()> {
    for (field_name, present) in fields {
        if !present {
            return Err(AppError::MissingField(field_name.to_string()));
        }
    }
    Ok(())
}

/// Sanitize string input (trim and validate not empty)
pub fn sanitize_string(input: &str, field_name: &str) -> AppResult<String> {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return Err(AppError::InvalidInput(
            format!("{} cannot be empty", field_name),
        ));
    }

    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_user_id() {
        // Valid UUID
        let valid = "550e8400-e29b-41d4-a716-446655440000";
        assert!(validate_user_id(valid).is_ok());

        // Invalid format
        let invalid = "not-a-uuid";
        assert!(validate_user_id(invalid).is_err());

        // Too short
        let short = "123";
        assert!(validate_user_id(short).is_err());
    }

    #[test]
    fn test_validate_pagination() {
        let (limit, offset) = validate_pagination(Some(50), Some(0)).unwrap();
        assert_eq!(limit, 50);
        assert_eq!(offset, 0);

        // Should clamp to max
        let (limit, _) = validate_pagination(Some(500), None).unwrap();
        assert_eq!(limit, API_MAX_LIMIT);

        // Invalid limit
        assert!(validate_pagination(Some(0), None).is_err());
    }

    #[test]
    fn test_validate_score() {
        assert!(validate_score(0.5).is_ok());
        assert!(validate_score(0.0).is_ok());
        assert!(validate_score(1.0).is_ok());

        assert!(validate_score(-0.1).is_err());
        assert!(validate_score(1.1).is_err());
    }
}
