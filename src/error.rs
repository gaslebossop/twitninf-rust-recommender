use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::fmt;

#[derive(Debug)]
pub enum AppError {
    // Database errors
    DatabaseError(String),
    DatabasePool(String),

    // Cache errors
    CacheError(String),

    // Validation errors
    InvalidInput(String),
    MissingField(String),

    // Service errors
    RecommendationFailed(String),
    ProfileBuildFailed(String),
    ScoringFailed(String),

    // System errors
    ConfigError(String),
    InternalError(String),
    Timeout(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            AppError::DatabasePool(msg) => write!(f, "Database pool error: {}", msg),
            AppError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            AppError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AppError::MissingField(field) => write!(f, "Missing field: {}", field),
            AppError::RecommendationFailed(msg) => write!(f, "Recommendation failed: {}", msg),
            AppError::ProfileBuildFailed(msg) => write!(f, "Profile build failed: {}", msg),
            AppError::ScoringFailed(msg) => write!(f, "Scoring failed: {}", msg),
            AppError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            AppError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            AppError::Timeout(msg) => write!(f, "Timeout: {}", msg),
        }
    }
}

impl AppError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            AppError::InvalidInput(_) | AppError::MissingField(_) => StatusCode::BAD_REQUEST,
            AppError::DatabasePool(_) | AppError::CacheError(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            AppError::DatabaseError(_) => "DB_ERROR",
            AppError::DatabasePool(_) => "DB_POOL_ERROR",
            AppError::CacheError(_) => "CACHE_ERROR",
            AppError::InvalidInput(_) => "INVALID_INPUT",
            AppError::MissingField(_) => "MISSING_FIELD",
            AppError::RecommendationFailed(_) => "RECOMMENDATION_FAILED",
            AppError::ProfileBuildFailed(_) => "PROFILE_BUILD_FAILED",
            AppError::ScoringFailed(_) => "SCORING_FAILED",
            AppError::ConfigError(_) => "CONFIG_ERROR",
            AppError::InternalError(_) => "INTERNAL_ERROR",
            AppError::Timeout(_) => "TIMEOUT",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_code = self.error_code();
        let message = self.to_string();

        (
            status,
            Json(json!({
                "success": false,
                "error": {
                    "code": error_code,
                    "message": message,
                }
            })),
        )
            .into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
