//! TwitNinf Rust Recommender — NeuralRank Fusion Engine
//!
//! A high-performance recommendation engine with 8 dimensions of real-time scoring.
//!
//! # Architecture
//!
//! The recommender pipeline:
//! 1. **Profile Building** → Load user data from database
//! 2. **Candidate Collection** → Parallel queries from 8 sources
//! 3. **Deduplication** → Keep highest weight for duplicates
//! 4. **Scoring** → Apply 8 dimension scoring + modifiers
//! 5. **Ranking** → Sort by final score
//! 6. **Response** → Paginated results with metadata

pub mod algorithm;
pub mod constants;
pub mod error;
pub mod handlers;
pub mod models;
pub mod services;
pub mod utils;
pub mod middleware;

// Re-export common types
pub use error::{AppError, AppResult};
pub use models::{RecommendRequest, RecommendResponse, RawTweet, UserProfile};

// Feature flags for optional functionality

// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
