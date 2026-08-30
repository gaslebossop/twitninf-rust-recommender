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

pub mod admin; // Noeud admin : bans, shadowbans, contrôle algo
pub mod ads; // Targeted advertising
pub mod algorithm;
pub mod bandit; // Phase 3: Contextual Bandit
pub mod calibration; // Recalibration explicite de l'algo, depuis les Paramètres
pub mod collab; // Plongements collaboratifs : lecteurs et auteurs dans le meme espace
pub mod constants;
pub mod cooccurrence;
pub mod embeddings;
pub mod eval; // Mesure hors-echantillon des modeles : AUC, log-loss, calibration, NDCG
pub mod experiments;
pub mod handlers;
pub mod middleware;
pub mod ml; // Phase 2: ML CTR Predictor + User Weights + AutoTuner
pub mod models;
pub mod neural; // Client du service taste-model (modele neuronal entraine en continu)
pub mod services;
pub mod shadowban; // Suppression des comptes contenu poubelle
pub mod utils;
pub mod velocity; // Frein temporaire automatique (1h), distinct du shadowban

// Re-export common types
pub use models::{RawTweet, RecommendRequest, RecommendResponse, UserProfile};

// Feature flags for optional functionality

// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
