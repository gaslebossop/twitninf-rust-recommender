/// Individual dimension scoring modules
/// Each dimension is isolated for clarity, testability, and maintainability

pub mod d1_engagement_velocity;
pub mod d2_content_intelligence;
pub mod d3_social_graph;
pub mod d4_temporal_dynamics;
pub mod d5_behavioral_prediction;
pub mod d6_content_diversity;
pub mod d7_viral_prediction;
pub mod d8_personalization_depth;

pub use d1_engagement_velocity::calculate as calculate_d1;
pub use d2_content_intelligence::calculate as calculate_d2;
pub use d3_social_graph::calculate as calculate_d3;
pub use d4_temporal_dynamics::calculate as calculate_d4;
pub use d5_behavioral_prediction::calculate as calculate_d5;
pub use d6_content_diversity::calculate as calculate_d6;
pub use d7_viral_prediction::calculate as calculate_d7;
pub use d8_personalization_depth::calculate as calculate_d8;
