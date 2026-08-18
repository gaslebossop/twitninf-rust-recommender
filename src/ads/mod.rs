//! Publicité — ciblage par les signaux de l'algorithme.
//!
//! Remplace un module entier (`models.rs` + `targeting.rs`, ~600 lignes :
//! enchères CPM/CPC/CPA, segments d'audience, score de valeur à vie, risque
//! de départ) qui n'était déclaré nulle part et n'a donc jamais compilé ni
//! tourné. Il décrivait une régie publicitaire complète pour une plateforme
//! qui n'avait pas encore servi sa première publicité.
//!
//! Ce qui le remplace tient dans un seul fichier et est branché au fil réel.
pub mod serving;

pub use serving::{select_for_feed, AdPlacement};
