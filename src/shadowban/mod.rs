//! Restriction de distribution — modèle à deux plans.
//!
//! - **Contenu** ([`eligibility`]) : ce post-là n'est plus recommandé. Il reste en
//!   ligne, sur le profil et dans le fil des abonnés.
//! - **Compte** ([`strikes`]) : conséquence courante d'avertissements datés qui
//!   expirent seuls au bout de 90 jours, avec des seuils propres à chaque domaine
//!   de règle.
//!
//! [`enforcer`] applique les deux, surface par surface, avec un invariant :
//! un lecteur ne perd jamais un compte qu'il suit.
//!
//! Sources du modèle (système d'application de TikTok, refonte 2023) :
//! - <https://www.tiktok.com/community-guidelines/en/fyf-standards> — inéligibilité
//!   au fil de recommandation, distincte du retrait.
//! - <https://newsroom.tiktok.com/en-us/supporting-creators-with-an-updated-account-enforcement-system>
//!   — avertissements expirant à 90 jours, seuils par domaine et par
//!   fonctionnalité, page « état du compte », alerte avant le point de non-retour.

pub mod detector;
pub mod eligibility;
pub mod enforcer;
pub mod models;
pub mod store;
pub mod strikes;

pub use detector::GarbageContentDetector;
pub use eligibility::content_eligibility;
pub use enforcer::{ShadowbanEnforcer, Verdict};
pub use models::{
    AccountQualityScore, ContentEligibility, GarbageSignals, IneligibilityReason, ShadowbanLevel,
    Strike, StrikePolicy, Surface, STRIKE_TTL_DAYS,
};
pub use strikes::{AccountStatus, StrikeLedger};
