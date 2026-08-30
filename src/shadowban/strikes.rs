//! Registre d'avertissements à expiration — le cœur du modèle.
//!
//! Une restriction de compte n'est plus un drapeau posé une fois pour toutes :
//! c'est la **conséquence courante** d'avertissements datés qui expirent seuls au
//! bout de [`STRIKE_TTL_DAYS`] jours. Rien à lever à la main, rien à oublier de
//! lever. Un compte qui arrête d'enfreindre remonte de lui-même, avertissement
//! par avertissement, et l'on peut dire au créateur **à quelle date**.
//!
//! Trois propriétés reprises du système d'application de TikTok :
//!
//! 1. **Expiration à 90 jours** — un avertissement « n'est plus pris en compte »
//!    passé ce délai.
//! 2. **Seuils par domaine** — on franchit le seuil *d'un domaine*, pas d'un
//!    total mélangé, et les seuils sont plus stricts là où la nuisance est plus
//!    grande.
//! 3. **Avertissement avant le point de non-retour** — le compte est prévenu
//!    quand il approche du bannissement définitif, au lieu de le découvrir.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::models::{strike_ttl, ShadowbanLevel, Strike, StrikePolicy, Surface, STRIKE_TTL_DAYS};

/// À partir de quelle fraction du seuil de bannissement on prévient le compte.
const NEARING_BAN_RATIO: f64 = 0.75;

// ─── Registre ─────────────────────────────────────────────────────────────────

/// Tous les avertissements connus d'un compte, plus l'éventuelle décision
/// manuelle de l'équipe de modération.
#[derive(Debug, Clone, Default)]
pub struct StrikeLedger {
    pub user_id: String,
    pub strikes: Vec<Strike>,
    /// Décision humaine. Elle prime toujours sur le calcul automatique — dans les
    /// deux sens : elle peut durcir comme elle peut blanchir un compte que
    /// l'heuristique a mal jugé.
    pub manual_override: Option<ShadowbanLevel>,
    /// Fin de la décision manuelle. `None` = sans terme (réservé aux décisions
    /// prises en connaissance de cause).
    pub manual_expires_at: Option<DateTime<Utc>>,
}

impl StrikeLedger {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Avertissements encore comptabilisés à l'instant `now`.
    pub fn active(&self, now: DateTime<Utc>) -> impl Iterator<Item = &Strike> {
        self.strikes.iter().filter(move |s| s.is_active(now))
    }

    pub fn active_count(&self, now: DateTime<Utc>) -> u32 {
        self.active(now).count() as u32
    }

    pub fn count_for(&self, policy: StrikePolicy, now: DateTime<Utc>) -> u32 {
        self.active(now).filter(|s| s.policy == policy).count() as u32
    }

    /// Décision manuelle encore valable, le cas échéant.
    fn active_override(&self, now: DateTime<Utc>) -> Option<ShadowbanLevel> {
        let level = self.manual_override?;
        match self.manual_expires_at {
            Some(exp) if exp <= now => None,
            _ => Some(level),
        }
    }

    /// Niveau issu des seuls avertissements — le plus sévère parmi les domaines.
    ///
    /// C'est un *max*, pas une somme : un compte qui accumule du spam à faible
    /// nuisance ne doit pas franchir par addition un seuil réservé au harcèlement,
    /// et un unique fait grave ne doit pas être dilué par une majorité de faits
    /// bénins.
    pub fn computed_level(&self, now: DateTime<Utc>) -> ShadowbanLevel {
        StrikePolicy::ALL
            .iter()
            .map(|&p| p.level_for(self.count_for(p, now)))
            .max()
            .unwrap_or(ShadowbanLevel::Clean)
    }

    /// Niveau effectif appliqué au scoring.
    pub fn level(&self, now: DateTime<Utc>) -> ShadowbanLevel {
        self.active_override(now)
            .unwrap_or_else(|| self.computed_level(now))
    }

    /// Vrai si un domaine a franchi son seuil de bannissement définitif.
    ///
    /// Volontairement **consultatif** : la fonction signale, elle ne bannit pas.
    /// Un bannissement définitif est irréversible et déclenché ici par des
    /// heuristiques (densité de hashtags, taux de signalement) qui se trompent —
    /// il reste une décision humaine, via `/admin/ban`.
    pub fn permanent_ban_reached(&self, now: DateTime<Utc>) -> Option<StrikePolicy> {
        StrikePolicy::ALL
            .iter()
            .copied()
            .find(|&p| self.count_for(p, now) >= p.thresholds().permanent)
    }

    /// Domaine où le compte approche du bannissement, avec le compte courant et
    /// le seuil. TikTok prévient explicitement les créateurs dans ce cas plutôt
    /// que de les laisser le découvrir : sans avertissement, la sanction
    /// n'enseigne rien.
    pub fn nearing_permanent_ban(&self, now: DateTime<Utc>) -> Option<(StrikePolicy, u32, u32)> {
        StrikePolicy::ALL.iter().copied().find_map(|p| {
            let count = self.count_for(p, now);
            let limit = p.thresholds().permanent;
            if count == 0 || count >= limit {
                return None;
            }
            let ratio = count as f64 / limit as f64;
            (ratio >= NEARING_BAN_RATIO).then_some((p, count, limit))
        })
    }

    /// Date à laquelle le niveau baissera, si rien d'autre n'est ajouté.
    ///
    /// C'est l'expiration du plus ancien avertissement encore actif dans le
    /// domaine qui détermine le niveau courant : c'est lui qui, en tombant, fait
    /// repasser le compte sous le seuil.
    pub fn level_expires_at(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        if self.active_override(now).is_some() {
            return self.manual_expires_at;
        }
        let current = self.computed_level(now);
        if current == ShadowbanLevel::Clean {
            return None;
        }
        StrikePolicy::ALL
            .iter()
            .copied()
            .filter(|&p| p.level_for(self.count_for(p, now)) == current)
            .filter_map(|p| {
                self.active(now)
                    .filter(|s| s.policy == p)
                    .map(|s| s.expires_at())
                    .min()
            })
            .min()
    }

    /// Durée de vie utile du niveau en cache, en secondes.
    ///
    /// Bornée par l'expiration du prochain avertissement : au-delà, la valeur en
    /// cache serait plus sévère que la réalité. C'est ce qui rend l'expiration
    /// automatique sans tâche périodique — la clé Redis meurt d'elle-même au bon
    /// moment.
    pub fn cache_ttl_secs(&self, now: DateTime<Utc>) -> u64 {
        match self.level_expires_at(now) {
            Some(exp) => (exp - now)
                .num_seconds()
                .clamp(60, strike_ttl().num_seconds()) as u64,
            None => 3600,
        }
    }

    /// État lisible par le créateur concerné.
    pub fn status(&self, now: DateTime<Utc>) -> AccountStatus {
        let level = self.level(now);
        let mut per_policy: Vec<PolicyStanding> = StrikePolicy::ALL
            .iter()
            .copied()
            .filter_map(|p| {
                let count = self.count_for(p, now);
                (count > 0).then(|| PolicyStanding {
                    policy: p,
                    label: p.label(),
                    reason: p.creator_reason(),
                    active_strikes: count,
                    permanent_limit: p.thresholds().permanent,
                    next_expiry: self
                        .active(now)
                        .filter(|s| s.policy == p)
                        .map(|s| s.expires_at())
                        .min(),
                })
            })
            .collect();
        per_policy.sort_by(|a, b| b.active_strikes.cmp(&a.active_strikes));

        let restricted_surfaces = [
            Surface::ForYou,
            Surface::Discover,
            Surface::Trending,
            Surface::FollowerFeed,
        ]
        .into_iter()
        .filter(|&s| level.restricts(s))
        .map(|s| s.label())
        .collect();

        AccountStatus {
            user_id: self.user_id.clone(),
            level,
            level_label: level.label(),
            summary: level.creator_summary(),
            manual: self.active_override(now).is_some(),
            active_strikes: self.active_count(now),
            strike_ttl_days: STRIKE_TTL_DAYS,
            restricted_surfaces,
            recovers_at: self.level_expires_at(now),
            nearing_permanent_ban: self.nearing_permanent_ban(now).map(|(p, c, l)| BanWarning {
                policy: p,
                reason: p.creator_reason(),
                active_strikes: c,
                limit: l,
            }),
            per_policy,
        }
    }
}

// ─── État de compte (destiné au créateur) ────────────────────────────────────

/// Ce qu'un compte peut lire sur lui-même.
///
/// La refonte de TikTok tenait moins aux sanctions qu'à leur lisibilité : une
/// page « État du compte » a remplacé un fonctionnement que les créateurs
/// décrivaient comme opaque. Une suppression de portée que personne ne peut voir
/// ni dater ne corrige rien — elle produit juste des créateurs qui devinent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountStatus {
    pub user_id: String,
    pub level: ShadowbanLevel,
    pub level_label: &'static str,
    pub summary: &'static str,
    /// Vrai si l'état vient d'une décision humaine et non du calcul.
    pub manual: bool,
    pub active_strikes: u32,
    pub strike_ttl_days: i64,
    /// Surfaces actuellement fermées, nommées.
    pub restricted_surfaces: Vec<&'static str>,
    /// Date à laquelle la restriction s'allège si rien ne s'ajoute.
    pub recovers_at: Option<DateTime<Utc>>,
    pub nearing_permanent_ban: Option<BanWarning>,
    pub per_policy: Vec<PolicyStanding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyStanding {
    pub policy: StrikePolicy,
    pub label: &'static str,
    pub reason: &'static str,
    pub active_strikes: u32,
    pub permanent_limit: u32,
    pub next_expiry: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanWarning {
    pub policy: StrikePolicy,
    pub reason: &'static str,
    pub active_strikes: u32,
    pub limit: u32,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn strike_at(policy: StrikePolicy, days_ago: i64, now: DateTime<Utc>) -> Strike {
        Strike {
            policy,
            issued_at: now - Duration::days(days_ago),
            tweet_id: None,
            reason: None,
        }
    }

    fn ledger(policy: StrikePolicy, ages: &[i64], now: DateTime<Utc>) -> StrikeLedger {
        StrikeLedger {
            user_id: "u1".into(),
            strikes: ages.iter().map(|&d| strike_at(policy, d, now)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn compte_neuf_est_clean() {
        let now = Utc::now();
        assert_eq!(StrikeLedger::new("u1").level(now), ShadowbanLevel::Clean);
    }

    #[test]
    fn avertissement_expire_apres_90_jours() {
        let now = Utc::now();
        // 2 spams suffisent à Suppressed depuis le resserrage du 2026-08-22…
        let recent = ledger(StrikePolicy::Spam, &[1, 2], now);
        assert_eq!(recent.level(now), ShadowbanLevel::Suppressed);

        // …mais plus rien passé la fenêtre : le compte remonte seul.
        let old = ledger(StrikePolicy::Spam, &[91, 92, 100, 200], now);
        assert_eq!(old.active_count(now), 0);
        assert_eq!(old.level(now), ShadowbanLevel::Clean);
    }

    #[test]
    fn seuils_plus_stricts_selon_la_nuisance() {
        let now = Utc::now();
        // ⚠ Depuis le resserrage du 2026-08-22, spam et haine ferment leurs
        // surfaces AU MEME RYTHME (2 faits -> Suppressed, 3 -> Ghosted). La
        // graduation par nuisance ne subsiste plus que sur le bannissement
        // definitif : 8 faits pour le spam, 4 pour la haine. C'est une
        // consequence assumee de la demande « au bout de 2 on doit deja y
        // etre » — la noter ici plutot que la laisser se decouvrir en prod.
        assert_eq!(
            ledger(StrikePolicy::Spam, &[1, 2], now).level(now),
            ShadowbanLevel::Suppressed
        );
        assert_eq!(
            ledger(StrikePolicy::HatefulConduct, &[1, 2], now).level(now),
            ShadowbanLevel::Suppressed
        );
        // Ce qui reste gradue : le point de non-retour.
        assert!(
            StrikePolicy::HatefulConduct.thresholds().permanent
                < StrikePolicy::Spam.thresholds().permanent
        );
        // Et « repris sans apport » reste plus indulgent que « spam ».
        assert!(
            StrikePolicy::Unoriginal.thresholds().suppressed
                > StrikePolicy::Spam.thresholds().suppressed
        );
    }

    /// Le comportement demandé le 2026-08-22 : hors « Pour toi » ET hors
    /// Explorer une fois le palier atteint. C'est `Ghosted` qui le fait — la
    /// définition des niveaux n'a pas bougé, seuls les seuils ont baissé.
    #[test]
    fn trois_avertissements_spam_ferment_pour_toi_et_explorer() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::Spam, &[1, 2, 3], now);
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
        assert!(l.level(now).restricts(Surface::ForYou));
        assert!(l.level(now).restricts(Surface::Trending));
        assert!(l.level(now).restricts(Surface::Discover));
        // Ghosted ferme désormais jusqu'au fil d'abonnement (2026-08-30).
        assert!(l.level(now).restricts(Surface::FollowerFeed));
    }

    #[test]
    fn nuisance_immediate_coupe_des_le_premier_fait() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::ViolentThreat, &[1], now);
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
        assert!(l.level(now).restricts(Surface::ForYou));
    }

    #[test]
    fn les_domaines_ne_s_additionnent_pas() {
        let now = Utc::now();
        // 3 domaines à 1 fait chacun : aucun ne franchit son propre seuil de
        // SUPPRESSION, donc aucune surface ne se ferme. C'est tout l'objet du
        // test : additionnés, ces 3 faits vaudraient Ghosted côté spam.
        let l = StrikeLedger {
            user_id: "u1".into(),
            strikes: vec![
                strike_at(StrikePolicy::Spam, 1, now),
                strike_at(StrikePolicy::Unoriginal, 1, now),
                strike_at(StrikePolicy::EngagementBait, 1, now),
            ],
            ..Default::default()
        };
        assert_eq!(l.level(now), ShadowbanLevel::Monitoring);
        assert!(!l.level(now).restricts(Surface::Trending));
    }

    #[test]
    fn ghosted_ferme_jusqu_au_fil_d_abonnement() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::HatefulConduct, &[1, 2, 3, 4, 5], now);
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
        assert!(l.level(now).restricts(Surface::FollowerFeed));
        assert!(l.level(now).restricts(Surface::Trending));
    }

    /// Le cran juste en dessous, qui porte toute la différence : Suppressed
    /// ferme « Pour toi » comme Ghosted, mais laisse le fil d'abonnement.
    #[test]
    fn suppressed_laisse_le_fil_d_abonnement() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::Spam, &[1, 2], now);
        assert_eq!(l.level(now), ShadowbanLevel::Suppressed);
        assert!(l.level(now).restricts(Surface::ForYou));
        assert!(l.level(now).restricts(Surface::Trending));
        assert!(!l.level(now).restricts(Surface::FollowerFeed));
    }

    #[test]
    fn decision_manuelle_prime_dans_les_deux_sens() {
        let now = Utc::now();
        let mut l = ledger(StrikePolicy::Spam, &[1, 2, 3, 4, 5, 6, 7], now);
        assert_eq!(l.computed_level(now), ShadowbanLevel::Ghosted);

        // Blanchiment : l'heuristique s'est trompée.
        l.manual_override = Some(ShadowbanLevel::Clean);
        assert_eq!(l.level(now), ShadowbanLevel::Clean);

        // Durcissement.
        l.manual_override = Some(ShadowbanLevel::Ghosted);
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
    }

    #[test]
    fn decision_manuelle_expiree_est_ignoree() {
        let now = Utc::now();
        let mut l = StrikeLedger::new("u1");
        l.manual_override = Some(ShadowbanLevel::Ghosted);
        l.manual_expires_at = Some(now - Duration::hours(1));
        assert_eq!(l.level(now), ShadowbanLevel::Clean);
    }

    #[test]
    fn date_de_retour_est_celle_du_plus_ancien_fait_actif() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::Spam, &[10, 40, 80], now);
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
        let recovers = l.level_expires_at(now).expect("une date de retour");
        // Le plus ancien (80 j) expire dans ~10 jours.
        let days = (recovers - now).num_days();
        assert!((9..=11).contains(&days), "retour dans {days} jours");
    }

    #[test]
    fn compte_propre_n_a_pas_de_date_de_retour() {
        let now = Utc::now();
        assert!(StrikeLedger::new("u1").level_expires_at(now).is_none());
    }

    #[test]
    fn avertissement_avant_le_bannissement() {
        let now = Utc::now();
        // Harcèlement : seuil définitif à 6, alerte à partir de 75 % → 5.
        let l = ledger(StrikePolicy::Harassment, &[1, 2, 3, 4, 5], now);
        let (policy, count, limit) = l.nearing_permanent_ban(now).expect("une alerte");
        assert_eq!(policy, StrikePolicy::Harassment);
        assert_eq!((count, limit), (5, 6));

        // Sous le seuil d'alerte : rien.
        let calme = ledger(StrikePolicy::Harassment, &[1], now);
        assert!(calme.nearing_permanent_ban(now).is_none());
    }

    #[test]
    fn seuil_definitif_signale_mais_jamais_applique_seul() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::Harassment, &[1, 2, 3, 4, 5, 6], now);
        assert_eq!(l.permanent_ban_reached(now), Some(StrikePolicy::Harassment));
        // Le niveau reste Ghosted : la fonction signale, elle ne bannit pas.
        assert_eq!(l.level(now), ShadowbanLevel::Ghosted);
        // Et le compte n'est plus « proche » du seuil, il l'a franchi.
        assert!(l.nearing_permanent_ban(now).is_none());
    }

    #[test]
    fn ttl_de_cache_borne_par_la_prochaine_expiration() {
        let now = Utc::now();
        let l = ledger(StrikePolicy::Spam, &[89], now);
        let ttl = l.cache_ttl_secs(now);
        // Le fait expire dans ~1 jour : le cache ne doit pas survivre au-delà.
        assert!(ttl <= 86_400 + 60, "ttl={ttl}");
        assert!(ttl >= 60);
    }

    #[test]
    fn statut_expose_les_surfaces_fermees_et_la_date() {
        let now = Utc::now();
        // Palier Suppressed : toute recommandation se ferme, le fil d'abonnement reste.
        let l = ledger(StrikePolicy::Spam, &[1, 2], now);
        let st = l.status(now);
        assert_eq!(st.level, ShadowbanLevel::Suppressed);
        assert!(st.restricted_surfaces.contains(&"trending"));
        assert!(st.restricted_surfaces.contains(&"discover"));
        assert!(st.restricted_surfaces.contains(&"for_you"));
        assert!(!st.restricted_surfaces.contains(&"follower_feed"));
        assert!(st.recovers_at.is_some());
        assert_eq!(st.active_strikes, 2);
        assert_eq!(st.per_policy.len(), 1);
        assert_eq!(st.per_policy[0].active_strikes, 2);

        // Palier suivant : le fil d'abonnement se ferme à son tour, et le statut le dit.
        let st3 = ledger(StrikePolicy::Spam, &[1, 2, 3], now).status(now);
        assert_eq!(st3.level, ShadowbanLevel::Ghosted);
        assert!(st3.restricted_surfaces.contains(&"for_you"));
        assert!(st3.restricted_surfaces.contains(&"follower_feed"));
    }
}
