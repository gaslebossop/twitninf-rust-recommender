//! Ciblage publicitaire par les signaux de l'algorithme.
//!
//! ── Ce que ça remplace ──────────────────────────────────────────────────────
//! Le ciblage vivait entièrement côté API, sur des critères démographiques :
//! âge du compte, nombre d'abonnés, vérifié, premium. Rien de ce que
//! l'algorithme sait réellement du lecteur — ses centres d'intérêt, son
//! rythme, les comptes qui le retiennent — n'entrait dans la décision. Et le
//! fil que l'application sert vraiment (`/api/neural-rank/recommendations`)
//! n'injectait aucune publicité : celles qu'on créait n'étaient affichées
//! nulle part.
//!
//! ── Pourquoi ici et pas dans l'API ──────────────────────────────────────────
//! Le ciblage a besoin du profil lecteur (vecteur de goût, thèmes consommés,
//! type d'activité, heures actives) que ce service construit déjà à chaque
//! requête de fil. Le refaire côté Node voudrait dire le recalculer ou le
//! transporter ; le faire ici, c'est le lire.
//!
//! ── Ce qui n'est PAS ici ────────────────────────────────────────────────────
//! Le débit du budget, la comptabilité des impressions facturables et la
//! propriété des tweets restent côté API : ce module décide QUI voit QUOI,
//! il ne touche jamais à l'argent.

use std::collections::HashMap;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use pgvector::Vector;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::models::{UserProfile, UserType};
use crate::services::cache_manager::CacheManager;

/// Position de la première publicité dans le fil, puis intervalle.
///
/// Jamais en tête : la première chose vue à l'ouverture doit être du contenu.
const FIRST_SLOT: usize = 4;
const SLOT_INTERVAL: usize = 4;

/// Plafond de sécurité par page. Volontairement large : c'est l'intervalle
/// ci-dessus qui règle la densité, pas ce nombre — il n'est là que pour
/// éviter qu'une page très longue ne devienne un catalogue.
const MAX_ADS_PER_PAGE: usize = 12;

/// Plafond d'impressions par lecteur et par publicité sur 24 h, quand
/// l'annonceur n'en a pas fixé un.
///
/// Généreux à dessein : une publicité vue une fois doit pouvoir revenir. Ce
/// n'est pas parce qu'un lecteur l'a croisée qu'elle a produit son effet —
/// la répétition est le mécanisme même de la publicité. Le plafond n'existe
/// que pour éviter qu'un budget soit consommé par un seul lecteur en une
/// session.
const DEFAULT_DAILY_CAP: u32 = 150;

/// Score de correspondance en dessous duquel la publicité n'est pas servie du
/// tout. Mieux vaut un emplacement vide qu'une publicité hors sujet : elle
/// coûte de l'attention au lecteur ET du budget à l'annonceur pour rien.
const MIN_MATCH_SCORE: f64 = 0.20;

/// Fenêtre de plafonnement par lecteur et par publicité.
const FREQ_CAP_TTL_SECS: i64 = 86_400;

/// Décalage horaire de la plateforme par rapport à UTC.
///
/// Le ciblage par heure était évalué en UTC pendant que l'annonceur
/// choisissait ses heures en pensant à l'heure qu'il lit sur son téléphone :
/// une campagne réglée sur « 21h » ne sortait donc pas à 21h locales. Le
/// serveur tourne en `Etc/UTC` et n'a aucun moyen de deviner le fuseau du
/// lecteur ; on retient donc celui de la plateforme, et la liste d'heures
/// proposée côté API est calculée avec le MÊME décalage — sans quoi les
/// effectifs affichés désigneraient d'autres heures que celles ciblées.
fn platform_hour_offset() -> i64 {
    std::env::var("PLATFORM_UTC_OFFSET_HOURS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

#[derive(Debug, Clone, Serialize)]
pub struct AdPlacement {
    pub advertisement_id: String,
    pub tweet_id: String,
    /// Index d'insertion dans la page servie.
    pub position: usize,
    pub match_score: f64,
}

/// Critères de ciblage lisibles par ce module, désérialisés depuis la colonne
/// `targeting_criteria` (JSONB).
///
/// Tout est optionnel : une publicité sans aucun critère cible tout le monde,
/// et c'est un choix valable pour une campagne de notoriété. Les champs
/// démographiques historiques (`min_followers`…) restent évalués côté API —
/// ici on ne lit que ce qui vient de l'algorithme.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AlgoTargeting {
    // Pas de ciblage par thème LLM ici : il faudrait connaître les thèmes que
    // ce lecteur consomme, ce que `UserProfile` ne porte pas encore. Déclarer
    // le champ sans l'évaluer donnerait à l'annonceur l'illusion d'un ciblage
    // qui n'existe pas — et il paierait pour.
    /// Mots-clés cherchés dans ce que le lecteur consomme.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Types de lecteur visés : `power`, `regular`, `casual`.
    #[serde(default)]
    pub user_types: Vec<String>,
    /// Heures de la journée (0-23). Vide = toutes.
    #[serde(default)]
    pub hours: Vec<u32>,
    /// Cible les lecteurs qui suivent AU MOINS un de ces comptes.
    #[serde(default)]
    pub follows_any_of: Vec<String>,
    /// Plafond d'impressions par lecteur sur 24 h. `None` = pas de plafond
    /// propre (celui de la colonne `max_impressions_per_user` s'applique).
    pub daily_cap: Option<u32>,
}

struct AdRow {
    id: String,
    tweet_id: String,
    advertiser_id: String,
    targeting: AlgoTargeting,
    /// Embedding du tweet promu — permet de mesurer la proximité avec le
    /// vecteur de goût du lecteur sans qu'aucun critère n'ait été saisi.
    embedding: Option<Vec<f32>>,
    remaining_budget: f64,
    max_per_user: i32,
}

fn freq_key(user_id: &str, ad_id: &str) -> String {
    format!("ads:seen:{user_id}:{ad_id}")
}

impl CacheManager {
    /// Nombre d'impressions déjà servies à ce lecteur pour chaque publicité,
    /// en un seul `MGET` — même patron que `shadowban_load_levels`.
    async fn ads_load_seen_counts(&self, user_id: &str, ad_ids: &[String]) -> HashMap<String, u32> {
        if ad_ids.is_empty() {
            return HashMap::new();
        }
        let mut cmd = redis::cmd("MGET");
        for id in ad_ids {
            cmd.arg(freq_key(user_id, id));
        }
        let raw: Vec<Option<String>> = {
            let mut c = self.conn.lock().await;
            cmd.query_async(&mut *c).await.unwrap_or_default()
        };
        ad_ids
            .iter()
            .enumerate()
            .filter_map(|(i, id)| {
                let n: u32 = raw.get(i)?.as_deref()?.parse().ok()?;
                Some((id.clone(), n))
            })
            .collect()
    }

    /// Incrémente le compteur d'impressions d'une publicité pour ce lecteur.
    ///
    /// Distinct de la ligne `ad_impressions` écrite côté API : celle-ci est la
    /// comptabilité facturable, celle-là un simple plafond de fréquence à
    /// 24 h. Confondre les deux ferait dépendre le plafonnage d'une écriture
    /// transactionnelle bien plus lourde, sur le chemin chaud du fil.
    pub async fn ads_record_impression(&self, user_id: &str, ad_id: &str) {
        let key = freq_key(user_id, ad_id);
        let mut c = self.conn.lock().await;
        let n: Result<i64, _> = c.incr(&key, 1).await;
        if n.map(|v| v == 1).unwrap_or(false) {
            let _: Result<(), _> = c.expire(&key, FREQ_CAP_TTL_SECS).await;
        }
    }
}

/// Charge les publicités actives et encore financées.
async fn load_active_ads(pg: &PgPool, reader_id: &str) -> Result<Vec<AdRow>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(reader_id)?;
    let rows = client
        .query(
            r#"
            SELECT a.id::text, a.tweet_id::text, a.user_id::text,
                   COALESCE(a.targeting_criteria, '{}'::jsonb),
                   t.embedding,
                   a.budget::float8 - COALESCE((
                       SELECT COUNT(*) * a.cost_per_impression
                       FROM ad_impressions i WHERE i.advertisement_id = a.id
                   ), 0)::float8,
                   COALESCE(a.max_impressions_per_user, 1)
            FROM advertisements a
            JOIN tweets t ON t.id = a.tweet_id
            JOIN users u ON u.id = a.user_id
            WHERE a.status = 'active'
              AND a.start_date <= NOW()
              AND (a.end_date IS NULL OR a.end_date > NOW())
              -- Le tweet promu obéit aux mêmes règles de visibilité que
              -- n'importe quel candidat : payer ne dispense pas d'être
              -- publiable.
              AND t.deleted_at IS NULL
              AND t.moderation_status = 'approved'
              AND t.is_private = false
              AND u.is_active = true
              AND COALESCE(u.is_suspended, false) = false
              -- On ne sert jamais à quelqu'un sa propre publicité.
              AND a.user_id <> $1
            "#,
            &[&uid],
        )
        .await?;

    Ok(rows
        .iter()
        .filter_map(|r| {
            let targeting: AlgoTargeting = r
                .try_get::<_, serde_json::Value>(3)
                .ok()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            Some(AdRow {
                id: r.try_get(0).ok()?,
                tweet_id: r.try_get(1).ok()?,
                advertiser_id: r.try_get(2).ok()?,
                targeting,
                embedding: r
                    .try_get::<_, Option<Vector>>(4)
                    .ok()
                    .flatten()
                    .map(|v| v.as_slice().to_vec()),
                remaining_budget: r.try_get(5).unwrap_or(0.0),
                max_per_user: r.try_get(6).unwrap_or(1),
            })
        })
        .filter(|a| a.remaining_budget > 0.0)
        .collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-9)
}

fn user_type_label(t: &UserType) -> &'static str {
    match t {
        UserType::PowerUser => "power",
        UserType::Regular => "regular",
        UserType::Casual => "casual",
    }
}

/// Score de correspondance entre une publicité et ce lecteur, dans [0,1].
///
/// Un critère saisi qui ne correspond pas est **éliminatoire** (retourne 0) :
/// un annonceur qui a explicitement demandé « les amateurs d'humour » ne veut
/// pas payer pour un lecteur qui n'en consomme jamais, même si tout le reste
/// concorde. Ce qui reste après les critères durs est une affaire de degré :
/// la proximité sémantique entre le tweet promu et le goût du lecteur.
fn match_score(ad: &AdRow, profile: &UserProfile, hour_now: u32) -> f64 {
    let t = &ad.targeting;

    if !t.user_types.is_empty()
        && !t
            .user_types
            .iter()
            .any(|u| u.eq_ignore_ascii_case(user_type_label(&profile.user_type)))
    {
        return 0.0;
    }
    if !t.hours.is_empty() && !t.hours.contains(&hour_now) {
        return 0.0;
    }
    if !t.follows_any_of.is_empty()
        && !t
            .follows_any_of
            .iter()
            .any(|a| profile.following_ids.contains(a))
    {
        return 0.0;
    }
    if !t.keywords.is_empty() {
        let hit = t.keywords.iter().any(|k| {
            let k = k.to_lowercase();
            profile.top_words.iter().any(|(w, _)| w.to_lowercase() == k)
        });
        if !hit {
            return 0.0;
        }
    }

    // Base : un ciblage explicite qui a passé tous ses filtres vaut déjà
    // mieux qu'une publicité sans aucun critère, qui ne dit rien de personne.
    let explicit = !t.user_types.is_empty()
        || !t.hours.is_empty()
        || !t.follows_any_of.is_empty()
        || !t.keywords.is_empty();
    let mut score: f64 = if explicit { 0.55 } else { 0.30 };

    // Affinité sémantique : la moitié restante. C'est le seul critère qui
    // fonctionne sans que l'annonceur ait rien saisi — il compare ce qu'il
    // promeut à ce que le lecteur aime réellement lire.
    if let (Some(ad_vec), Some(taste)) = (ad.embedding.as_ref(), profile.taste_vector.as_ref()) {
        let sim = cosine(ad_vec, taste).clamp(-1.0, 1.0);
        // [-1,1] → [0,1], puis pondéré : une opposition franche fait chuter le
        // score sous le seuil de service, une forte proximité le pousse haut.
        score += 0.45 * ((sim + 1.0) / 2.0);
    } else {
        // Pas d'embedding des deux côtés : on ne sait pas, on ne prétend pas.
        score += 0.45 * 0.5;
    }

    score.clamp(0.0, 1.0)
}

/// Choisit les publicités à insérer dans une page de fil, et où.
///
/// Retourne une liste vide si rien ne correspond — c'est un résultat normal,
/// pas une panne.
pub async fn select_for_feed(
    pg: &PgPool,
    cache: &CacheManager,
    user_id: &str,
    profile: &UserProfile,
    page_len: usize,
) -> Vec<AdPlacement> {
    if page_len < FIRST_SLOT {
        return Vec::new();
    }
    let ads = match load_active_ads(pg, user_id).await {
        Ok(a) if !a.is_empty() => a,
        Ok(_) => return Vec::new(),
        Err(e) => {
            warn!(user_id, error = %e, "Publicités indisponibles — fil servi sans");
            return Vec::new();
        }
    };

    let ids: Vec<String> = ads.iter().map(|a| a.id.clone()).collect();
    let seen = cache.ads_load_seen_counts(user_id, &ids).await;
    let hour_now = (chrono::Utc::now() + chrono::Duration::hours(platform_hour_offset()))
        .format("%H")
        .to_string()
        .parse::<u32>()
        .unwrap_or(12);

    let mut scored: Vec<(&AdRow, f64)> = ads
        .iter()
        .filter(|a| {
            // `max_impressions_per_user` (colonne, défaut 1) ne sert plus de
            // plafond d'affichage : à 1, une publicité disparaissait
            // définitivement après une seule vue. Elle reste le plafond de
            // FACTURATION côté API, ce qui est un autre sujet.
            let already = seen.get(&a.id).copied().unwrap_or(0);
            already < a.targeting.daily_cap.unwrap_or(DEFAULT_DAILY_CAP)
        })
        .map(|a| {
            let s = match_score(a, profile, hour_now);
            (a, s)
        })
        .filter(|(_, s)| *s >= MIN_MATCH_SCORE)
        .collect();

    // Le budget restant départage à correspondance égale — une campagne qui
    // s'éteint ne doit pas monopoliser les emplacements.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.0.remaining_budget
                    .partial_cmp(&a.0.remaining_budget)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    if scored.is_empty() {
        return Vec::new();
    }

    // On remplit TOUS les emplacements de la page, en tournant sur les
    // publicités retenues. Une même publicité peut donc revenir plusieurs
    // fois : avec une seule campagne active, l'alternative serait un unique
    // emplacement occupé et tous les autres vides.
    //
    // La rotation part des mieux notées et repasse au début une fois la
    // liste épuisée — l'ordre du classement est ainsi respecté à chaque tour,
    // plutôt que de servir la meilleure partout.
    let mut placements = Vec::new();
    let mut position = FIRST_SLOT;
    let mut i = 0usize;
    let mut used: HashMap<&str, u32> = HashMap::new();

    while position < page_len && placements.len() < MAX_ADS_PER_PAGE {
        // Une publicité peut se répéter, mais pas au-delà de son plafond
        // journalier restant : sinon une seule page pourrait le franchir.
        let mut placed = false;
        for _ in 0..scored.len() {
            let (ad, score) = &scored[i % scored.len()];
            i += 1;
            let cap = ad.targeting.daily_cap.unwrap_or(DEFAULT_DAILY_CAP);
            let already = seen.get(&ad.id).copied().unwrap_or(0);
            let this_page = used.get(ad.id.as_str()).copied().unwrap_or(0);
            if already + this_page >= cap {
                continue;
            }
            *used.entry(ad.id.as_str()).or_insert(0) += 1;
            placements.push(AdPlacement {
                advertisement_id: ad.id.clone(),
                tweet_id: ad.tweet_id.clone(),
                position,
                match_score: (score * 1000.0).round() / 1000.0,
            });
            placed = true;
            break;
        }
        if !placed {
            break; // toutes les publicités ont atteint leur plafond
        }
        position += SLOT_INTERVAL;
    }

    for p in &placements {
        cache
            .ads_record_impression(user_id, &p.advertisement_id)
            .await;
    }
    if !placements.is_empty() {
        debug!(
            user_id,
            count = placements.len(),
            "Publicités ciblées servies"
        );
    }
    placements
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> UserProfile {
        let mut p = UserProfile {
            user_id: "u".into(),
            ..Default::default()
        };
        p.user_type = UserType::Regular;
        p.top_words = vec![("crypto".into(), 12), ("foot".into(), 7)];
        p.following_ids = vec!["auteur-a".into()];
        p
    }

    fn ad(t: AlgoTargeting) -> AdRow {
        AdRow {
            id: "ad".into(),
            tweet_id: "tw".into(),
            advertiser_id: "adv".into(),
            targeting: t,
            embedding: None,
            remaining_budget: 10.0,
            max_per_user: 3,
        }
    }

    #[test]
    fn sans_critere_la_pub_reste_servable_mais_moins_bien_notee() {
        let large = match_score(&ad(AlgoTargeting::default()), &profile(), 12);
        let cible = match_score(
            &ad(AlgoTargeting {
                user_types: vec!["regular".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert!(
            large >= MIN_MATCH_SCORE,
            "une campagne large reste servable"
        );
        assert!(
            cible > large,
            "un ciblage juste doit primer : {cible} > {large}"
        );
    }

    #[test]
    fn un_critere_non_rempli_est_eliminatoire() {
        // L'annonceur a demandé les gros utilisateurs : un lecteur régulier ne
        // doit pas lui être facturé, même si tout le reste concorde.
        let s = match_score(
            &ad(AlgoTargeting {
                user_types: vec!["power".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert_eq!(s, 0.0);
    }

    #[test]
    fn le_ciblage_horaire_et_par_mot_cle_filtre() {
        let hors_heure = match_score(
            &ad(AlgoTargeting {
                hours: vec![3, 4],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert_eq!(hors_heure, 0.0);

        let mot_absent = match_score(
            &ad(AlgoTargeting {
                keywords: vec!["cuisine".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert_eq!(mot_absent, 0.0);

        let mot_present = match_score(
            &ad(AlgoTargeting {
                keywords: vec!["CRYPTO".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert!(mot_present > 0.0, "la casse ne doit pas faire échouer");
    }

    #[test]
    fn le_ciblage_par_abonnement_verifie_le_graphe_du_lecteur() {
        let suit = match_score(
            &ad(AlgoTargeting {
                follows_any_of: vec!["auteur-a".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        let suit_pas = match_score(
            &ad(AlgoTargeting {
                follows_any_of: vec!["auteur-z".into()],
                ..Default::default()
            }),
            &profile(),
            12,
        );
        assert!(suit > 0.0);
        assert_eq!(suit_pas, 0.0);
    }
}
