//! Recalibration de l'algorithme — page dédiée, accessible uniquement depuis
//! les Paramètres, jamais proposée automatiquement (demande explicite).
//!
//! 3 tours de 6 cartes, sélectionnées dans l'espace des embeddings — pas par
//! auteur ni par thème, qui ne sont que des étiquettes posées à côté du
//! contenu.
//!
//! **Tour 1 — couverture.** On ne sait rien : les 6 cartes sont choisies pour
//! être le plus éloignées possible les unes des autres, de façon que
//! n'importe quel goût trouve au moins une prise. C'est un échantillonnage du
//! point le plus éloigné, obtenu par la pénalité de redondance de
//! `greedy_pick`.
//!
//! **Tours 2-3 — la frontière, pas le déjà-acquis.** La première version
//! montrait ici les plus proches voisins de ce qui venait d'être aimé : une
//! question dont on connaît déjà la réponse n'apprend rien, et sur un corpus
//! dominé par quelques comptes elle ramenait toujours les mêmes. On cherche
//! désormais les contenus À ÉGALE DISTANCE de ce qui a été accepté et de ce
//! qui a été refusé — là où le modèle ne sait pas trancher, donc là où une
//! réponse vaut le plus. C'est la transposition de la sélection par
//! entropie/variance en apprentissage actif.
//!
//! Dans les deux cas la sélection est GLOUTONNE AVEC PÉNALITÉ DE REDONDANCE :
//! chaque carte est choisie en tenant compte de celles déjà retenues pour ce
//! tour. Noter les items isolément produit des graines redondantes — c'est le
//! résultat central de « Deep Rating Elicitation » (arXiv 2402.16327).
//!
//! Signal privé : contrairement à un like normal, un choix de recalibration
//! n'écrit jamais dans `tweet_likes` (pas de notification à l'auteur, pas de
//! compteur public qui bouge) — seul l'algorithme en tient compte.
//! `finish()` déclenche ce que des likes ordinaires auraient déclenché côté
//! algo (boost temps réel par auteur, cooccurrence globale — mêmes fonctions
//! qu'un like normal, voir `handlers::tracking::track_handler`), plus un
//! effet qu'un like ordinaire, dilué dans l'activité normale, n'a pas : un
//! vecteur de goût dédié et concentré, mélangé au vecteur naturel (90 jours
//! de likes) au prochain rechargement de profil — voir `blend_taste`.

use std::collections::HashSet;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use pgvector::Vector;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::embeddings::average_vectors;
use crate::services::cache_manager::CacheManager;

pub const ROUNDS: u8 = 3;
pub const TWEETS_PER_ROUND: i64 = 6;

/// Taille du vivier chargé à chaque tour. Assez large pour que la dispersion
/// et l'incertitude aient de quoi choisir, assez borné pour que le calcul de
/// distances reste instantané (O(vivier × cartes), quelques milliers
/// d'opérations).
const POOL_LIMIT: i64 = 240;

/// Tweets max par auteur DANS LE VIVIER (avant même la sélection).
const POOL_PER_AUTHOR: i64 = 8;

/// Poids de la pénalité de redondance dans la sélection gloutonne. À 1.0, un
/// candidat identique à une carte déjà retenue est écarté aussi sûrement
/// qu'un candidat sans aucun intérêt propre.
const REDUNDANCY_WEIGHT: f32 = 1.0;

/// Tours purement diversifiés, sans aucun signal de similarité — la
/// cartographie initiale. Au-delà, chaque tour reste un MÉLANGE (voir
/// `round_candidates`), jamais une bascule totale vers la seule similarité :
/// sur un corpus aussi mince que celui de la plateforme aujourd'hui (une
/// poignée de tweets embeddés, deux ou trois auteurs qui dominent), une
/// similarité pure ne renvoie plus que ces mêmes comptes — observé en test
/// réel : 2 auteurs sur les 5 tours.
const DIVERSITY_ENFORCED_THROUGH_ROUND: u8 = 1;

/// Nombre max de tweets d'un même auteur DANS UN TOUR, quelle que soit la
/// source (diversifiée ou par similarité). Sans lui, `similarity_candidates`
/// n'avait aucune limite : les plus proches voisins d'un vecteur de goût
/// peuvent très bien être 6 tweets du même compte si c'est lui qui domine le
/// corpus embeddé.
const MAX_PER_AUTHOR_PER_ROUND: usize = 2;

/// TTL du vecteur de goût de calibration. Rien ne le fait naturellement
/// expirer comme un avertissement (voir `shadowban::strikes`) : un compte qui
/// ne recalibre plus jamais garde son dernier réglage indéfiniment tant qu'il
/// est actif. Le TTL n'est qu'un filet — purger un compte abandonné plutôt
/// que de faire grossir Redis sans fin.
const CALIBRATION_TASTE_TTL_SECS: i64 = 180 * 24 * 3600;

/// Poids du vecteur de calibration face au vecteur naturel (90 jours de
/// likes) quand les deux existent — voir `blend_taste`. Un choix explicite et
/// concentré sur 5 tours pèse plus qu'un like ordinaire noyé dans trois mois
/// d'activité, mais ne l'efface pas entièrement : le compte a aussi un passé.
const CALIBRATION_TASTE_WEIGHT: f32 = 0.65;

#[derive(Debug, Deserialize)]
pub struct CalibrationRoundRequest {
    pub user_id: String,
    pub round: u8,
    /// Cumulés depuis le premier tour de CETTE session, pas seulement le tour
    /// précédent — round_candidates doit connaître tout ce qui a déjà été
    /// montré pour ne jamais répéter un tweet.
    #[serde(default)]
    pub liked_tweet_ids: Vec<String>,
    #[serde(default)]
    pub skipped_tweet_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CalibrationRoundResult {
    pub round: u8,
    pub tweet_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CalibrationFinishRequest {
    pub user_id: String,
    pub liked_tweet_ids: Vec<String>,
}

fn calibration_taste_key(user_id: &str) -> String {
    format!("calibration:taste:{user_id}")
}

impl CacheManager {
    async fn calibration_save_taste(&self, user_id: &str, taste: &[f32]) {
        let Ok(json) = serde_json::to_string(taste) else {
            return;
        };
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c
            .set_ex(
                calibration_taste_key(user_id),
                json,
                CALIBRATION_TASTE_TTL_SECS as u64,
            )
            .await;
    }

    /// Vecteur de goût explicite du compte, s'il a déjà recalibré — voir
    /// `blend_taste` pour comment il se combine au vecteur naturel.
    pub async fn calibration_load_taste(&self, user_id: &str) -> Option<Vec<f32>> {
        let mut c = self.conn.lock().await;
        let raw: Option<String> = c.get(calibration_taste_key(user_id)).await.ok().flatten();
        drop(c);
        raw.and_then(|s| serde_json::from_str(&s).ok())
    }
}

/// Combine le vecteur de calibration explicite et le vecteur naturel (90
/// jours de likes) — voir `CALIBRATION_TASTE_WEIGHT`. Reste correct côté
/// `pgvector` même sans renormaliser : la distance cosinus (`<=>`, utilisée
/// partout où ce vecteur sert) est déjà invariante à l'échelle.
pub fn blend_taste(calibration: &[f32], natural: &[f32]) -> Vec<f32> {
    calibration
        .iter()
        .zip(natural.iter())
        .map(|(c, n)| c * CALIBRATION_TASTE_WEIGHT + n * (1.0 - CALIBRATION_TASTE_WEIGHT))
        .collect()
}

/// Sélectionne les candidats d'UN tour.
///
/// Au-delà de `DIVERSITY_ENFORCED_THROUGH_ROUND`, le tour reste un MÉLANGE —
/// la moitié au plus vient de la similarité sémantique, le reste du vivier
/// diversifié. Une bascule totale vers la similarité donnait, sur le corpus
/// actuel, le même auteur tour après tour : resserrer ne doit pas dégénérer
/// en boucle sur deux ou trois comptes.
/// Un candidat du vivier, avec son vecteur — tout le raisonnement de
/// sélection se fait en mémoire sur ce vivier, pas en SQL : les critères
/// (dispersion, incertitude) portent sur les distances ENTRE candidats, ce
/// qu'une requête `ORDER BY` ne sait pas exprimer.
struct Candidate {
    id: String,
    author: String,
    vec: Vec<f32>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    dot / denom
}

/// Charge le vivier une fois par tour : tweets visibles, embeddés, plafonnés
/// par auteur DÈS LA REQUÊTE.
///
/// Le plafond SQL est ce qui rend la suite honnête : sans lui, un vivier
/// trié globalement est composé aux trois quarts du compte le plus
/// prolifique (169 tweets éligibles contre 78 au suivant, relevé en prod), et
/// aucune sélection en aval ne peut réparer un vivier déjà biaisé.
async fn load_pool(
    pg: &PgPool,
    user_id: &str,
    excluded: &HashSet<String>,
) -> Result<Vec<Candidate>> {
    let client = pg.get().await?;
    let uid = uuid::Uuid::parse_str(user_id)?;
    let rows = client
        .query(
            r#"
            SELECT id::text, author::text, embedding FROM (
                SELECT t.id, t.user_id AS author, t.embedding,
                       ROW_NUMBER() OVER (
                           PARTITION BY t.user_id
                           ORDER BY COALESCE(ll.quality_score, 0.5) DESC, t.created_at DESC
                       ) AS rn
                FROM tweets t
                JOIN users u ON u.id = t.user_id
                LEFT JOIN tweet_llm_labels ll ON ll.tweet_id = t.id
                WHERE t.embedding IS NOT NULL
                  AND t.deleted_at IS NULL
                  AND t.moderation_status = 'approved'
                  AND t.is_private = false
                  AND COALESCE(t.is_data_test, false) = false
                  AND u.is_active = true
                  AND COALESCE(u.is_suspended, false) = false
                  AND t.user_id != $1
                  AND t.parent_tweet_id IS NULL
                  AND COALESCE(t.is_retweet, false) = false
                  AND COALESCE(t.content, '') != ''
                  AND LENGTH(COALESCE(t.content, '')) >= 40
                  AND COALESCE(ll.theme, 'autre') != 'spam_vide'
            ) ranked
            WHERE rn <= $2
            LIMIT $3
            "#,
            &[&uid, &POOL_PER_AUTHOR, &POOL_LIMIT],
        )
        .await?;

    Ok(rows
        .iter()
        .filter_map(|r| {
            let id: String = r.try_get(0).ok()?;
            if excluded.contains(&id) {
                return None;
            }
            let author: String = r.try_get(1).ok()?;
            let v: Vector = r.try_get(2).ok()?;
            Some(Candidate {
                id,
                author,
                vec: v.as_slice().to_vec(),
            })
        })
        .collect())
}

/// Sélection gloutonne commune aux deux stratégies.
///
/// `base_score` dit ce qu'on cherche (dispersion, ou incertitude) ; la
/// pénalité de redondance est ajoutée ici, une fois, parce qu'elle vaut dans
/// les deux cas : deux cartes quasi identiques dans le même tour, c'est une
/// question posée deux fois. C'est le point central de la littérature sur le
/// sujet — les méthodes qui notent chaque item isolément sélectionnent des
/// graines redondantes, et il faut tenir compte des interactions entre elles
/// (voir « Deep Rating Elicitation », arXiv 2402.16327).
fn greedy_pick<F>(pool: &[Candidate], k: usize, base_score: F) -> Vec<String>
where
    F: Fn(&Candidate) -> f32,
{
    let mut picked: Vec<usize> = Vec::new();
    let mut per_author: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

    while picked.len() < k {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in pool.iter().enumerate() {
            if picked.contains(&i) {
                continue;
            }
            if per_author.get(c.author.as_str()).copied().unwrap_or(0) >= MAX_PER_AUTHOR_PER_ROUND {
                continue;
            }
            // Redondance : proximité maximale à ce qui est DÉJÀ retenu pour
            // ce tour.
            let redundancy = picked
                .iter()
                .map(|&j| cosine(&c.vec, &pool[j].vec))
                .fold(f32::NEG_INFINITY, f32::max);
            let redundancy = if redundancy == f32::NEG_INFINITY {
                0.0
            } else {
                redundancy
            };
            let score = base_score(c) - REDUNDANCY_WEIGHT * redundancy;
            if best.map_or(true, |(_, b)| score > b) {
                best = Some((i, score));
            }
        }
        match best {
            Some((i, _)) => {
                *per_author.entry(pool[i].author.as_str()).or_insert(0) += 1;
                picked.push(i);
            }
            None => break,
        }
    }
    picked.into_iter().map(|i| pool[i].id.clone()).collect()
}

/// Premier tour — couverture maximale de l'espace des goûts.
///
/// On ne sait rien du lecteur : le meilleur usage de 6 cartes est de couvrir
/// le plus largement possible ce que la plateforme contient, pour que
/// n'importe quel goût trouve au moins une prise. La dispersion est obtenue
/// par la seule pénalité de redondance de `greedy_pick` (score de base
/// constant) — c'est exactement un échantillonnage du point le plus éloigné.
fn spread_pick(pool: &[Candidate], k: usize) -> Vec<String> {
    greedy_pick(pool, k, |_| 0.0)
}

/// Tours suivants — les cartes les plus INFORMATIVES, pas les plus proches.
///
/// Montrer les plus proches voisins de ce qui vient d'être aimé était l'erreur
/// de la première version : la réponse est connue d'avance, donc la carte
/// n'apprend rien. Ce qui apprend, c'est la frontière — les contenus dont on
/// ne sait pas trancher s'ils plaisent ou non, ceux qui sont à égale distance
/// de ce qui a été accepté et de ce qui a été refusé. C'est la transposition
/// directe de la sélection par entropie/variance en apprentissage actif :
/// interroger là où le désaccord est maximal, pas là où le résultat est acquis.
fn informative_pick(
    pool: &[Candidate],
    liked: Option<&Vec<f32>>,
    disliked: Option<&Vec<f32>>,
    k: usize,
) -> Vec<String> {
    greedy_pick(pool, k, |c| {
        let s_like = liked.map(|v| cosine(&c.vec, v)).unwrap_or(0.0);
        let s_dislike = disliked.map(|v| cosine(&c.vec, v)).unwrap_or(0.0);
        // Marge proche de zéro = frontière = incertitude maximale.
        -(s_like - s_dislike).abs()
    })
}

/// Sélectionne les candidats d'UN tour.
pub async fn round_candidates(
    pg: &PgPool,
    user_id: &str,
    round: u8,
    liked_so_far: &[String],
    skipped_so_far: &[String],
) -> Result<Vec<String>> {
    let mut excluded: HashSet<String> = liked_so_far.iter().cloned().collect();
    excluded.extend(skipped_so_far.iter().cloned());

    let pool = load_pool(pg, user_id, &excluded).await?;
    if pool.is_empty() {
        return Ok(Vec::new());
    }
    let k = TWEETS_PER_ROUND as usize;

    if round <= DIVERSITY_ENFORCED_THROUGH_ROUND {
        return Ok(spread_pick(&pool, k));
    }

    let liked_vec = mean_embedding(pg, liked_so_far).await?;
    let disliked_vec = mean_embedding(pg, skipped_so_far).await?;
    if liked_vec.is_none() && disliked_vec.is_none() {
        // Aucune réponse exploitable (tout ignoré, ou embeddings absents) :
        // continuer à couvrir large vaut mieux que de deviner.
        return Ok(spread_pick(&pool, k));
    }
    Ok(informative_pick(
        &pool,
        liked_vec.as_ref(),
        disliked_vec.as_ref(),
        k,
    ))
}

async fn mean_embedding(pg: &PgPool, tweet_ids: &[String]) -> Result<Option<Vec<f32>>> {
    if tweet_ids.is_empty() {
        return Ok(None);
    }
    let uuids: Vec<uuid::Uuid> = tweet_ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .collect();
    if uuids.is_empty() {
        return Ok(None);
    }
    let client = pg.get().await?;
    let rows = client
        .query(
            "SELECT embedding FROM tweets WHERE id = ANY($1) AND embedding IS NOT NULL",
            &[&uuids],
        )
        .await?;
    let vectors: Vec<Vec<f32>> = rows
        .iter()
        .filter_map(|row| row.try_get::<_, Vector>(0).ok())
        .map(|v| v.as_slice().to_vec())
        .collect();
    Ok(average_vectors(&vectors))
}

/// Traite les résultats d'une session complète — voir la doc de module pour
/// ce que ça déclenche et pourquoi ce n'est PAS un like public.
pub async fn finish(
    pg: &PgPool,
    cache: &CacheManager,
    user_id: &str,
    liked_tweet_ids: &[String],
) -> Result<usize> {
    if liked_tweet_ids.is_empty() {
        return Ok(0);
    }

    if let Some(taste) = mean_embedding(pg, liked_tweet_ids).await? {
        cache.calibration_save_taste(user_id, &taste).await;
    }

    // Sans ces deux invalidations, une recalibration ne change RIEN au
    // fil pendant jusqu'à 5 minutes : le profil (`twitninf:profile:*`,
    // contient le vecteur de goût) et la liste déjà classée
    // (`twitninf:reco:*`) restent tous deux en cache indépendamment de ce
    // qui vient d'être écrit ci-dessus. C'est le point même de la
    // fonctionnalité — la sanctionner par un délai silencieux la rend
    // indiscernable d'un bouton qui ne fait rien.
    cache.invalidate_profile(user_id).await;
    cache.invalidate_recommendations(user_id).await;

    let uuids: Vec<uuid::Uuid> = liked_tweet_ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .collect();
    if uuids.is_empty() {
        return Ok(0);
    }
    let client = pg.get().await?;
    let rows = client
        .query(
            "SELECT user_id::text FROM tweets WHERE id = ANY($1)",
            &[&uuids],
        )
        .await?;
    drop(client);

    for row in &rows {
        let author_id: String = row.get(0);
        cache
            .record_author_feedback(user_id, &author_id, true)
            .await;
        cache.record_like_cooccurrence(user_id, &author_id).await;
    }
    debug!(user_id, picks = rows.len(), "Recalibration terminée");
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EMBEDDING_DIM;

    #[test]
    fn blend_pondere_vers_la_calibration() {
        let calib = vec![1.0f32; EMBEDDING_DIM];
        let natural = vec![0.0f32; EMBEDDING_DIM];
        let blended = blend_taste(&calib, &natural);
        assert!((blended[0] - CALIBRATION_TASTE_WEIGHT).abs() < 1e-6);
    }

    fn cand(id: &str, author: &str, v: &[f32]) -> Candidate {
        let mut vec = vec![0.0f32; EMBEDDING_DIM];
        vec[..v.len()].copy_from_slice(v);
        Candidate {
            id: id.into(),
            author: author.into(),
            vec,
        }
    }

    #[test]
    fn le_premier_tour_disperse_au_lieu_de_regrouper() {
        // Trois quasi-jumeaux et deux contenus distincts : une sélection de 3
        // doit prendre UN des jumeaux, pas les trois. C'est le défaut que la
        // pénalité de redondance existe pour corriger.
        let pool = vec![
            cand("a1", "u1", &[1.0, 0.0, 0.0]),
            cand("a2", "u2", &[0.99, 0.01, 0.0]),
            cand("a3", "u3", &[0.98, 0.02, 0.0]),
            cand("b", "u4", &[0.0, 1.0, 0.0]),
            cand("c", "u5", &[0.0, 0.0, 1.0]),
        ];
        let picked = spread_pick(&pool, 3);
        assert_eq!(picked.len(), 3);
        assert!(picked.contains(&"b".to_string()), "picked={picked:?}");
        assert!(picked.contains(&"c".to_string()), "picked={picked:?}");
        let jumeaux = picked.iter().filter(|id| id.starts_with('a')).count();
        assert_eq!(jumeaux, 1, "un seul des trois jumeaux : {picked:?}");
    }

    #[test]
    fn les_tours_suivants_visent_la_frontiere_pas_le_deja_acquis() {
        // Aimé = axe X, rejeté = axe Y. Le contenu le plus informatif est
        // celui qui tient des deux (la diagonale), pas celui qui ressemble
        // exactement à ce qui a déjà été aimé : sa réponse est déjà connue.
        let pool = vec![
            cand("deja_acquis", "u1", &[1.0, 0.0, 0.0]),
            cand("deja_rejete", "u2", &[0.0, 1.0, 0.0]),
            cand("frontiere", "u3", &[0.7, 0.7, 0.0]),
        ];
        let liked = {
            let mut v = vec![0.0f32; EMBEDDING_DIM];
            v[0] = 1.0;
            v
        };
        let disliked = {
            let mut v = vec![0.0f32; EMBEDDING_DIM];
            v[1] = 1.0;
            v
        };
        let picked = informative_pick(&pool, Some(&liked), Some(&disliked), 1);
        assert_eq!(picked, vec!["frontiere".to_string()]);
    }

    #[test]
    fn le_plafond_par_auteur_tient_meme_si_un_compte_domine_le_vivier() {
        let pool = vec![
            cand("x1", "gros", &[1.0, 0.0, 0.0]),
            cand("x2", "gros", &[0.0, 1.0, 0.0]),
            cand("x3", "gros", &[0.0, 0.0, 1.0]),
            cand("y1", "petit", &[0.5, 0.5, 0.0]),
        ];
        let picked = spread_pick(&pool, 4);
        assert!(
            picked.iter().filter(|id| id.starts_with('x')).count() <= MAX_PER_AUTHOR_PER_ROUND,
            "picked={picked:?}"
        );
    }

    #[test]
    fn blend_egal_aux_deux_bouts_quand_un_seul_existe() {
        // Documente juste la formule : à poids 0.65/0.35, un vecteur nul d'un
        // côté ne redonne PAS l'autre vecteur tel quel — c'est round_candidates
        // /le point d'appel dans `recommender.rs` qui gère l'absence de l'un
        // des deux en amont, pas `blend_taste`.
        let calib = vec![2.0f32; EMBEDDING_DIM];
        let natural = vec![2.0f32; EMBEDDING_DIM];
        let blended = blend_taste(&calib, &natural);
        assert!((blended[0] - 2.0).abs() < 1e-6);
    }
}
