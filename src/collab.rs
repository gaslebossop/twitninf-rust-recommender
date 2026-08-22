//! Plongements collaboratifs — placer lecteurs et auteurs dans le MÊME espace.
//!
//! ── L'écart que ce module comble ────────────────────────────────────────────
//! Le moteur avait deux notions de ressemblance, et il manquait la troisième :
//!
//!   * `crate::embeddings` (MiniLM) rapproche deux tweets qui parlent de la
//!     même chose. C'est de la ressemblance de CONTENU.
//!   * `crate::cooccurrence` sait dire « les gens qui aiment A aiment souvent
//!     B ». C'est un voisinage, une liste — pas une position.
//!   * Il n'existait aucune représentation dans laquelle un LECTEUR et un
//!     AUTEUR puissent être comparés directement.
//!
//! C'est exactement ce que fait `SimClusters` chez X : une factorisation
//! matricielle du graphe produit des vecteurs de communauté, et la
//! recommandation devient un produit scalaire entre le vecteur du lecteur
//! (« interested in ») et celui du producteur (« known for »). C'est aussi ce
//! que les tables de plongement de `Monolith` (TikTok) apprennent, mais par
//! descente de gradient sur des milliards d'interactions — un chemin qui n'a
//! rien sur quoi mordre à notre volume.
//!
//! ── Pourquoi une factorisation et pas un réseau ─────────────────────────────
//! Une table de plongement apprise par SGD a besoin de beaucoup d'observations
//! PAR ENTITÉ pour que chaque vecteur veuille dire quelque chose. Une
//! factorisation, elle, extrait la structure d'un coup de la matrice entière :
//! elle donne un résultat utile dès que la matrice a une structure, même avec
//! peu de lignes. C'est le bon outil à cette échelle, et c'est le même que
//! celui de SimClusters.
//!
//! ── Comment, sans dépendance ────────────────────────────────────────────────
//! Itération orthogonale (« subspace iteration ») sur la matrice de
//! co-occurrence symétrique et normalisée. On part de `k` vecteurs aléatoires,
//! on les multiplie par la matrice, on les réorthonormalise, on recommence. Ils
//! convergent vers les `k` premiers vecteurs propres — c'est-à-dire vers les
//! `k` axes qui expliquent le plus de la structure du graphe. Une cinquantaine
//! de lignes, aucune bibliothèque d'algèbre linéaire.
//!
//! ── Honnêteté sur l'échelle ─────────────────────────────────────────────────
//! À dix auteurs éligibles, ce module ne dira presque rien : dix points dans un
//! espace de seize dimensions n'ont pas de structure à extraire. Le code est
//! écrit pour être insensible à l'échelle et se mettra à mordre quand le corpus
//! grossira. `is_usable()` dit franchement si la sortie vaut quelque chose,
//! plutôt que de rendre un nombre qui aurait l'air d'un signal.

use std::collections::HashMap;

use redis::AsyncCommands;
use tracing::{debug, info, warn};

use crate::services::cache_manager::CacheManager;

/// Dimensions du plongement.
///
/// Seize, et pas les centaines qu'utilisent X ou TikTok : le nombre d'axes
/// qu'on peut extraire est borné par le nombre d'entités. Demander plus d'axes
/// que la matrice n'en porte ne produit pas plus d'information, seulement du
/// bruit orthonormé.
pub const DIM: usize = 16;

/// En dessous, la factorisation n'a pas de sens — on préfère ne rien rendre.
///
/// Trente auteurs pour seize axes est déjà maigre ; c'est un plancher, pas une
/// cible.
const MIN_AUTHORS: usize = 30;

/// Plafond de la matrice. Au-delà, on garde les auteurs les mieux connectés :
/// le coût de l'itération est linéaire en nombre d'arêtes, et les auteurs à une
/// seule co-occurrence n'apportent aucun axe.
const MAX_AUTHORS: usize = 4000;

/// Passes d'itération orthogonale. La convergence est géométrique ; au-delà
/// d'une vingtaine de passes les axes ne bougent plus visiblement.
const ITERATIONS: usize = 24;

// ═══════════════════════════════════════════════════════════════════════════
// L'espace
// ═══════════════════════════════════════════════════════════════════════════

/// Les auteurs, placés. Reconstruit périodiquement, jamais modifié sur place.
#[derive(Debug, Clone, Default)]
pub struct CollabSpace {
    /// Vecteur unitaire par auteur.
    vectors: HashMap<String, [f32; DIM]>,
}

impl CollabSpace {
    pub fn is_usable(&self) -> bool {
        self.vectors.len() >= MIN_AUTHORS
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn author(&self, author_id: &str) -> Option<&[f32; DIM]> {
        self.vectors.get(author_id)
    }

    /// Vecteur du LECTEUR : la moyenne pondérée des auteurs avec lesquels il a
    /// interagi, ramenée à la norme 1.
    ///
    /// C'est la construction « interested in » de SimClusters : un lecteur
    /// n'a pas de position propre, il occupe le barycentre de ce qu'il
    /// consomme. Conséquence directe et voulue : un lecteur sans historique
    /// n'a pas de vecteur du tout, et ce module se tait pour lui — c'est plus
    /// honnête que de le placer au centre, où il serait à égale distance de
    /// tout et produirait un signal constant.
    pub fn reader(&self, affinities: &[(String, f64)]) -> Option<[f32; DIM]> {
        let mut acc = [0.0f32; DIM];
        let mut seen = 0usize;
        for (author_id, weight) in affinities {
            let Some(v) = self.vectors.get(author_id) else {
                continue;
            };
            let w = *weight as f32;
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += w * x;
            }
            seen += 1;
        }
        if seen == 0 {
            return None;
        }
        normalize(&mut acc).then_some(acc)
    }

    /// Affinité collaborative entre un lecteur placé et un auteur, ramenée sur
    /// `[0, 1]`.
    ///
    /// Le cosinus vit sur `[-1, 1]` ; un trait de modèle doit vivre sur la même
    /// échelle que ses voisins, sinon son poids appris n'est pas comparable aux
    /// autres. `0,5` signifie donc « orthogonal », c'est-à-dire aucun rapport —
    /// ce qui est bien la valeur neutre attendue.
    pub fn affinity(&self, reader: &[f32; DIM], author_id: &str) -> Option<f64> {
        let a = self.vectors.get(author_id)?;
        let dot: f32 = reader.iter().zip(a.iter()).map(|(x, y)| x * y).sum();
        Some(((dot as f64) + 1.0) / 2.0)
    }

    // ─── Construction ──────────────────────────────────────────────────────

    /// Lit la matrice de co-occurrence dans Redis et la factorise.
    pub async fn build(cache: &CacheManager) -> Self {
        let edges = match read_cooccurrence(cache).await {
            Ok(e) => e,
            Err(e) => {
                warn!("Plongements collaboratifs : lecture de la co-occurrence impossible ({e})");
                return Self::default();
            }
        };
        Self::from_edges(edges)
    }

    /// Partie PURE, testable sans Redis : des arêtes pondérées vers un espace.
    pub fn from_edges(edges: Vec<(String, String, f64)>) -> Self {
        if edges.is_empty() {
            return Self::default();
        }

        // ── Index des auteurs, plafonné aux mieux connectés ────────────────
        let mut degree: HashMap<&str, f64> = HashMap::new();
        for (a, b, w) in &edges {
            *degree.entry(a.as_str()).or_insert(0.0) += *w;
            *degree.entry(b.as_str()).or_insert(0.0) += *w;
        }
        let mut ranked: Vec<(&str, f64)> = degree.into_iter().collect();
        // Tri par degre DECROISSANT, puis par identifiant croissant.
        //
        // Le second critere n'est pas cosmetique : sans lui, deux auteurs de
        // meme degre sont departages par l'ordre d'iteration de la `HashMap`,
        // qui est aleatoire a chaque execution en Rust. Deux reconstructions
        // successives placeraient alors les memes auteurs a des indices
        // differents, donc dans une base tournee differemment — et le poids
        // appris sur le trait d'affinite collaborative poursuivrait une cible
        // mouvante. Le repere doit etre le meme d'un rafraichissement au
        // suivant, sinon le trait ne veut rien dire.
        ranked.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.0.cmp(y.0))
        });
        ranked.truncate(MAX_AUTHORS);

        if ranked.len() < MIN_AUTHORS {
            debug!(
                authors = ranked.len(),
                minimum = MIN_AUTHORS,
                "Plongements collaboratifs : pas assez d'auteurs, espace vide"
            );
            return Self::default();
        }

        let index: HashMap<&str, usize> = ranked
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();
        let strength: Vec<f64> = ranked.iter().map(|(_, d)| *d).collect();
        let n = ranked.len();

        // ── Matrice creuse, symétrique, normalisée ─────────────────────────
        //
        // `c_ij / sqrt(d_i · d_j)` — la normalisation cosinus habituelle des
        // matrices d'affinité. Sans elle, les axes extraits ne décriraient que
        // « qui est populaire », qui est déjà ce que D1 mesure : on veut la
        // structure du graphe, pas son volume.
        let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (a, b, w) in &edges {
            let (Some(&i), Some(&j)) = (index.get(a.as_str()), index.get(b.as_str())) else {
                continue;
            };
            if i == j {
                continue;
            }
            let norm = (strength[i] * strength[j]).sqrt();
            if norm <= 0.0 {
                continue;
            }
            rows[i].push((j, (*w / norm) as f32));
        }

        let basis = subspace_iteration(&rows, n, DIM, ITERATIONS);

        let mut vectors = HashMap::with_capacity(n);
        for (id, i) in index {
            let mut v = [0.0f32; DIM];
            for (d, item) in v.iter_mut().enumerate() {
                *item = basis[d][i];
            }
            if normalize(&mut v) {
                vectors.insert(id.to_string(), v);
            }
        }

        info!(
            authors = vectors.len(),
            dims = DIM,
            "Plongements collaboratifs reconstruits"
        );
        Self { vectors }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Algèbre
// ═══════════════════════════════════════════════════════════════════════════

/// Les `k` premiers vecteurs propres d'une matrice creuse symétrique, par
/// itération orthogonale.
///
/// Le procédé tient en trois lignes : multiplier la base par la matrice,
/// réorthonormaliser, recommencer. Chaque produit amplifie les directions de
/// plus grande valeur propre, la réorthonormalisation empêche les `k` vecteurs
/// de s'effondrer tous sur la première.
fn subspace_iteration(
    rows: &[Vec<(usize, f32)>],
    n: usize,
    k: usize,
    iterations: usize,
) -> Vec<Vec<f32>> {
    // Départ déterministe : un générateur congruentiel à graine fixe. Une base
    // aléatoire non reproductible rendrait deux reconstructions successives
    // incomparables, et surtout impossibles à tester.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };

    let mut basis: Vec<Vec<f32>> = (0..k).map(|_| (0..n).map(|_| next()).collect()).collect();
    orthonormalize(&mut basis);

    let mut scratch = vec![0.0f32; n];
    for _ in 0..iterations {
        for vector in basis.iter_mut() {
            scratch.iter_mut().for_each(|x| *x = 0.0);
            for (i, row) in rows.iter().enumerate() {
                let mut acc = 0.0f32;
                for (j, w) in row {
                    acc += w * vector[*j];
                }
                scratch[i] = acc;
            }
            vector.copy_from_slice(&scratch);
        }
        orthonormalize(&mut basis);
    }
    basis
}

/// Gram-Schmidt modifié — numériquement plus stable que la version classique,
/// et c'est ce qui compte ici : on réorthonormalise vingt-quatre fois de suite,
/// donc la moindre dérive s'accumule.
fn orthonormalize(basis: &mut [Vec<f32>]) {
    for i in 0..basis.len() {
        for j in 0..i {
            let dot: f32 = basis[i]
                .iter()
                .zip(basis[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            for idx in 0..basis[i].len() {
                basis[i][idx] -= dot * basis[j][idx];
            }
        }
        let norm: f32 = basis[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            basis[i].iter_mut().for_each(|x| *x /= norm);
        }
    }
}

/// Ramène un vecteur à la norme 1. Rend `false` s'il est nul — auquel cas
/// l'auteur n'a pas de position et il vaut mieux ne pas en inventer une.
fn normalize(v: &mut [f32; DIM]) -> bool {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-6 {
        return false;
    }
    v.iter_mut().for_each(|x| *x /= norm);
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// Lecture de Redis
// ═══════════════════════════════════════════════════════════════════════════

/// Parcourt les `cooccur:pair:*` et en tire les arêtes.
///
/// `SCAN` et non `KEYS` : `KEYS` bloque le serveur Redis le temps de parcourir
/// tout l'espace de clés, et ce Redis sert aussi les caches de fil.
async fn read_cooccurrence(cache: &CacheManager) -> redis::RedisResult<Vec<(String, String, f64)>> {
    const PREFIX: &str = "cooccur:pair:";
    let mut edges = Vec::new();
    let mut cursor: u64 = 0;

    loop {
        let (next, keys): (u64, Vec<String>) = {
            let mut c = cache.conn.lock().await;
            redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(format!("{PREFIX}*"))
                .arg("COUNT")
                .arg(500)
                .query_async(&mut *c)
                .await?
        };

        for key in keys {
            let Some(author) = key.strip_prefix(PREFIX) else {
                continue;
            };
            let pairs: Vec<(String, f64)> = {
                let mut c = cache.conn.lock().await;
                c.zrevrange_withscores(&key, 0, 63).await.unwrap_or_default()
            };
            for (other, score) in pairs {
                if score > 0.0 {
                    edges.push((author.to_string(), other, score));
                }
            }
        }

        cursor = next;
        if cursor == 0 {
            break;
        }
    }
    Ok(edges)
}

// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Construit deux communautés nettement séparées : les auteurs `a0..a29`
    /// sont co-aimés entre eux, `b0..b29` entre eux, et rien ne traverse.
    fn deux_communautes() -> Vec<(String, String, f64)> {
        let mut edges = Vec::new();
        for prefix in ["a", "b"] {
            for i in 0..30 {
                for j in 0..30 {
                    if i != j {
                        edges.push((format!("{prefix}{i}"), format!("{prefix}{j}"), 1.0));
                    }
                }
            }
        }
        edges
    }

    #[test]
    fn un_espace_vide_ne_pretend_rien() {
        let space = CollabSpace::from_edges(Vec::new());
        assert!(!space.is_usable());
        assert!(space.author("a0").is_none());
    }

    /// Sous le plancher, on préfère ne rien rendre plutôt qu'un placement qui
    /// aurait l'air d'un signal.
    #[test]
    fn sous_le_plancher_d_auteurs_l_espace_reste_vide() {
        let edges: Vec<(String, String, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (format!("x{i}"), format!("x{j}"), 1.0)))
            .filter(|(a, b, _)| a != b)
            .collect();
        let space = CollabSpace::from_edges(edges);
        assert!(!space.is_usable(), "5 auteurs ne font pas un espace");
    }

    /// LE test du module : deux communautés disjointes doivent se retrouver
    /// séparées dans l'espace. Un lecteur placé sur la première doit être plus
    /// proche de ses auteurs que de ceux de la seconde.
    #[test]
    fn deux_communautes_disjointes_se_separent() {
        let space = CollabSpace::from_edges(deux_communautes());
        assert!(space.is_usable(), "60 auteurs, l'espace doit exister");

        // Un lecteur qui n'a aimé que la communauté « a ».
        let affinites: Vec<(String, f64)> =
            (0..5).map(|i| (format!("a{i}"), 1.0)).collect();
        let lecteur = space.reader(&affinites).expect("lecteur placable");

        let proche = space.affinity(&lecteur, "a20").unwrap();
        let lointain = space.affinity(&lecteur, "b20").unwrap();
        assert!(
            proche > lointain,
            "un auteur de la communaute du lecteur doit etre plus proche : \
             a20={proche} b20={lointain}"
        );
    }

    /// Un lecteur sans historique connu n'a pas de position — et le module doit
    /// le dire, pas le placer au centre.
    #[test]
    fn un_lecteur_inconnu_n_a_pas_de_vecteur() {
        let space = CollabSpace::from_edges(deux_communautes());
        assert!(space.reader(&[]).is_none());
        assert!(space
            .reader(&[("compte-jamais-vu".to_string(), 1.0)])
            .is_none());
    }

    /// L'affinité doit vivre sur `[0,1]`, comme tous les autres traits.
    #[test]
    fn l_affinite_reste_dans_zero_un() {
        let space = CollabSpace::from_edges(deux_communautes());
        let affinites: Vec<(String, f64)> = (0..5).map(|i| (format!("a{i}"), 1.0)).collect();
        let lecteur = space.reader(&affinites).unwrap();
        for id in ["a0", "a29", "b0", "b29"] {
            let v = space.affinity(&lecteur, id).unwrap();
            assert!((0.0..=1.0).contains(&v), "{id} -> {v}");
        }
    }

    /// La reconstruction est déterministe : deux appels sur les mêmes arêtes
    /// doivent rendre exactement le même espace. Sans cela, le trait
    /// changerait de sens à chaque rafraîchissement.
    #[test]
    fn la_reconstruction_est_deterministe() {
        let a = CollabSpace::from_edges(deux_communautes());
        let b = CollabSpace::from_edges(deux_communautes());
        let va = a.author("a7").unwrap();
        let vb = b.author("a7").unwrap();
        for (x, y) in va.iter().zip(vb.iter()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    /// Les vecteurs sont unitaires : c'est ce qui fait du produit scalaire un
    /// cosinus, donc une quantité comparable d'un auteur à l'autre.
    #[test]
    fn les_vecteurs_sont_unitaires() {
        let space = CollabSpace::from_edges(deux_communautes());
        let v = space.author("b3").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norme = {norm}");
    }
}
