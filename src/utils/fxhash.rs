//! Table de hachage rapide pour des clés que personne ne choisit.
//!
//! ── Pourquoi ne pas garder le hacheur par défaut ────────────────────────────
//! `HashMap` et `HashSet` utilisent SipHash-1-3, qui est résistant aux
//! collisions provoquées : il empêche qu'un utilisateur, en choisissant ses
//! clés, transforme une table en liste chaînée. C'est le bon choix par défaut,
//! et il coûte une trentaine de nanosecondes sur une chaîne de 36 octets.
//!
//! Les ensembles du scoring n'ont pas ce problème : leurs clés sont des UUID
//! que la base a produits (abonnements, mutuels, second degré, auteurs) ou des
//! identifiants de tweets. Personne ne les choisit, donc il n'y a rien à
//! protéger — et D3 en interroge trois par candidat, 5100 par recommandation.
//!
//! ⚠ Ne pas étendre ce hacheur à une table dont les clés viennent d'une requête
//! entrante (un pseudo, un texte de recherche, une entête). Là, SipHash n'est
//! pas un luxe.
//!
//! L'algorithme est celui de `rustc-hash` (FxHash), lui-même repris de
//! Firefox : une multiplication et une rotation par bloc de huit octets.
//! Recopié ici plutôt qu'ajouté en dépendance — trente lignes contre une
//! caisse de plus dans l'arbre de compilation.

use std::hash::{BuildHasherDefault, Hasher};

/// Constante multiplicative de FxHash : les bits de la partie fractionnaire de
/// la racine de 5, choisis pour disperser les bits hauts vers les bas.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while rest.len() >= 8 {
            let (head, tail) = rest.split_at(8);
            self.add(u64::from_ne_bytes(head.try_into().unwrap()));
            rest = tail;
        }
        if rest.len() >= 4 {
            let (head, tail) = rest.split_at(4);
            self.add(u32::from_ne_bytes(head.try_into().unwrap()) as u64);
            rest = tail;
        }
        for &b in rest {
            self.add(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    /// ⚠ Le brassage final n'est pas décoratif.
    ///
    /// La dernière opération de `add` est une multiplication, et les bits BAS
    /// d'un produit ne dépendent que des bits bas des opérandes : ils portent
    /// donc très peu d'entropie. Or `hashbrown` — la table qui est derrière
    /// `HashMap` et `HashSet` — choisit le seau avec les bits bas.
    ///
    /// Sans rien, mille UUID consécutifs ne tombaient que dans **32** seaux sur
    /// 1024 : chaque seau devenait une liste de trente entrées à parcourir, et
    /// la « table de hachage rapide » aurait été plus LENTE que celle qu'elle
    /// remplace. C'est exactement la faute qu'a attrapée
    /// `les_uuid_se_dispersent`, écrit avant de mesurer quoi que ce soit.
    ///
    /// La rotation de 20 que `rustc-hash` 2.0 a ajoutée pour cette raison
    /// remonte à 404 seaux — mieux, mais toujours loin des 639 attendus d'un
    /// hachage idéal, parce que nos clés sont extrêmement structurées (des UUID
    /// qui ne diffèrent que par leurs derniers caractères). On finit donc par
    /// l'avalanche de `splitmix64` : deux multiplications et trois décalages,
    /// une paire de nanosecondes, et la dispersion rejoint l'idéal.
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.hash;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn un_ensemble_repond_comme_celui_de_la_bibliotheque() {
        // Le seul contrat qui compte : appartenance identique. Le hacheur peut
        // rendre n'importe quelles valeurs, tant que les mêmes clés se
        // retrouvent et que les autres non.
        let cles: Vec<String> = (0..2_000)
            .map(|i| format!("00000000-0000-0000-0000-{i:012}"))
            .collect();
        let rapide: FxHashSet<String> = cles.iter().cloned().collect();
        let standard: std::collections::HashSet<String> = cles.iter().cloned().collect();

        assert_eq!(rapide.len(), standard.len());
        for cle in &cles {
            assert!(rapide.contains(cle));
        }
        for absente in ["", "inconnu", "00000000-0000-0000-0000-999999999999"] {
            assert_eq!(rapide.contains(absente), standard.contains(absente));
        }
    }

    #[test]
    fn les_uuid_se_dispersent() {
        // Une dispersion catastrophique (tout dans le même seau) ne se verrait
        // pas dans le test ci-dessus, seulement en lenteur. On vérifie donc que
        // les empreintes des UUID consécutifs diffèrent bien, y compris sur
        // leurs bits BAS — ce sont eux qui choisissent le seau.
        use std::hash::Hash;
        let seaux: std::collections::HashSet<u64> = (0..1_000)
            .map(|i| {
                let mut h = FxHasher::default();
                format!("00000000-0000-0000-0000-{i:012}").hash(&mut h);
                h.finish() & 0x3ff
            })
            .collect();
        assert!(
            seaux.len() > 600,
            "dispersion trop faible : {} seaux distincts sur 1000 clés",
            seaux.len()
        );
    }
}
