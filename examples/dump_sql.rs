//! Ecrit sur la sortie standard le SQL de collecte reellement engendre.
//!
//! Sert a le faire valider par la base avant un deploiement : ce texte est
//! construit a l execution, et une erreur de syntaxe casserait CHAQUE requete
//! de fil — que le repli silencieux du client Node masquerait en simple baisse
//! de qualite.
//!
//!   cargo run --release --example dump_sql > /tmp/collecte.sql

fn main() {
    print!("{}", twitninf_recommender::services::recommender::candidates_sql());
}
