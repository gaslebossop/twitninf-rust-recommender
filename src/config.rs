use anyhow::{bail, Result};
use std::env;
use tracing::warn;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    /// Adresse d'écoute. Reste `127.0.0.1` par défaut : ce service n'a aucune
    /// authentification sur ses routes de scoring, il ne doit jamais se
    /// retrouver exposé par accident. On ne l'élargit que pour l'IP du tunnel
    /// WireGuard, afin que l'instance API du second VPS puisse l'interroger —
    /// sans quoi elle retombe silencieusement sur le classement JS et perd les
    /// événements CTR qui alimentent le modèle.
    pub bind_host: String,
    pub db_host: String,
    pub db_port: u16,
    pub db_name: String,
    pub db_user: String,
    pub db_password: String,
    pub db_pool_size: usize,
    pub redis_url: String,
    pub log_level: String,
    pub node_api_url: String,
    /// Clé admin pour /admin/* — DOIT être définie via ADMIN_SECRET
    pub admin_secret: String,
    /// Clé interne service-to-service pour /track et /invalidate
    pub internal_secret: String,
}

const DEFAULT_ADMIN_SENTINEL: &str = "changeme-admin-secret";
const DEFAULT_INTERNAL_SENTINEL: &str = "changeme-internal-secret";

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let admin_secret =
            env::var("ADMIN_SECRET").unwrap_or_else(|_| DEFAULT_ADMIN_SENTINEL.to_string());
        if admin_secret == DEFAULT_ADMIN_SENTINEL {
            bail!(
                "ADMIN_SECRET environment variable is not set. \
                 Refusing to start with the default insecure value. \
                 Set a strong random secret: ADMIN_SECRET=$(openssl rand -hex 32)"
            );
        }

        let internal_secret =
            env::var("INTERNAL_SECRET").unwrap_or_else(|_| DEFAULT_INTERNAL_SENTINEL.to_string());
        if internal_secret == DEFAULT_INTERNAL_SENTINEL {
            warn!(
                "INTERNAL_SECRET is using the default insecure value. \
                 Set INTERNAL_SECRET in production to protect /track and /invalidate."
            );
        }

        let db_password = env::var("DB_PASSWORD").unwrap_or_else(|_| String::new());
        if db_password.is_empty() {
            warn!("DB_PASSWORD is not set — connecting to PostgreSQL without a password.");
        }

        Ok(Config {
            port: env::var("RUST_PORT")
                .unwrap_or_else(|_| "3002".to_string())
                .parse()?,
            bind_host: env::var("RUST_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            db_host: env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string()),
            db_port: env::var("DB_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse()?,
            db_name: env::var("DB_NAME").unwrap_or_else(|_| "twitninf".to_string()),
            db_user: env::var("DB_USER").unwrap_or_else(|_| "admin".to_string()),
            db_password,
            db_pool_size: env::var("DB_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()?,
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            node_api_url: env::var("NODE_API_URL")
                .unwrap_or_else(|_| "http://localhost:3001".to_string()),
            admin_secret,
            internal_secret,
        })
    }

    pub fn pg_config(&self) -> deadpool_postgres::Config {
        let mut cfg = deadpool_postgres::Config::new();
        cfg.host = Some(self.db_host.clone());
        cfg.port = Some(self.db_port);
        cfg.dbname = Some(self.db_name.clone());
        cfg.user = Some(self.db_user.clone());
        cfg.password = Some(self.db_password.clone());
        cfg.pool = Some(deadpool_postgres::PoolConfig {
            max_size: self.db_pool_size,
            ..Default::default()
        });
        cfg
    }
}
