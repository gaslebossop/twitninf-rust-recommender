use anyhow::Result;
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub db_host: String,
    pub db_port: u16,
    pub db_name: String,
    pub db_user: String,
    pub db_password: String,
    pub db_pool_size: usize,
    pub redis_url: String,
    pub log_level: String,
    pub node_api_url: String,
    pub admin_secret: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Config {
            port: env::var("RUST_PORT")
                .unwrap_or_else(|_| "3002".to_string())
                .parse()?,
            db_host: env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string()),
            db_port: env::var("DB_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse()?,
            db_name: env::var("DB_NAME").unwrap_or_else(|_| "twitninf".to_string()),
            db_user: env::var("DB_USER").unwrap_or_else(|_| "admin".to_string()),
            db_password: env::var("DB_PASSWORD").unwrap_or_else(|_| "REDACTED-ROTATED-CREDENTIAL".to_string()),
            db_pool_size: env::var("DB_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()?,
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            node_api_url: env::var("NODE_API_URL")
                .unwrap_or_else(|_| "http://localhost:3001".to_string()),
            admin_secret: env::var("ADMIN_SECRET")
                .unwrap_or_else(|_| "changeme-admin-secret".to_string()),
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
