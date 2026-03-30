use anyhow::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::str::FromStr;

pub async fn init_pool(database_url: &str) -> Result<SqlitePool> {
    let opts = SqliteConnectOptions::from_str(database_url)?
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(opts)
        .await?;

    // Run migrations inline
    sqlx::query(include_str!("../migrations/001_init.sql"))
        .execute(&pool)
        .await?;
    sqlx::query(include_str!("../migrations/002_deployments.sql"))
        .execute(&pool)
        .await?;
    sqlx::query(include_str!("../migrations/003_cached_models.sql"))
        .execute(&pool)
        .await?;
    sqlx::query(include_str!("../migrations/004_split_deployments.sql"))
        .execute(&pool)
        .await?;

    Ok(pool)
}
