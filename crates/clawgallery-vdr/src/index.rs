use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveIndexConfig {
    pub model: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VdrStatus {
    pub active_images: usize,
    pub active_vectors: usize,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
    pub db: PathBuf,
}

pub(super) fn latest_active_index_config(conn: &Connection) -> Result<Option<ActiveIndexConfig>> {
    Ok(conn
        .query_row(
            "select model, dimensions from vdr_embeddings
             where active = 1 order by indexed_at desc limit 1",
            [],
            |row| {
                Ok(ActiveIndexConfig {
                    model: row.get(0)?,
                    dimensions: row.get(1)?,
                })
            },
        )
        .optional()?)
}

pub(super) fn status(db_path: &Path, active_images: usize, conn: &Connection) -> Result<VdrStatus> {
    let active_vectors: usize = conn.query_row(
        "select count(*) from vdr_embeddings where active = 1",
        [],
        |row| row.get(0),
    )?;
    let config = latest_active_index_config(conn)?;
    Ok(VdrStatus {
        active_images,
        active_vectors,
        model: config.as_ref().map(|value| value.model.clone()),
        dimensions: config.map(|value| value.dimensions),
        db: db_path.to_path_buf(),
    })
}
