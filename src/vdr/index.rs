use crate::AppPaths;
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug)]
pub(super) struct ActiveIndexConfig {
    pub(super) model: String,
    pub(super) dimensions: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct VdrStatus {
    active_images: usize,
    active_vectors: usize,
    model: Option<String>,
    dimensions: Option<usize>,
    db: PathBuf,
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

pub(super) fn status(paths: &AppPaths, conn: &Connection) -> Result<VdrStatus> {
    let active_vectors: usize = conn.query_row(
        "select count(*) from vdr_embeddings where active = 1",
        [],
        |row| row.get(0),
    )?;
    let config = latest_active_index_config(conn)?;
    Ok(VdrStatus {
        active_images: crate::latest_images(paths)?.len(),
        active_vectors,
        model: config.as_ref().map(|value| value.model.clone()),
        dimensions: config.map(|value| value.dimensions),
        db: paths.vdr_db.clone(),
    })
}
