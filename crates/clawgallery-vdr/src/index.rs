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
pub struct IndexChannelStatus {
    pub available: bool,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
    pub active_vectors: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VdrStatus {
    pub active_images: usize,
    pub active_vectors: usize,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
    pub db: PathBuf,
    pub keyword: bool,
    pub dense: IndexChannelStatus,
    pub sparse: IndexChannelStatus,
    pub hybrid: bool,
}

pub(super) fn latest_active_index_config(conn: &Connection) -> Result<Option<ActiveIndexConfig>> {
    latest_index_config(conn, Some("dense"))
}

pub(super) fn latest_sparse_index_config(conn: &Connection) -> Result<Option<ActiveIndexConfig>> {
    latest_index_config(conn, Some("sparse"))
}

pub(super) fn latest_any_index_config(conn: &Connection) -> Result<Option<ActiveIndexConfig>> {
    latest_index_config(conn, None)
}

fn latest_index_config(
    conn: &Connection,
    encoding: Option<&str>,
) -> Result<Option<ActiveIndexConfig>> {
    let config = match encoding {
        Some(encoding) => conn.query_row(
            "select model, dimensions from vdr_embeddings
             where active = 1 and encoding = ?1 order by indexed_at desc limit 1",
            [encoding],
            |row| {
                Ok(ActiveIndexConfig {
                    model: row.get(0)?,
                    dimensions: row.get(1)?,
                })
            },
        ),
        None => conn.query_row(
            "select model, dimensions from vdr_embeddings
             where active = 1 order by indexed_at desc limit 1",
            [],
            |row| {
                Ok(ActiveIndexConfig {
                    model: row.get(0)?,
                    dimensions: row.get(1)?,
                })
            },
        ),
    };
    Ok(config.optional()?)
}

fn active_vector_count(conn: &Connection, encoding: Option<&str>) -> Result<usize> {
    match encoding {
        Some(encoding) => Ok(conn.query_row(
            "select count(*) from vdr_embeddings where active = 1 and encoding = ?1",
            [encoding],
            |row| row.get(0),
        )?),
        None => Ok(conn.query_row(
            "select count(*) from vdr_embeddings where active = 1",
            [],
            |row| row.get(0),
        )?),
    }
}

fn channel_status(conn: &Connection, encoding: &str) -> Result<IndexChannelStatus> {
    let config = latest_index_config(conn, Some(encoding))?;
    Ok(IndexChannelStatus {
        available: config.is_some(),
        model: config.as_ref().map(|value| value.model.clone()),
        dimensions: config.as_ref().map(|value| value.dimensions),
        active_vectors: active_vector_count(conn, Some(encoding))?,
    })
}

pub(super) fn status(db_path: &Path, active_images: usize, conn: &Connection) -> Result<VdrStatus> {
    let active_vectors = active_vector_count(conn, None)?;
    let config = latest_any_index_config(conn)?;
    let dense = channel_status(conn, "dense")?;
    let sparse = channel_status(conn, "sparse")?;
    Ok(VdrStatus {
        active_images,
        active_vectors,
        model: config.as_ref().map(|value| value.model.clone()),
        dimensions: config.map(|value| value.dimensions),
        db: db_path.to_path_buf(),
        keyword: true,
        dense,
        sparse,
        hybrid: true,
    })
}
