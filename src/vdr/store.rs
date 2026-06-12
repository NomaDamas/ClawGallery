use crate::{AppPaths, CaptionRecord, ImageRecord};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use super::{EmbeddingKind, PendingEmbedding};

pub(super) fn open_store(paths: &AppPaths) -> Result<Connection> {
    if let Some(parent) = paths.vdr_db.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(&paths.vdr_db)
        .with_context(|| format!("failed to open {}", paths.vdr_db.display()))?;
    conn.execute_batch(
        "create table if not exists vdr_embeddings (
            id integer primary key,
            image_id text not null,
            path text not null,
            sha256 text not null,
            kind text not null,
            model text not null,
            dimensions integer not null,
            content_hash text not null default '',
            vector_json text not null,
            active integer not null,
            indexed_at text not null
        );
        create index if not exists vdr_embeddings_lookup
            on vdr_embeddings(image_id, sha256, kind, model, dimensions, active);
        create index if not exists vdr_embeddings_content_lookup
            on vdr_embeddings(image_id, content_hash, kind, model, dimensions, active);
        create index if not exists vdr_embeddings_active
            on vdr_embeddings(active);",
    )?;
    migrate_content_hash(&conn)?;
    Ok(conn)
}

pub(super) fn update_active_vector_paths(
    conn: &Connection,
    active_images: &[ImageRecord],
    model: &str,
    dimensions: usize,
) -> Result<()> {
    for image in active_images {
        let path = image.path.to_string_lossy();
        conn.execute(
            "update vdr_embeddings set path = ?1
             where image_id = ?2 and model = ?3 and dimensions = ?4 and active = 1 and path <> ?1",
            params![path.as_ref(), image.id, model, dimensions],
        )?;
    }
    Ok(())
}
pub(super) fn pending_embeddings(
    conn: &Connection,
    images: Vec<ImageRecord>,
    captions: &HashMap<PathBuf, CaptionRecord>,
    model: &str,
    dimensions: usize,
) -> Result<Vec<PendingEmbedding>> {
    let mut pending = Vec::new();
    for image in images {
        if !has_current_vector(
            conn,
            &image.id,
            &image.sha256,
            &image.sha256,
            EmbeddingKind::Image,
            model,
            dimensions,
        )? {
            pending.push(PendingEmbedding {
                image_id: image.id.clone(),
                path: image.path.clone(),
                sha256: image.sha256.clone(),
                content_hash: image.sha256.clone(),
                kind: EmbeddingKind::Image,
                value: image.path.display().to_string(),
            });
        }
        if let Some(caption) = captions.get(&image.path) {
            let caption_text = caption_text(caption);
            let caption_hash = content_hash(&caption_text);
            if !has_current_vector(
                conn,
                &image.id,
                &image.sha256,
                &caption_hash,
                EmbeddingKind::Caption,
                model,
                dimensions,
            )? {
                pending.push(PendingEmbedding {
                    image_id: image.id,
                    path: image.path,
                    sha256: image.sha256,
                    content_hash: caption_hash,
                    kind: EmbeddingKind::Caption,
                    value: caption_text,
                });
            }
        }
    }
    Ok(pending)
}

pub(super) fn deactivate_existing_kind(
    conn: &Connection,
    image_id: &str,
    kind: EmbeddingKind,
) -> Result<()> {
    conn.execute(
        "update vdr_embeddings set active = 0 where image_id = ?1 and kind = ?2",
        params![image_id, kind.as_str()],
    )?;
    Ok(())
}

pub(super) fn prune_inactive_vectors(
    conn: &Connection,
    active_images: &[ImageRecord],
) -> Result<()> {
    let active_ids: HashSet<&str> = active_images
        .iter()
        .map(|image| image.id.as_str())
        .collect();
    let mut stmt = conn.prepare("select distinct image_id from vdr_embeddings where active = 1")?;
    let ids: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for id in ids {
        if !active_ids.contains(id.as_str()) {
            conn.execute(
                "update vdr_embeddings set active = 0 where image_id = ?1",
                params![id],
            )?;
        }
    }
    Ok(())
}

pub(super) fn insert_vector(
    conn: &Connection,
    item: &PendingEmbedding,
    model: &str,
    dimensions: usize,
    vectors: &[Vec<f32>],
) -> Result<()> {
    let path = item.path.to_string_lossy();
    conn.execute(
        "insert into vdr_embeddings
         (image_id, path, sha256, kind, model, dimensions, content_hash, vector_json, active, indexed_at)
         values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9)",
        params![
            item.image_id,
            path.as_ref(),
            item.sha256,
            item.kind.as_str(),
            model,
            dimensions,
            item.content_hash,
            serde_json::to_string(vectors)?,
            Utc::now().to_rfc3339()
        ],
    )?;
    Ok(())
}

pub(super) fn active_vectors(
    conn: &Connection,
    active_images: &HashMap<String, ImageRecord>,
    model: &str,
    dimensions: usize,
) -> Result<Vec<StoredVector>> {
    let mut stmt = conn.prepare(
        "select image_id, kind, vector_json from vdr_embeddings
         where active = 1 and model = ?1 and dimensions = ?2",
    )?;
    let rows = stmt.query_map(params![model, dimensions], |row| {
        let image_id: String = row.get(0)?;
        let kind: String = row.get(1)?;
        let vector_json: String = row.get(2)?;
        Ok((image_id, kind, vector_json))
    })?;
    let mut vectors = Vec::new();
    for row in rows {
        let (image_id, kind, vector_json) = row?;
        if !active_images.contains_key(&image_id) {
            continue;
        }
        let kind = match kind.as_str() {
            "image" => EmbeddingKind::Image,
            "caption" => EmbeddingKind::Caption,
            _ => continue,
        };
        vectors.push(StoredVector {
            image_id,
            kind,
            vectors: parse_stored_vectors(&vector_json)?,
        });
    }
    Ok(vectors)
}

fn has_current_vector(
    conn: &Connection,
    image_id: &str,
    sha256: &str,
    content_hash: &str,
    kind: EmbeddingKind,
    model: &str,
    dimensions: usize,
) -> Result<bool> {
    let count: usize = conn.query_row(
        "select count(*) from vdr_embeddings
         where image_id = ?1 and sha256 = ?2 and content_hash = ?3 and kind = ?4
         and model = ?5 and dimensions = ?6 and active = 1",
        params![
            image_id,
            sha256,
            content_hash,
            kind.as_str(),
            model,
            dimensions
        ],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn migrate_content_hash(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("pragma table_info(vdr_embeddings)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !columns.iter().any(|column| column == "content_hash") {
        conn.execute(
            "alter table vdr_embeddings add column content_hash text not null default ''",
            [],
        )?;
    }
    Ok(())
}

fn caption_text(caption: &CaptionRecord) -> String {
    format!("{}\n{}", caption.title, caption.description)
}

fn content_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Parses stored vector JSON, accepting both the legacy single-vector shape
/// (`[f32, …]`) and the multi-vector shape (`[[f32, …], …]`).
fn parse_stored_vectors(vector_json: &str) -> Result<Vec<Vec<f32>>> {
    let value: serde_json::Value = serde_json::from_str(vector_json)?;
    match value.as_array().and_then(|items| items.first()) {
        Some(serde_json::Value::Array(_)) => Ok(serde_json::from_value(value)?),
        Some(_) => Ok(vec![serde_json::from_value(value)?]),
        None => Ok(Vec::new()),
    }
}

#[derive(Debug)]
pub(super) struct StoredVector {
    pub(super) image_id: String,
    pub(super) kind: EmbeddingKind,
    pub(super) vectors: Vec<Vec<f32>>,
}
