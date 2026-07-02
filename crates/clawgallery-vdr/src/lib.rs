use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

mod client;
mod duplicates;
mod index;
mod schema;
mod search;
mod store;

pub use client::DEFAULT_MAX_RETRIES;
pub use index::{ActiveIndexConfig, VdrStatus};
pub use search::{EmbeddingSearchHit, late_interaction_score};

pub const DEFAULT_EMBEDDING_URL: &str = "http://127.0.0.1:8765";
pub const DEFAULT_VDR_MODEL: &str = "vidore/colqwen2-v1.0";
pub const DEFAULT_DIMENSIONS: usize = 128;

#[derive(Debug, Clone)]
pub struct ImageDocument {
    pub id: String,
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct CaptionDocument {
    pub image_id: String,
    pub path: PathBuf,
    pub title: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub db_path: PathBuf,
    pub model: String,
    pub dimensions: usize,
    pub embedding_url: Option<String>,
    pub max_retries: usize,
    pub prune: bool,
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub db_path: PathBuf,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
    pub embedding_url: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOutcome {
    pub indexed_vectors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilarImageDuplicate {
    pub image_id: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilarImageGroup {
    pub representative_id: String,
    pub duplicates: Vec<SimilarImageDuplicate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingKind {
    Image,
    Caption,
}

impl EmbeddingKind {
    const fn as_str(self) -> &'static str {
        match self {
            EmbeddingKind::Image => "image",
            EmbeddingKind::Caption => "caption",
        }
    }

    const fn matched_field(self) -> &'static str {
        match self {
            EmbeddingKind::Image => "embedding_image",
            EmbeddingKind::Caption => "embedding_caption",
        }
    }

    const fn is_image(self) -> bool {
        matches!(self, EmbeddingKind::Image)
    }
}

#[derive(Debug)]
struct PendingEmbedding {
    image_id: String,
    path: PathBuf,
    sha256: String,
    content_hash: String,
    kind: EmbeddingKind,
    value: String,
}

pub fn sync(
    config: &SyncConfig,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<SyncOutcome> {
    let conn = store::open_store(&config.db_path)?;
    let captions = captions_by_path(captions);
    if config.prune {
        store::prune_inactive_vectors(&conn, &images)?;
    }
    store::update_active_vector_paths(&conn, &images, &config.model, config.dimensions)?;
    let pending =
        store::pending_embeddings(&conn, images, &captions, &config.model, config.dimensions)?;
    if pending.is_empty() {
        return Ok(SyncOutcome { indexed_vectors: 0 });
    }
    let inputs = pending
        .iter()
        .map(|item| client::EmbedInput {
            kind: item.kind.as_str().to_string(),
            role: "document".to_string(),
            value: item.value.clone(),
        })
        .collect();
    let url = client::resolve_embedding_url(config.embedding_url.as_deref());
    let response = client::embed_with_retries(
        &url,
        &config.model,
        config.dimensions,
        inputs,
        config.max_retries,
    )?;
    if response.embeddings.len() != pending.len() {
        bail!(
            "embedding server returned {} embedding(s) for {} input(s)",
            response.embeddings.len(),
            pending.len()
        );
    }
    let indexed_vectors = response.embeddings.len();
    for (item, vector) in pending.into_iter().zip(response.embeddings) {
        store::deactivate_existing_kind(&conn, &item.image_id, item.kind)?;
        store::insert_vector(&conn, &item, &response.model, config.dimensions, &vector)?;
    }
    Ok(SyncOutcome { indexed_vectors })
}

pub fn embedding_search(
    config: &SearchConfig,
    query: &str,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<Vec<EmbeddingSearchHit>> {
    search::embedding_search(config, query, images, captions)
}

pub fn status(db_path: &Path, active_images: usize) -> Result<VdrStatus> {
    let conn = store::open_store(db_path)?;
    index::status(db_path, active_images, &conn)
}

pub fn deactivate_image_vectors(db_path: &Path, image_id: &str) -> Result<()> {
    if !db_path.exists() {
        return Ok(());
    }
    let conn = store::open_store(db_path)?;
    store::deactivate_image_vectors(&conn, image_id)
}

pub fn similar_image_groups(
    db_path: &Path,
    images: &[ImageDocument],
    threshold: f64,
) -> Result<Vec<SimilarImageGroup>> {
    duplicates::similar_image_groups(db_path, images, threshold)
}

fn captions_by_path(captions: Vec<CaptionDocument>) -> HashMap<PathBuf, CaptionDocument> {
    captions
        .into_iter()
        .map(|caption| (caption.path.clone(), caption))
        .collect()
}
