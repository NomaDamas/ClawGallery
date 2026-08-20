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
mod sparse;
mod store;

pub use client::DEFAULT_MAX_RETRIES;
pub use index::{ActiveIndexConfig, VdrStatus};
pub use search::{EmbeddingSearchHit, late_interaction_score};
pub use sparse::{SparseVector, rrf_score};

pub const DEFAULT_EMBEDDING_URL: &str = "http://127.0.0.1:8765";
pub const DEFAULT_VDR_MODEL: &str = "vidore/colqwen2-v1.0";
pub const DEFAULT_DIMENSIONS: usize = 128;
pub const DEFAULT_VSPLADE_MODEL: &str = "NomaDamas/v-splade-efficient-mlx";
pub const DEFAULT_VSPLADE_DIMENSIONS: usize = 50368;
const SYNC_EMBEDDING_BATCH_SIZE: usize = 1;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorEncoding {
    #[default]
    Dense,
    Sparse,
}

#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub db_path: PathBuf,
    pub model: String,
    pub dimensions: usize,
    pub embedding_url: Option<String>,
    pub max_retries: usize,
    pub prune: bool,
    pub encoding: VectorEncoding,
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
    let include_captions = config.encoding != VectorEncoding::Sparse;
    let pending = store::pending_embeddings(
        &conn,
        images,
        &captions,
        &config.model,
        config.dimensions,
        include_captions,
    )?;
    if pending.is_empty() {
        return Ok(SyncOutcome { indexed_vectors: 0 });
    }
    let url = client::resolve_embedding_url(config.embedding_url.as_deref());
    let mut indexed_vectors = 0;
    for batch in pending.chunks(SYNC_EMBEDDING_BATCH_SIZE) {
        let inputs = batch
            .iter()
            .map(|item| client::EmbedInput {
                kind: item.kind.as_str().to_string(),
                role: "document".to_string(),
                value: item.value.clone(),
            })
            .collect();
        indexed_vectors += match config.encoding {
            VectorEncoding::Dense => {
                let response = client::embed_with_retries(
                    &url,
                    &config.model,
                    config.dimensions,
                    inputs,
                    config.max_retries,
                )?;
                if response.embeddings.len() != batch.len() {
                    bail!(
                        "embedding server returned {} embedding(s) for {} input(s)",
                        response.embeddings.len(),
                        batch.len()
                    );
                }
                for (item, vector) in batch.iter().zip(response.embeddings) {
                    let tx = conn.unchecked_transaction()?;
                    store::deactivate_stale_vectors(
                        &tx,
                        &item.image_id,
                        &item.sha256,
                        &config.model,
                    )?;
                    store::deactivate_existing_kind(&tx, &item.image_id, item.kind, &config.model)?;
                    store::insert_vector(&tx, item, &response.model, config.dimensions, &vector)?;
                    tx.commit()?;
                }
                batch.len()
            }
            VectorEncoding::Sparse => {
                let response = client::embed_sparse(
                    &url,
                    &config.model,
                    config.dimensions,
                    inputs,
                    config.max_retries,
                )?;
                if response.embeddings.len() != batch.len() {
                    bail!(
                        "embedding server returned {} embedding(s) for {} input(s)",
                        response.embeddings.len(),
                        batch.len()
                    );
                }
                for (item, vector) in batch.iter().zip(response.embeddings) {
                    let tx = conn.unchecked_transaction()?;
                    store::deactivate_stale_vectors(
                        &tx,
                        &item.image_id,
                        &item.sha256,
                        &config.model,
                    )?;
                    store::deactivate_existing_kind(&tx, &item.image_id, item.kind, &config.model)?;
                    store::insert_sparse_vector(
                        &tx,
                        item,
                        &response.model,
                        config.dimensions,
                        &vector,
                    )?;
                    tx.commit()?;
                }
                batch.len()
            }
        };
    }
    Ok(SyncOutcome { indexed_vectors })
}

pub fn pending_embedding_count(
    config: &SyncConfig,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<usize> {
    let conn = store::open_store(&config.db_path)?;
    let captions = captions_by_path(captions);
    if config.prune {
        store::prune_inactive_vectors(&conn, &images)?;
    }
    store::update_active_vector_paths(&conn, &images, &config.model, config.dimensions)?;
    let pending = store::pending_embeddings(
        &conn,
        images,
        &captions,
        &config.model,
        config.dimensions,
        config.encoding != VectorEncoding::Sparse,
    )?;
    Ok(pending.len())
}

pub fn embedding_search(
    config: &SearchConfig,
    query: &str,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<Vec<EmbeddingSearchHit>> {
    search::embedding_search(config, query, images, captions)
}

pub fn lexical_search(
    config: &SearchConfig,
    query: &str,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<Vec<EmbeddingSearchHit>> {
    search::lexical_search(config, query, images, captions)
}

pub fn latest_sparse_index_config(db_path: &Path) -> Result<Option<ActiveIndexConfig>> {
    latest_index_config(db_path, VectorEncoding::Sparse)
}

pub fn latest_dense_index_config(db_path: &Path) -> Result<Option<ActiveIndexConfig>> {
    latest_index_config(db_path, VectorEncoding::Dense)
}

fn latest_index_config(
    db_path: &Path,
    encoding: VectorEncoding,
) -> Result<Option<ActiveIndexConfig>> {
    if !db_path.exists() {
        return Ok(None);
    }
    let conn = store::open_store(db_path)?;
    match encoding {
        VectorEncoding::Dense => index::latest_active_index_config(&conn),
        VectorEncoding::Sparse => index::latest_sparse_index_config(&conn),
    }
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
