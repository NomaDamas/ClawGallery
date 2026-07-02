use crate::{CaptionDocument, ImageDocument, SearchConfig};
use anyhow::{Result, bail};
use std::{collections::HashMap, path::PathBuf};

use super::{
    client::{default_model, embed, query_input, resolve_embedding_url},
    index::{ActiveIndexConfig, latest_active_index_config},
    store::{active_vectors, open_store},
};

#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingSearchHit {
    pub path: PathBuf,
    pub title: String,
    pub description: String,
    pub score: f64,
    pub matched_field: &'static str,
    pub matched_atoms: Vec<String>,
    pub source: &'static str,
}

pub(super) fn embedding_search(
    config: &SearchConfig,
    query: &str,
    images: Vec<ImageDocument>,
    captions: Vec<CaptionDocument>,
) -> Result<Vec<EmbeddingSearchHit>> {
    let conn = open_store(&config.db_path)?;
    let url = resolve_embedding_url(config.embedding_url.as_deref());
    let index_config = match (config.model.clone(), config.dimensions) {
        (Some(model), Some(dimensions)) => ActiveIndexConfig { model, dimensions },
        _ => latest_active_index_config(&conn)?.unwrap_or_else(default_index_config),
    };
    let response = embed(
        &url,
        &index_config.model,
        index_config.dimensions,
        vec![query_input(query)],
    )?;
    let query_vectors = response
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("embedding server returned no query embedding"))?;
    let active_images: HashMap<String, ImageDocument> = images
        .into_iter()
        .map(|image| (image.id.clone(), image))
        .collect();
    let captions: HashMap<PathBuf, CaptionDocument> = captions
        .into_iter()
        .map(|caption| (caption.path.clone(), caption))
        .collect();
    let mut best_by_image = HashMap::new();
    for stored in active_vectors(
        &conn,
        &active_images,
        &index_config.model,
        index_config.dimensions,
    )? {
        let Some(image) = active_images.get(&stored.image_id) else {
            continue;
        };
        let score = late_interaction_score(&query_vectors, &stored.vectors)?;
        let caption = captions.get(&image.path);
        let hit = EmbeddingSearchHit {
            path: image.path.clone(),
            title: caption
                .map(|c| c.title.clone())
                .unwrap_or_else(|| "<missing>".to_string()),
            description: caption
                .map(|c| c.description.clone())
                .unwrap_or_else(|| "<missing>".to_string()),
            score,
            matched_field: stored.kind.matched_field(),
            matched_atoms: vec![query.to_string()],
            source: "embedding",
        };
        best_by_image
            .entry(stored.image_id)
            .and_modify(|existing: &mut EmbeddingSearchHit| {
                if hit.score > existing.score {
                    existing.path = hit.path.clone();
                    existing.title = hit.title.clone();
                    existing.description = hit.description.clone();
                    existing.score = hit.score;
                    existing.matched_field = hit.matched_field;
                    existing.matched_atoms = hit.matched_atoms.clone();
                    existing.source = hit.source;
                }
            })
            .or_insert(hit);
    }
    let mut hits: Vec<_> = best_by_image.into_values().collect();
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    hits.truncate(config.limit);
    Ok(hits)
}

fn default_index_config() -> ActiveIndexConfig {
    ActiveIndexConfig {
        model: default_model().to_string(),
        dimensions: super::DEFAULT_DIMENSIONS,
    }
}

/// ColBERT-style late-interaction MaxSim: for each query token vector, take
/// the maximum cosine similarity over all document vectors, then average over
/// query tokens. With single-vector inputs (1x1) this degenerates to plain
/// cosine similarity, so legacy single-vector indexes keep working.
pub fn late_interaction_score(query: &[Vec<f32>], document: &[Vec<f32>]) -> Result<f64> {
    if query.is_empty() || document.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0_f64;
    for query_vector in query {
        let mut best = f64::NEG_INFINITY;
        for document_vector in document {
            let score = cosine_similarity(query_vector, document_vector)?;
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    Ok(total / query.len() as f64)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f64> {
    if left.len() != right.len() {
        bail!(
            "embedding dimension mismatch: query has {}, row has {}",
            left.len(),
            right.len()
        );
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let l = f64::from(*left_value);
        let r = f64::from(*right_value);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}
