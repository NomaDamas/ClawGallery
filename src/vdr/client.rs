use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;

use super::{DEFAULT_EMBEDDING_URL, DEFAULT_VDR_MODEL};

#[derive(Debug, Serialize)]
pub(super) struct EmbedInput {
    pub(super) kind: String,
    pub(super) role: String,
    pub(super) value: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct EmbedResponse {
    pub(super) model: String,
    pub(super) embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
}

pub(super) fn resolve_embedding_url(cli_url: Option<&str>) -> String {
    cli_url
        .map(str::to_string)
        .or_else(|| env::var("CLAWGALLERY_VDR_EMBEDDING_URL").ok())
        .unwrap_or_else(|| DEFAULT_EMBEDDING_URL.to_string())
}

pub(super) fn embed(
    url: &str,
    model: &str,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
) -> Result<EmbedResponse> {
    let endpoint = format!("{}/embed", url.trim_end_matches('/'));
    let request = EmbedRequest {
        model: model.to_string(),
        dimensions,
        inputs,
    };
    let response: Value = reqwest::blocking::Client::new()
        .post(&endpoint)
        .json(&request)
        .send()
        .with_context(|| format!("failed to connect to VDR embedding server at {url}"))?
        .error_for_status()
        .with_context(|| format!("VDR embedding server returned an error at {url}"))?
        .json()?;
    let parsed: EmbedResponse = serde_json::from_value(response)?;
    if parsed.embeddings.iter().any(Vec::is_empty) {
        bail!("embedding server returned an empty vector");
    }
    Ok(parsed)
}

pub(super) fn query_input(query: &str) -> EmbedInput {
    EmbedInput {
        kind: "text".to_string(),
        role: "query".to_string(),
        value: query.to_string(),
    }
}

pub(super) fn default_model() -> &'static str {
    DEFAULT_VDR_MODEL
}
