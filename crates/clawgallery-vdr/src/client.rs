use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, thread, time::Duration};

use super::{DEFAULT_EMBEDDING_URL, DEFAULT_VDR_MODEL};

pub const DEFAULT_MAX_RETRIES: usize = 3;

#[derive(Debug, Serialize)]
pub(super) struct EmbedInput {
    pub(super) kind: String,
    pub(super) role: String,
    pub(super) value: String,
}

#[derive(Debug)]
pub(super) struct EmbedResponse {
    pub(super) model: String,
    pub(super) embeddings: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct RawEmbedResponse {
    model: String,
    embeddings: Vec<Value>,
    dimensions: Option<usize>,
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
    embed_with_retries(url, model, dimensions, inputs, DEFAULT_MAX_RETRIES)
}

pub(super) fn embed_with_retries(
    url: &str,
    model: &str,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
    max_retries: usize,
) -> Result<EmbedResponse> {
    let endpoint = format!("{}/embed", url.trim_end_matches('/'));
    let request = EmbedRequest {
        model: model.to_string(),
        dimensions,
        inputs,
    };
    let client = reqwest::blocking::Client::new();
    let response = send_with_retry(&client, &endpoint, &request, max_retries).map_err(|err| {
        anyhow!(
            "{} at {}: {}",
            err.context,
            redact_url(url),
            redact_message(&err.message)
        )
    })?;
    let response: Value = response.json()?;
    let raw: RawEmbedResponse = serde_json::from_value(response)?;
    if raw.model != model {
        bail!(
            "embedding server returned model {} but {} was requested; pass --model/--dimensions matching the running server",
            raw.model,
            model
        );
    }
    if let Some(response_dimensions) = raw.dimensions
        && response_dimensions != dimensions
    {
        bail!(
            "embedding server returned dimensions {} but {} was requested; pass --model/--dimensions matching the running server",
            response_dimensions,
            dimensions
        );
    }
    let mut embeddings = Vec::with_capacity(raw.embeddings.len());
    for value in raw.embeddings {
        let multivector = parse_multivector(value)?;
        for row in &multivector {
            if row.len() != dimensions {
                bail!(
                    "embedding server returned vector row with dimensions {} but {} was requested; pass --model/--dimensions matching the running server",
                    row.len(),
                    dimensions
                );
            }
        }
        embeddings.push(multivector);
    }
    Ok(EmbedResponse {
        model: raw.model,
        embeddings,
    })
}

struct HttpRetryError {
    context: &'static str,
    message: String,
}

fn send_with_retry(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    request: &EmbedRequest,
    max_retries: usize,
) -> Result<reqwest::blocking::Response, HttpRetryError> {
    for attempt in 0..=max_retries {
        match client.post(endpoint).json(request).send() {
            Ok(response) => {
                let status = response.status();
                let retry_after = retry_after_delay(response.headers());
                if !is_retryable_status(status) {
                    return response.error_for_status().map_err(|err| HttpRetryError {
                        context: "VDR embedding server returned an error",
                        message: err.to_string(),
                    });
                }
                if attempt == max_retries {
                    return response.error_for_status().map_err(|err| HttpRetryError {
                        context: "VDR embedding server returned an error",
                        message: err.to_string(),
                    });
                }
                thread::sleep(retry_after.unwrap_or_else(|| retry_delay(attempt)));
            }
            Err(err) => {
                if attempt == max_retries {
                    return Err(HttpRetryError {
                        context: "failed to connect to VDR embedding server",
                        message: err.to_string(),
                    });
                }
                thread::sleep(retry_delay(attempt));
            }
        }
    }
    Err(HttpRetryError {
        context: "failed to connect to VDR embedding server",
        message: "retry loop exhausted".to_string(),
    })
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retry_delay(attempt: usize) -> Duration {
    let base = 25_u64.saturating_mul(1_u64 << attempt.min(6));
    let jitter = ((attempt as u64 + 1) * 17) % 23;
    Duration::from_millis(base + jitter)
}

fn retry_after_delay(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Accepts a single vector (`[f32, …]`) or a multi-vector (`[[f32, …], …]`),
/// normalizing both to the multi-vector shape.
fn parse_multivector(value: Value) -> Result<Vec<Vec<f32>>> {
    let Value::Array(items) = value else {
        bail!("embedding server returned a non-array embedding");
    };
    match items.first() {
        None => bail!("embedding server returned an empty vector"),
        Some(Value::Array(_)) => {
            let rows: Vec<Vec<f32>> = serde_json::from_value(Value::Array(items))?;
            if rows.iter().any(Vec::is_empty) {
                bail!("embedding server returned an empty vector row");
            }
            Ok(rows)
        }
        Some(_) => {
            let row: Vec<f32> = serde_json::from_value(Value::Array(items))?;
            Ok(vec![row])
        }
    }
}

fn redact_url(url: &str) -> String {
    let (without_query, had_query) = url
        .split_once('?')
        .map_or((url, false), |(base, _)| (base, true));
    let mut redacted = without_query.to_string();
    if let Some(scheme_end) = redacted.find("://") {
        let authority_start = scheme_end + 3;
        if let Some(authority_end) = redacted[authority_start..].find('/') {
            redact_userinfo(
                &mut redacted,
                authority_start,
                authority_start + authority_end,
            );
        } else {
            let authority_end = redacted.len();
            redact_userinfo(&mut redacted, authority_start, authority_end);
        }
    }
    if had_query {
        redacted.push_str("?…");
    }
    redacted
}

fn redact_userinfo(url: &mut String, authority_start: usize, authority_end: usize) {
    if let Some(at_offset) = url[authority_start..authority_end].rfind('@') {
        url.replace_range(authority_start..authority_start + at_offset, "***");
    }
}

fn redact_message(message: &str) -> String {
    message
        .split_whitespace()
        .map(redact_url)
        .collect::<Vec<_>>()
        .join(" ")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_does_not_retry_non_429_4xx() {
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(reqwest::StatusCode::FORBIDDEN));
    }

    #[test]
    fn retry_after_header_parses_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "1".parse().unwrap());
        assert_eq!(retry_after_delay(&headers), Some(Duration::from_secs(1)));
    }
}
