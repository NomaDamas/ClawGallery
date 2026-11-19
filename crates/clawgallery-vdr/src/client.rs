use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, path::Path, thread, time::Duration};

use super::{DEFAULT_EMBEDDING_URL, DEFAULT_VDR_MODEL, EmbeddingProtocol};

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

#[derive(Debug)]
pub(super) struct SparseEmbedResponse {
    pub(super) model: String,
    pub(super) embeddings: Vec<crate::sparse::SparseVector>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    inputs: Vec<EmbedInput>,
}

/// One item of the `inl` `POST /embed_multimodal` body: either text or one
/// or more base64-encoded images.
#[derive(Debug, Serialize)]
struct MultimodalItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MultimodalRequest {
    inputs: Vec<MultimodalItem>,
    input_type: &'static str,
}

pub(super) fn resolve_embedding_url(cli_url: Option<&str>) -> String {
    cli_url
        .map(str::to_string)
        .or_else(|| env::var("CLAWGALLERY_VDR_EMBEDDING_URL").ok())
        .unwrap_or_else(|| DEFAULT_EMBEDDING_URL.to_string())
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
        format: None,
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

/// Embed `inputs` over the protocol the active backend speaks.
pub(super) fn embed_with_protocol(
    protocol: EmbeddingProtocol,
    url: &str,
    model: &str,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
    max_retries: usize,
) -> Result<EmbedResponse> {
    match protocol {
        EmbeddingProtocol::LateInteraction => {
            embed_with_retries(url, model, dimensions, inputs, max_retries)
        }
        EmbeddingProtocol::Multimodal => {
            embed_multimodal(url, model, dimensions, inputs, max_retries)
        }
    }
}

/// `POST /embed_multimodal` for the `inl` multimodal daemon. Image inputs
/// carry their file bytes as base64; text and caption inputs carry their text.
/// The daemon applies `input_type` (query/passage) over the whole batch.
pub(super) fn embed_multimodal(
    url: &str,
    model: &str,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
    max_retries: usize,
) -> Result<EmbedResponse> {
    let input_type = inputs
        .first()
        .map_or("passage", |input| match input.role.as_str() {
            "query" => "query",
            _ => "passage",
        });
    let mut items = Vec::with_capacity(inputs.len());
    for input in inputs {
        items.push(multimodal_item(input)?);
    }
    let request = MultimodalRequest {
        inputs: items,
        input_type,
    };
    let endpoint = format!("{}/embed_multimodal", url.trim_end_matches('/'));
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

fn multimodal_item(input: EmbedInput) -> Result<MultimodalItem> {
    if input.kind == "image" {
        Ok(MultimodalItem {
            text: None,
            images: vec![image_base64(Path::new(&input.value))?],
        })
    } else {
        Ok(MultimodalItem {
            text: Some(input.value),
            images: Vec::new(),
        })
    }
}

/// File-byte base64 for an image input. The `inl` vision stack decodes
/// PNG/JPEG/WebP; HEIC/HEIF is rejected here with a clear message instead of
/// being left to fail opaquely on the server.
fn image_base64(path: &Path) -> Result<String> {
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str())
        && (extension.eq_ignore_ascii_case("heic") || extension.eq_ignore_ascii_case("heif"))
    {
        bail!(
            "the inl multimodal backend decodes PNG/JPEG/WebP images, not HEIC/HEIF; convert {} to JPEG or PNG",
            path.display()
        );
    }
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read image {}", path.display()))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

struct HttpRetryError {
    context: &'static str,
    message: String,
}

fn send_with_retry<T: Serialize>(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    request: &T,
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

pub(super) fn embed_sparse(
    url: &str,
    model: &str,
    dimensions: usize,
    inputs: Vec<EmbedInput>,
    max_retries: usize,
) -> Result<SparseEmbedResponse> {
    let endpoint = format!("{}/embed", url.trim_end_matches('/'));
    let request = EmbedRequest {
        model: model.to_string(),
        dimensions,
        format: Some("sparse"),
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
        embeddings.push(crate::sparse::parse_sparse_vector(&value)?);
    }
    Ok(SparseEmbedResponse {
        model: raw.model,
        embeddings,
    })
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
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::{TcpListener, TcpStream},
        sync::{Arc, Mutex},
    };

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

    struct FakeHttpServer {
        url: String,
        request: Arc<Mutex<Option<Value>>>,
        _handle: std::thread::JoinHandle<()>,
    }

    fn spawn_http_server(status_line: &'static str, body: String) -> FakeHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake server");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("fake server addr")
        );
        let request = Arc::new(Mutex::new(None));
        let captured = Arc::clone(&request);
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                serve_once(stream, status_line, &body, &captured);
            }
        });
        FakeHttpServer {
            url,
            request,
            _handle: handle,
        }
    }

    fn serve_once(
        mut stream: TcpStream,
        status_line: &'static str,
        body: &str,
        captured: &Mutex<Option<Value>>,
    ) {
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
        let mut content_len = 0_usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).expect("read header") == 0 {
                return;
            }
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_len = value.trim().parse().expect("content length");
            }
        }
        let mut raw = vec![0_u8; content_len];
        if content_len > 0 {
            reader.read_exact(&mut raw).expect("read request body");
        }
        if let Ok(value) = serde_json::from_slice::<Value>(&raw) {
            *captured.lock().expect("lock") = Some(value);
        }
        let reply = format!(
            "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(reply.as_bytes()).expect("write response");
    }

    fn multimodal_ok_body() -> String {
        serde_json::json!({
            "model": "google/embeddinggemma-2",
            "dimensions": 3,
            "embeddings": [[0.25, 0.5, 0.75]],
        })
        .to_string()
    }

    #[test]
    fn multimodal_posts_base64_images_and_text_passages() {
        let mut image = tempfile::NamedTempFile::new().expect("temp file");
        std::io::Write::write_all(&mut image, b"IMG").expect("write image");
        let server = spawn_http_server("200 OK", multimodal_ok_body());

        let response = embed_multimodal(
            &server.url,
            "google/embeddinggemma-2",
            3,
            vec![
                EmbedInput {
                    kind: "image".to_string(),
                    role: "document".to_string(),
                    value: image.path().display().to_string(),
                },
                EmbedInput {
                    kind: "caption".to_string(),
                    role: "document".to_string(),
                    value: "a red square".to_string(),
                },
            ],
            0,
        )
        .expect("multimodal embed");

        let request = server
            .request
            .lock()
            .expect("lock")
            .clone()
            .expect("captured request");
        assert_eq!(request["input_type"], "passage");
        assert_eq!(
            request["inputs"][0]["images"][0],
            base64::engine::general_purpose::STANDARD.encode(b"IMG")
        );
        assert_eq!(request["inputs"][1]["text"], "a red square");
        assert!(request.get("model").is_none(), "inl takes no model field");
        assert_eq!(response.model, "google/embeddinggemma-2");
        assert_eq!(response.embeddings, vec![vec![vec![0.25, 0.5, 0.75]]]);
    }

    #[test]
    fn multimodal_uses_query_input_type_for_queries() {
        let server = spawn_http_server("200 OK", multimodal_ok_body());
        embed_multimodal(
            &server.url,
            "google/embeddinggemma-2",
            3,
            vec![query_input("a red square")],
            0,
        )
        .expect("multimodal query embed");
        let request = server
            .request
            .lock()
            .expect("lock")
            .clone()
            .expect("captured request");
        assert_eq!(request["input_type"], "query");
        assert_eq!(request["inputs"][0]["text"], "a red square");
    }

    #[test]
    fn multimodal_rejects_heic_before_posting() {
        let error = embed_multimodal(
            "http://127.0.0.1:1",
            "google/embeddinggemma-2",
            3,
            vec![EmbedInput {
                kind: "image".to_string(),
                role: "document".to_string(),
                value: "photo.HEIC".to_string(),
            }],
            0,
        )
        .expect_err("HEIC has no decoder in the inl vision stack");
        assert!(format!("{error:#}").contains("HEIC"), "got: {error:#}");
    }

    #[test]
    fn multimodal_error_redacts_url_credentials_and_query() {
        let server = spawn_http_server(
            "500 Internal Server Error",
            "{\"error\":\"boom\"}".to_string(),
        );
        let host = server.url.trim_start_matches("http://");
        let url = format!("http://user:secret@{host}?token=abc");
        let error = embed_multimodal(
            &url,
            "google/embeddinggemma-2",
            3,
            vec![query_input("x")],
            0,
        )
        .expect_err("500 should fail");
        let message = format!("{error:#}");
        assert!(message.contains("***@"), "got: {message}");
        assert!(message.contains("?…"), "got: {message}");
        assert!(!message.contains("secret"), "credential leaked: {message}");
    }
}
