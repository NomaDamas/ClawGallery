use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
};

fn python_interpreter() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        }
    })
}

struct ColqwenServer {
    child: Child,
    port: u16,
}

impl ColqwenServer {
    fn start() -> Self {
        let port = TcpListener::bind(("127.0.0.1", 0))
            .expect("ephemeral port")
            .local_addr()
            .expect("local address")
            .port();
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/colqwen2_server.py");
        let python = python_interpreter();
        let mut child = Command::new(python)
            .env("CLAWGALLERY_VDR_COLQWEN_FAKE", "1")
            .arg(script)
            .args(["--port", &port.to_string(), "--dimensions", "128"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("ColQwen server should start");
        let stdout = child.stdout.take().expect("server stdout");
        let mut ready = String::new();
        BufReader::new(stdout)
            .read_line(&mut ready)
            .expect("server readiness line");
        assert!(ready.contains("\"backend\": \"colqwen\""), "got: {ready}");
        assert!(ready.contains("\"fake\": true"), "got: {ready}");
        Self { child, port }
    }

    fn send(&self, content_type: &str, origin: Option<&str>, body: &str) -> String {
        let mut stream =
            TcpStream::connect(("127.0.0.1", self.port)).expect("server should accept requests");
        let origin_header = origin
            .map(|value| format!("Origin: {value}\r\n"))
            .unwrap_or_default();
        write!(
            stream,
            "POST /embed HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nContent-Type: {}\r\n{}Content-Length: {}\r\n\r\n{}",
            self.port,
            content_type,
            origin_header,
            body.len(),
            body
        )
        .expect("request should write");
        stream.shutdown(Shutdown::Write).expect("request shutdown");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("response should read");
        response
    }
}

impl Drop for ColqwenServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn colqwen_server_embeds_image_text_and_caption_inputs() {
    // Given: the fake ColQwen dense server.
    let server = ColqwenServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let image = temp.path().join("dog.png");
    fs::write(&image, b"png-bytes").expect("image");

    // When: image, text, and caption inputs are embedded together.
    let body = format!(
        r#"{{"model":"vidore/colqwen2-v1.0","dimensions":128,"inputs":[{{"kind":"image","role":"document","value":"{}"}},{{"kind":"text","role":"query","value":"dog"}},{{"kind":"caption","role":"document","value":"a puppy"}}]}}"#,
        image.display()
    );
    let response = server.send("application/json", None, &body);

    // Then: one 128-d multi-vector is returned per input.
    assert!(response.starts_with("HTTP/1.0 200"), "got: {response}");
    let json_start = response.find('{').expect("json body");
    let payload: serde_json::Value =
        serde_json::from_str(&response[json_start..]).expect("embed json");
    assert_eq!(payload["model"], "vidore/colqwen2-v1.0");
    assert_eq!(payload["dimensions"], 128);
    let embeddings = payload["embeddings"].as_array().expect("embeddings");
    assert_eq!(embeddings.len(), 3);
    for embedding in embeddings {
        let rows = embedding.as_array().expect("multivector");
        assert!(!rows.is_empty());
        assert_eq!(rows[0].as_array().expect("row").len(), 128);
    }
}

#[test]
fn colqwen_server_rejects_model_and_dimension_mismatch() {
    let server = ColqwenServer::start();
    let model = server.send(
        "application/json",
        None,
        r#"{"model":"wrong-model","dimensions":128,"inputs":[]}"#,
    );
    let dimensions = server.send(
        "application/json",
        None,
        r#"{"model":"vidore/colqwen2-v1.0","dimensions":64,"inputs":[]}"#,
    );
    assert!(model.starts_with("HTTP/1.0 400"), "got: {model}");
    assert!(model.contains("\"error\""), "got: {model}");
    assert!(dimensions.starts_with("HTTP/1.0 400"), "got: {dimensions}");
    assert!(dimensions.contains("\"error\""), "got: {dimensions}");
}

#[test]
fn colqwen_server_returns_json_for_malformed_requests() {
    let server = ColqwenServer::start();
    let response = server.send("application/json", None, "{");
    assert!(response.starts_with("HTTP/1.0 400"), "got: {response}");
    assert!(response.contains("\"error\""), "got: {response}");
}

#[test]
fn colqwen_server_rejects_cross_origin_requests() {
    let server = ColqwenServer::start();
    let response = server.send(
        "application/json",
        Some("https://attacker.example"),
        r#"{"inputs":[{"kind":"image","role":"query","value":"/etc/passwd"}]}"#,
    );
    assert!(response.starts_with("HTTP/1.0 403"), "got: {response}");
    assert!(response.contains("\"error\""), "got: {response}");
}

#[test]
fn colqwen_incomplete_hf_cache_is_diagnosed_before_model_load() {
    // Given: an incomplete Hugging Face snapshot with no weights.
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("hub/models--vidore--colqwen2-v1.0/snapshots/deadbeef");
    fs::create_dir_all(&snapshot).expect("snapshot dir");
    fs::write(snapshot.join("adapter_config.json"), b"{}").expect("adapter config");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/colqwen2_server.py");
    let python = python_interpreter();

    // When: the real (non-fake) server starts against that cache.
    let output = Command::new(python)
        .env("HF_HOME", temp.path())
        .env_remove("CLAWGALLERY_VDR_COLQWEN_FAKE")
        .arg(script)
        .args(["--port", "0"])
        .output()
        .expect("colqwen server should run");

    // Then: startup fails with an actionable incomplete-download diagnostic.
    assert!(!output.status.success(), "incomplete cache should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("incomplete") && stderr.contains("HF_HUB_DISABLE_XET"),
        "got: {stderr}"
    );
}
