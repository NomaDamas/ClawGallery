//! Discovery and protocol tests for the `inl` (im-not-llm) VDR backend.
//!
//! The ClawGallery VDR backend must find the central `inl` daemon under the
//! user's cache root by matching `GET /health` on (kind, model_id), adopt it,
//! and speak its `POST /embed_multimodal` protocol. These tests stand up a
//! fake daemon with the real discovery files and drive the installed CLI.

use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{Arc, Mutex},
    thread,
};

const MODEL: &str = "google/embeddinggemma-2";
const DIMENSIONS: usize = 768;

fn clawgallery_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

struct FakeInlServer {
    host: String,
    port: u16,
    requests: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
    _handle: thread::JoinHandle<()>,
}

impl FakeInlServer {
    /// A fake multimodal daemon. `kind` and `model_id` are advertised over
    /// `/health` so discovery's matching can be exercised.
    fn start(kind: &'static str, model: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake inl server");
        let address = listener.local_addr().expect("fake inl address");
        let host = address.ip().to_string();
        let port = address.port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                serve(stream, kind, model, &recorded);
            }
        });
        Self {
            host,
            port,
            requests,
            _handle: handle,
        }
    }

    fn requests(&self) -> Vec<(String, serde_json::Value)> {
        self.requests.lock().expect("requests lock").clone()
    }
}

fn serve(
    mut stream: TcpStream,
    kind: &'static str,
    model: &'static str,
    recorded: &Mutex<Vec<(String, serde_json::Value)>>,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader
        .read_line(&mut request_line)
        .expect("read request line")
        == 0
    {
        return;
    }
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
    let body = if request_line.contains("/health") {
        serde_json::json!({
            "status": "ok",
            "kind": kind,
            "model_id": model,
            "dimension": DIMENSIONS,
        })
    } else {
        let request: serde_json::Value =
            serde_json::from_slice(&raw).expect("embed_multimodal request json");
        recorded
            .lock()
            .expect("requests lock")
            .push((request_line.trim().to_string(), request.clone()));
        let inputs = request["inputs"].as_array().expect("inputs array");
        let embeddings: Vec<Vec<f32>> = inputs.iter().map(|_| vec![0.0_f32; DIMENSIONS]).collect();
        serde_json::json!({
            "model": model,
            "dimensions": DIMENSIONS,
            "embeddings": embeddings,
        })
    };
    let body = body.to_string();
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(reply.as_bytes()).expect("write response");
}

/// The im-not-llm root under an isolated home, as `inl_core::device::cache_root`
/// computes it: `$HOME/.cache/im-not-llm`, `%LOCALAPPDATA%\im-not-llm` on Windows.
fn inl_root(home: &Path) -> PathBuf {
    if cfg!(windows) {
        home.join("LocalAppData").join("im-not-llm")
    } else {
        home.join(".cache").join("im-not-llm")
    }
}

/// An isolated `$HOME` with a discovery directory recording `server`.
fn isolated_home(server: &FakeInlServer) -> tempfile::TempDir {
    let home = tempfile::tempdir().expect("home tempdir");
    let state_dir = inl_root(home.path())
        .join("serve")
        .join("fake-multimodal-instance");
    fs::create_dir_all(&state_dir).expect("create serve state dir");
    fs::write(state_dir.join("port"), format!("{}\n", server.port)).expect("write port");
    fs::write(state_dir.join("host"), format!("{}\n", server.host)).expect("write host");
    home
}

fn image_library() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog.png bytes").expect("write image");
    assert_success(run_without_embedding_url(&config, &["init"]));
    assert_success(run_without_embedding_url(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
    ));
    (temp, config)
}

/// Run the CLI with `$HOME` pointed at an isolated directory and `PATH`
/// emptied so a missing `inl` is deterministic.
fn run_with_home(config: &Path, home: &Path, args: &[&str]) -> Output {
    let empty_path = home.join("empty-path");
    fs::create_dir_all(&empty_path).expect("create empty PATH dir");
    Command::new(clawgallery_bin())
        .env("HOME", home)
        .env("LOCALAPPDATA", home.join("LocalAppData"))
        .env("PATH", &empty_path)
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env_remove("CLAWGALLERY_VDR_EMBEDDING_URL")
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args)
        .output()
        .expect("clawgallery command should run")
}

fn run_without_embedding_url(config: &Path, args: &[&str]) -> Output {
    Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env_remove("CLAWGALLERY_VDR_EMBEDDING_URL")
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args)
        .output()
        .expect("clawgallery command should run")
}

fn assert_success(output: Output) -> String {
    if !output.status.success() {
        panic!(
            "command failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn vdr_sync_inl_adopts_matching_daemon_and_speaks_multimodal() {
    let server = FakeInlServer::start("multimodal", MODEL);
    let home = isolated_home(&server);
    let (_temp, config) = image_library();

    let synced = assert_success(run_with_home(
        &config,
        home.path(),
        &["vdr", "sync", "--backend", "inl"],
    ));

    assert!(
        synced.contains("adopted running inl multimodal server"),
        "sync should adopt the running daemon, got: {synced}"
    );
    assert!(synced.contains("indexed 1"), "got: {synced}");
    let requests = server.requests();
    let (line, body) = requests.first().expect("one embed request");
    assert!(line.starts_with("POST /embed_multimodal"), "got: {line}");
    assert_eq!(body["input_type"], "passage");
    assert!(
        body["inputs"][0]["images"][0].is_string(),
        "image input should carry base64 bytes, got: {body}"
    );
}

#[test]
fn embedding_search_auto_selects_inl_from_index_model() {
    let server = FakeInlServer::start("multimodal", MODEL);
    let home = isolated_home(&server);
    let (_temp, config) = image_library();
    assert_success(run_with_home(
        &config,
        home.path(),
        &["vdr", "sync", "--backend", "inl"],
    ));

    let search = assert_success(run_with_home(
        &config,
        home.path(),
        &["search", "--mode", "embedding", "dog", "--json"],
    ));

    assert!(search.contains("dog.png"), "got: {search}");
}

#[test]
fn vdr_sync_inl_ignores_daemon_of_another_kind() {
    // Given: a live daemon that advertises the model under the wrong kind.
    let server = FakeInlServer::start("dense", MODEL);
    let home = isolated_home(&server);
    let (_temp, config) = image_library();

    // When: sync runs the inl backend with no `inl` binary on PATH.
    let output = run_with_home(&config, home.path(), &["vdr", "sync", "--backend", "inl"]);

    // Then: discovery rejects the wrong kind, and the missing launcher names
    // the install command instead of silently embedding through it.
    assert!(
        !output.status.success(),
        "wrong-kind daemon should not be adopted"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("inl-server") && stderr.contains("install"),
        "expected install hint, got: {stderr}"
    );
    assert!(
        server.requests().is_empty(),
        "no embedding should reach the daemon"
    );
}
