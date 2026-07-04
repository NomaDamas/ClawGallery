use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

pub(crate) fn run_without_embedding_url(config: &Path, args: &[&str], mlx_fake: bool) -> Output {
    let mut command = Command::new(bin());
    if mlx_fake {
        command.env("CLAWGALLERY_VDR_MLX_FAKE", "1");
    }
    command
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env_remove("CLAWGALLERY_VDR_EMBEDDING_URL")
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args)
        .output()
        .expect("clawgallery command should run")
}

pub(crate) fn one_image_library() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write image");
    assert_success(run_without_embedding_url(&config, &["init"], false));
    assert_success(run_without_embedding_url(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        false,
    ));
    (temp, config)
}

pub(crate) fn assert_success(output: Output) -> String {
    if !output.status.success() {
        panic!(
            "command failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).to_string()
}

pub(crate) struct FakeEmbeddingServer {
    url: String,
    requests: Arc<AtomicUsize>,
    _handle: JoinHandle<()>,
}

impl FakeEmbeddingServer {
    pub(crate) fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake embedding server");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("fake server local addr")
        );
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                request_count.fetch_add(1, Ordering::SeqCst);
                handle_request(stream);
            }
        });
        Self {
            url,
            requests,
            _handle: handle,
        }
    }

    pub(crate) fn url(&self) -> &str {
        &self.url
    }

    pub(crate) fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

fn handle_request(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).expect("read header") == 0 {
            return;
        }
        if line == "\r\n" {
            break;
        }
    }
    let body = serde_json::json!({
        "model": "test-model",
        "dimensions": 4,
        "embeddings": [[1.0, 0.0, 0.0, 0.0]],
    })
    .to_string();
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(reply.as_bytes()).expect("write response");
}
