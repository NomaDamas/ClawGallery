use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

pub(crate) fn run(config: &Path, args: &[&str], embedding_url: &str) -> Output {
    Command::new(bin())
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env("CLAWGALLERY_VDR_EMBEDDING_URL", embedding_url)
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args)
        .output()
        .expect("clawgallery command should run")
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

pub(crate) fn write_caption(
    config: &Path,
    image_id: &str,
    image_path: &Path,
    title: &str,
    description: &str,
) {
    let caption = serde_json::json!({
        "image_id": image_id,
        "path": image_path,
        "title": title,
        "description": description,
        "model": "test",
        "provider": "test",
        "created_at": "2026-06-05T00:00:00Z",
        "filename_meaningful": false
    });
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.join("captions.jsonl"))
        .expect("open captions");
    writeln!(file, "{caption}").expect("write caption");
}

pub(crate) fn image_id_for(config: &Path, name: &str) -> (String, PathBuf) {
    let raw = fs::read_to_string(config.join("images.jsonl")).expect("images jsonl");
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("image record"))
        .find_map(|record| {
            let path = PathBuf::from(record["path"].as_str()?);
            path.ends_with(name)
                .then(|| (record["id"].as_str().expect("image id").to_string(), path))
        })
        .expect("image record for name")
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

fn handle_request(mut stream: TcpStream) {
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
    let mut body = vec![0_u8; content_len];
    reader.read_exact(&mut body).expect("read request body");
    let request: serde_json::Value = serde_json::from_slice(&body).expect("json request");
    let inputs = request["inputs"].as_array().expect("inputs array");
    let embeddings: Vec<_> = inputs.iter().map(embedding_for).collect();
    let response = serde_json::json!({
        "model": "jinaai/jina-embeddings-v5-omni-small",
        "dimensions": 4,
        "embeddings": embeddings
    });
    let body = response.to_string();
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(reply.as_bytes()).expect("write response");
}

fn embedding_for(input: &serde_json::Value) -> Vec<f32> {
    let haystack = input["value"].as_str().unwrap_or_default().to_lowercase();
    if haystack.contains("dog") || haystack.contains("puppy") {
        vec![1.0, 0.0, 0.0, 0.0]
    } else if haystack.contains("cat") || haystack.contains("kitten") {
        vec![0.0, 1.0, 0.0, 0.0]
    } else if haystack.contains("new") || haystack.contains("fresh") {
        vec![0.0, 0.0, 1.0, 0.0]
    } else {
        vec![0.0, 0.0, 0.0, 1.0]
    }
}
