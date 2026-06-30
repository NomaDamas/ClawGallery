use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{
        Arc, Mutex,
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
    _statuses: Arc<Mutex<Vec<u16>>>,
    _handle: JoinHandle<()>,
}

impl FakeEmbeddingServer {
    pub(crate) fn start() -> Self {
        Self::start_with_mode(false)
    }

    pub(crate) fn start_with_response_model(model: &'static str) -> Self {
        Self::start_with_mode_and_response_model(false, Some(model))
    }

    pub(crate) fn start_with_statuses(statuses: Vec<u16>) -> Self {
        Self::start_with_mode_response_model_and_statuses(false, None, statuses)
    }

    #[allow(dead_code)]
    pub(crate) fn start_multivector() -> Self {
        Self::start_with_mode_and_response_model(true, None)
    }

    fn start_with_mode(multivector: bool) -> Self {
        Self::start_with_mode_and_response_model(multivector, None)
    }

    fn start_with_mode_and_response_model(
        multivector: bool,
        response_model: Option<&'static str>,
    ) -> Self {
        Self::start_with_mode_response_model_and_statuses(multivector, response_model, Vec::new())
    }

    fn start_with_mode_response_model_and_statuses(
        multivector: bool,
        response_model: Option<&'static str>,
        statuses: Vec<u16>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake embedding server");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("fake server local addr")
        );
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::clone(&requests);
        let statuses = Arc::new(Mutex::new(statuses));
        let response_statuses = Arc::clone(&statuses);
        let handle = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                request_count.fetch_add(1, Ordering::SeqCst);
                let status = {
                    let mut statuses = response_statuses.lock().expect("status sequence lock");
                    if statuses.is_empty() {
                        200
                    } else {
                        statuses.remove(0)
                    }
                };
                handle_request(stream, multivector, response_model, status);
            }
        });
        Self {
            url,
            requests,
            _statuses: statuses,
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

fn handle_request(
    mut stream: TcpStream,
    multivector: bool,
    response_model: Option<&'static str>,
    status: u16,
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
    let mut body = vec![0_u8; content_len];
    reader.read_exact(&mut body).expect("read request body");
    let request: serde_json::Value = serde_json::from_slice(&body).expect("json request");
    if status != 200 {
        let body = format!("{{\"error\":\"status {status}\"}}");
        let reply = format!(
            "HTTP/1.1 {status} Test Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(reply.as_bytes()).expect("write response");
        return;
    }
    let inputs = request["inputs"].as_array().expect("inputs array");
    let model = request["model"].as_str().unwrap_or("test-model");
    let response_model = response_model.unwrap_or(model);
    let response = if multivector {
        let embeddings: Vec<_> = inputs.iter().map(multivector_for).collect();
        serde_json::json!({
            "model": response_model,
            "dimensions": 4,
            "embeddings": embeddings
        })
    } else {
        let embeddings: Vec<_> = inputs.iter().map(embedding_for).collect();
        serde_json::json!({
            "model": response_model,
            "dimensions": 4,
            "embeddings": embeddings
        })
    };
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

/// Emits one token-level vector per word so MaxSim has multiple rows to
/// reduce over, mimicking a late-interaction model.
fn multivector_for(input: &serde_json::Value) -> Vec<Vec<f32>> {
    let haystack = input["value"].as_str().unwrap_or_default().to_lowercase();
    let mut rows: Vec<Vec<f32>> = haystack
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            if word.contains("dog") || word.contains("puppy") {
                vec![1.0, 0.0, 0.0, 0.0]
            } else if word.contains("cat") || word.contains("kitten") {
                vec![0.0, 1.0, 0.0, 0.0]
            } else if word.contains("new") || word.contains("fresh") {
                vec![0.0, 0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 0.0, 1.0]
            }
        })
        .collect();
    if rows.is_empty() {
        rows.push(vec![0.0, 0.0, 0.0, 1.0]);
    }
    rows
}
