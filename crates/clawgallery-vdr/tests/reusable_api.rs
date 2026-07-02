use clawgallery_vdr::{
    CaptionDocument, DEFAULT_MAX_RETRIES, ImageDocument, SearchConfig, SyncConfig,
    embedding_search, status, sync,
};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    thread::{self, JoinHandle},
};

#[test]
fn reusable_vdr_api_indexes_and_searches_documents() {
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let image_path = temp.path().join("dog.png");
    let db_path = temp.path().join("vdr.sqlite3");
    fs::write(&image_path, b"dog image bytes").expect("write image");
    let images = vec![ImageDocument {
        id: "image-dog".to_string(),
        path: image_path.clone(),
        sha256: "sha-dog".to_string(),
    }];
    let captions = vec![CaptionDocument {
        image_id: "image-dog".to_string(),
        path: image_path.clone(),
        title: "Dog Park".to_string(),
        description: "puppy playing outside".to_string(),
    }];

    let outcome = sync(
        &SyncConfig {
            db_path: db_path.clone(),
            model: "test-model".to_string(),
            dimensions: 4,
            embedding_url: Some(server.url().to_string()),
            max_retries: DEFAULT_MAX_RETRIES,
            prune: true,
        },
        images.clone(),
        captions.clone(),
    )
    .expect("sync");
    let hits = embedding_search(
        &SearchConfig {
            db_path: db_path.clone(),
            model: Some("test-model".to_string()),
            dimensions: Some(4),
            embedding_url: Some(server.url().to_string()),
            limit: 1,
        },
        "dog",
        images,
        captions,
    )
    .expect("search");
    let vdr_status = status(&db_path, 1).expect("status");

    assert_eq!(outcome.indexed_vectors, 2);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].path, image_path);
    assert_eq!(hits[0].source, "embedding");
    assert!(
        hits[0].matched_field == "embedding_image" || hits[0].matched_field == "embedding_caption"
    );
    assert_eq!(hits[0].score, 1.0);
    assert_eq!(vdr_status.active_images, 1);
    assert_eq!(vdr_status.active_vectors, 2);
}

struct FakeEmbeddingServer {
    url: String,
    _handle: JoinHandle<()>,
}

impl FakeEmbeddingServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake embedding server");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("fake server local addr")
        );
        let handle = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                handle_request(stream);
            }
        });
        Self {
            url,
            _handle: handle,
        }
    }

    fn url(&self) -> &str {
        &self.url
    }
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
    let model = request["model"].as_str().expect("model");
    let response = serde_json::json!({
        "model": model,
        "dimensions": 4,
        "embeddings": inputs.iter().map(embedding_for).collect::<Vec<_>>()
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
    } else {
        vec![0.0, 0.0, 0.0, 1.0]
    }
}
