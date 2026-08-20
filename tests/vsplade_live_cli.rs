use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

fn clawgallery_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

fn python() -> PathBuf {
    env::var_os("CLAWGALLERY_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/jeffrey/Projects/SPLADE-mlx/.venv/bin/python"))
}

fn run(config: &Path, args: &[&str], extra_env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(clawgallery_bin());
    command
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env_remove("CLAWGALLERY_VDR_EMBEDDING_URL")
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.output().expect("clawgallery command should run")
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

fn write_text_png(python: &Path, path: &Path, text: &str) {
    let script = format!(
        r#"
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (768, 512), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
draw.text((40, 200), {text:?}, fill="black", font=font)
img.save({path:?})
"#
    );
    let status = Command::new(python)
        .arg("-c")
        .arg(script)
        .status()
        .expect("python png renderer");
    assert!(status.success(), "failed to render {}", path.display());
}

fn wait_for_embed(url: &str, model: &str, dimensions: usize, timeout: Duration) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("http client");
    let endpoint = format!("{url}/embed");
    let deadline = Instant::now() + timeout;
    loop {
        if Instant::now() >= deadline {
            panic!("V-SPLADE server at {url} did not become reachable");
        }
        let response = client
            .post(&endpoint)
            .json(&serde_json::json!({
                "model": model,
                "dimensions": dimensions,
                "format": "sparse",
                "inputs": []
            }))
            .send();
        if matches!(response, Ok(resp) if resp.status().is_success()) {
            return;
        }
        thread::sleep(Duration::from_millis(250));
    }
}

#[test]
#[ignore = "requires local SPLADE-mlx venv and V-SPLADE weights"]
fn live_vsplade_lexical_and_hybrid_search() {
    let python = python();
    assert!(
        python.exists(),
        "CLAWGALLERY_PYTHON or SPLADE-mlx venv python is required"
    );
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    let invoice = images.join("invoice.png");
    let beach = images.join("beach.png");
    write_text_png(&python, &invoice, "INVOICE TOTAL REVENUE 2023");
    write_text_png(&python, &beach, "SUNSET BEACH PHOTO");

    let empty = run(&config, &["init"], &[]);
    assert_success(empty);
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        &[],
    ));

    let mut captions = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.join("captions.jsonl"))
        .expect("captions");
    let images_jsonl = fs::read_to_string(config.join("images.jsonl")).expect("images");
    for line in images_jsonl.lines().filter(|line| !line.trim().is_empty()) {
        let record: serde_json::Value = serde_json::from_str(line).expect("image");
        let path = record["path"].as_str().expect("path");
        let title = if path.ends_with("invoice.png") {
            "Receipt Scan"
        } else {
            "Holiday Snapshot"
        };
        writeln!(
            captions,
            "{}",
            serde_json::json!({
                "image_id": record["id"],
                "path": path,
                "title": title,
                "description": title,
                "model": "test",
                "provider": "test",
                "created_at": "2026-08-19T00:00:00Z",
                "filename_meaningful": false
            })
        )
        .expect("caption");
    }

    let model = "NomaDamas/v-splade-efficient-mlx";
    let dimensions = 50368_usize;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    let url = format!("http://127.0.0.1:{port}");
    let script = include_str!("../scripts/vsplade_server.py");
    let mut child = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--model")
        .arg(model)
        .arg("--dimensions")
        .arg(dimensions.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start vsplade server");
    wait_for_embed(&url, model, dimensions, Duration::from_secs(20 * 60));

    let env = [("CLAWGALLERY_PYTHON", python.to_str().expect("utf8"))];
    let synced = assert_success(run(
        &config,
        &[
            "vdr",
            "sync",
            "--backend",
            "vsplade",
            "--model",
            model,
            "--dimensions",
            &dimensions.to_string(),
            "--embedding-url",
            &url,
            "--no-auto-start",
        ],
        &env,
    ));
    assert!(
        synced.contains("indexed 2"),
        "live sync should index both images, got: {synced}"
    );

    let lexical = assert_success(run(
        &config,
        &[
            "search",
            "--mode",
            "lexical",
            "--embedding-url",
            &url,
            "invoice total revenue",
            "--json",
            "--limit",
            "2",
        ],
        &env,
    ));
    let lexical_rows: Vec<serde_json::Value> = lexical
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    assert!(
        lexical_rows[0]["path"]
            .as_str()
            .expect("path")
            .ends_with("invoice.png"),
        "live lexical search should rank the invoice screenshot first, got: {lexical}"
    );
    assert_eq!(lexical_rows[0]["source"], "lexical");
    assert!(
        lexical_rows[0]["score"].as_f64().expect("score")
            > lexical_rows
                .get(1)
                .and_then(|row| row["score"].as_f64())
                .unwrap_or(0.0),
        "invoice should outscore beach, got: {lexical}"
    );

    let hybrid = assert_success(run(
        &config,
        &[
            "search",
            "--mode",
            "hybrid",
            "--embedding-url",
            &url,
            "invoice total revenue",
            "--json",
            "--limit",
            "5",
        ],
        &env,
    ));
    let hybrid_rows: Vec<serde_json::Value> = hybrid
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    assert!(
        hybrid_rows.iter().any(|row| {
            row["path"].as_str().expect("path").ends_with("invoice.png")
                && (row["source"] == "lexical"
                    || row["source"] == "hybrid"
                    || row["source"] == "fuzzy")
        }),
        "hybrid RRF should keep the invoice hit, got: {hybrid}"
    );

    let _ = child.kill();
    let _ = child.wait();
}
