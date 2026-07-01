use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

fn run(config: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .env("CLAWGALLERY_CONFIG_DIR", config)
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

fn inject_caption(config: &Path, image_path: &Path, image_id: &str, title: &str) {
    let record = serde_json::json!({
        "image_id": image_id,
        "path": image_path,
        "title": title,
        "description": format!("test description for {title}"),
        "model": "test",
        "provider": "test",
        "created_at": "2026-05-04T00:00:00Z",
        "filename_meaningful": false,
    });
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.join("captions.jsonl"))
        .unwrap();
    use std::io::Write;
    writeln!(file, "{record}").unwrap();
}

fn first_image_id(config: &Path) -> String {
    let line = fs::read_to_string(config.join("images.jsonl")).unwrap();
    let rec: serde_json::Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
    rec["id"].as_str().unwrap().to_string()
}

fn setup_search_fixture(
    entries: &[(&str, &str, &str, &str, bool)],
) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    assert_success(run(&config, &["init"]));

    let mut image_lines = String::new();
    let mut caption_lines = String::new();
    for (idx, (name, title, description, discovered_at, active)) in entries.iter().enumerate() {
        let path = images.join(name);
        fs::write(&path, b"not really png").unwrap();
        let canonical = path.canonicalize().unwrap();
        let id = format!("image-{idx}");
        let image = serde_json::json!({
            "id": id,
            "path": canonical,
            "original_path": canonical,
            "sha256": format!("sha{idx}"),
            "size": 1,
            "modified_at": null,
            "discovered_at": discovered_at,
            "extension": "png",
            "active": active,
            "removed_at": null,
        });
        image_lines.push_str(&format!("{image}\n"));
        let caption = serde_json::json!({
            "image_id": id,
            "path": canonical,
            "title": title,
            "description": description,
            "model": "test",
            "provider": "test",
            "created_at": "2026-05-04T00:00:00Z",
            "filename_meaningful": false
        });
        caption_lines.push_str(&format!("{caption}\n"));
    }
    fs::write(config.join("images.jsonl"), image_lines).unwrap();
    fs::write(config.join("captions.jsonl"), caption_lines).unwrap();
    (temp, config)
}

#[test]
fn folder_add_duplicate_is_not_noisy() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();

    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    let duplicate = assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));

    assert_eq!(
        duplicate.trim(),
        format!(
            "folder already tracked: {}",
            images.canonicalize().unwrap().display()
        )
    );
}

#[test]
fn bootstrap_unknown_folder_id_is_an_error() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();

    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    let output = run(&config, &["bootstrap", "--folder", "missing-folder-id"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no active folder matched 'missing-folder-id'"),
        "unknown folder id must fail clearly, got: {stderr}"
    );
}

#[test]
fn rename_dry_run_does_not_write_rename_records() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let image = images.join("IMG_0099.png");
    fs::write(&image, b"not really png").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));
    let canonical = image.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Dry Run Title");

    let dry_run = assert_success(run(&config, &["rename", "--dry-run", "--style", "title"]));

    assert!(dry_run.contains("dry-run"));
    assert_eq!(
        fs::read_to_string(config.join("renames.jsonl")).unwrap(),
        "",
        "rename dry-run must not mutate renames.jsonl"
    );
}

#[test]
fn rename_undo_dry_run_does_not_write_rename_records() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let original = images.join("IMG_0111.png");
    fs::write(&original, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = original.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Undo Dry Run");
    assert_success(run(&config, &["rename", "--apply", "--style", "title"]));
    let before = fs::read_to_string(config.join("renames.jsonl")).unwrap();

    let dry_run = assert_success(run(&config, &["rename", "--undo", "--last", "--dry-run"]));

    assert!(dry_run.contains("would undo"));
    assert_eq!(
        fs::read_to_string(config.join("renames.jsonl")).unwrap(),
        before,
        "rename undo dry-run must not mutate renames.jsonl"
    );
}

#[test]
fn caption_rejects_zero_concurrency() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));

    let output = run(&config, &["caption", "--dry-run", "--concurrency", "0"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--concurrency must be at least 1"),
        "zero concurrency should fail clearly, got: {stderr}"
    );
}

#[test]
fn poll_rejects_zero_interval() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));

    let output = run(&config, &["poll", "--once", "--interval", "0"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--interval must be at least 1"),
        "zero interval should fail clearly, got: {stderr}"
    );
}

#[test]
fn search_rejects_zero_limit() {
    let (_temp, config) =
        setup_search_fixture(&[("one.png", "Settings", "panel", "2026-05-04T00:00:00Z", true)]);

    let output = run(&config, &["search", "settings", "--limit", "0"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--limit must be at least 1"),
        "zero limit should fail clearly, got: {stderr}"
    );
}

#[test]
fn status_fails_on_corrupt_config() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));
    fs::write(config.join("config.json"), "{not json").unwrap();

    let output = run(&config, &["status"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("config.json"),
        "corrupt config should be surfaced, got: {stderr}"
    );
}

#[test]
fn malformed_jsonl_state_fails_loudly() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));
    fs::write(config.join("images.jsonl"), "{bad json\n").unwrap();

    let output = run(&config, &["status"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("images.jsonl") && stderr.contains("line 1"),
        "malformed state line should identify the file and line, got: {stderr}"
    );
}

#[test]
fn dedup_rejects_exact_and_similar_together() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));

    let output = run(&config, &["dedup", "--exact", "--similar"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--exact and --similar cannot be used together"),
        "conflicting dedup modes should fail clearly, got: {stderr}"
    );
}
