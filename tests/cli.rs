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

#[test]
fn folder_bootstrap_search_and_remove_flow() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("Screenshot 2026-05-03.png"), b"not really png").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    let listed = assert_success(run(&config, &["folder", "list"]));
    assert!(listed.contains(images.file_name().unwrap().to_str().unwrap()));

    let bootstrapped = assert_success(run(&config, &["bootstrap"]));
    assert!(bootstrapped.contains("ingested 1"));

    let search = assert_success(run(&config, &["search", "Screenshot"]));
    assert!(search.contains("Screenshot 2026-05-03.png"));

    assert_success(run(
        &config,
        &[
            "folder",
            "remove",
            images.canonicalize().unwrap().to_str().unwrap(),
        ],
    ));
    let listed_after = assert_success(run(&config, &["folder", "list"]));
    assert!(listed_after.trim().is_empty());
}

#[test]
fn caption_missing_auth_is_nonzero() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let image = temp.path().join("screen.png");
    fs::write(&image, b"not really png").unwrap();
    assert_success(run(&config, &["init"]));

    let output = run(&config, &["caption", "--file", image.to_str().unwrap()]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing visual model credentials"));
}

#[test]
fn rename_accepts_explicit_dry_run_flag() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let image = images.join("old.png");
    fs::write(&image, b"not really png").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    let image_path = image.canonicalize().unwrap();
    let image_line = fs::read_to_string(config.join("images.jsonl")).unwrap();
    let image_record: serde_json::Value =
        serde_json::from_str(image_line.lines().next().unwrap()).unwrap();
    let caption = serde_json::json!({
        "image_id": image_record["id"].as_str().unwrap(),
        "path": image_path,
        "title": "Important Settings Screen",
        "description": "A searchable settings screenshot",
        "model": "test",
        "provider": "test",
        "created_at": "2026-05-03T00:00:00Z",
        "filename_meaningful": false
    });
    fs::write(config.join("captions.jsonl"), format!("{}\n", caption)).unwrap();

    let dry_run = assert_success(run(&config, &["rename", "--dry-run"]));
    assert!(dry_run.contains("dry-run"));
    assert!(image.exists(), "dry-run must not rename original file");
}

#[test]
fn skill_path_materializes_embedded_skill() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let output = assert_success(run(&config, &["skill", "path"]));
    let path = PathBuf::from(output.trim());
    assert!(path.exists());
    let skill = fs::read_to_string(path).unwrap();
    assert!(skill.contains("name: clawgallery"));
}

#[test]
fn caption_dry_run_output_is_terse() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("IMG_0034.png"), b"x").unwrap();
    fs::write(images.join("Eva-William.png"), b"y").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));
    let stdout = assert_success(run(&config, &["caption", "--dry-run"]));

    assert!(
        stdout.contains("would caption "),
        "dry-run still announces planned captions, got: {stdout}"
    );
    for forbidden in [
        "title:",
        "filename_meaningful:",
        "(regex)",
        "(model)",
        " -> ",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "caption stdout must stay terse and not leak {forbidden:?}, got: {stdout}"
        );
    }
}

#[test]
fn caption_dry_run_does_not_require_credentials() {
    // Regression: dry-run never calls the network and must never require auth.
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let image = images.join("screen.png");
    fs::write(&image, b"not really png").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    let output = run(&config, &["caption", "--dry-run"]);
    if !output.status.success() {
        panic!(
            "caption --dry-run should succeed without credentials\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("would caption"),
        "dry-run output should list planned targets, got: {stdout}"
    );
    let captions = config.join("captions.jsonl");
    assert_eq!(
        fs::read_to_string(captions).unwrap(),
        "",
        "dry-run must not write captions.jsonl"
    );
}

#[test]
fn caption_dry_run_with_explicit_file_does_not_require_credentials() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let image = temp.path().join("screen.png");
    fs::write(&image, b"not really png").unwrap();
    assert_success(run(&config, &["init"]));

    let output = run(
        &config,
        &["caption", "--file", image.to_str().unwrap(), "--dry-run"],
    );
    if !output.status.success() {
        panic!(
            "caption --file --dry-run should succeed without credentials\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid jsonl line"))
        .collect()
}

#[test]
fn bootstrap_prune_marks_missing_files_inactive() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let kept = images.join("kept.png");
    let deleted = images.join("deleted.png");
    fs::write(&kept, b"kept").unwrap();
    fs::write(&deleted, b"deleted").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    let initial = assert_success(run(&config, &["bootstrap"]));
    assert!(initial.contains("ingested 2"));

    fs::remove_file(&deleted).unwrap();

    let pruned = assert_success(run(&config, &["bootstrap", "--prune"]));
    assert!(
        pruned.contains("pruned 1"),
        "bootstrap --prune should report 1 pruned, got: {pruned}"
    );

    let records = read_jsonl(&config.join("images.jsonl"));
    assert_eq!(
        records.len(),
        3,
        "1 kept + 1 active deleted + 1 inactive deleted"
    );

    let deleted_canonical = fs::canonicalize(deleted.parent().unwrap())
        .unwrap()
        .join("deleted.png");
    let deleted_records: Vec<_> = records
        .iter()
        .filter(|r| r["path"].as_str().unwrap() == deleted_canonical.to_str().unwrap())
        .collect();
    assert_eq!(
        deleted_records.len(),
        2,
        "deleted.png appears twice (active=true then active=false)"
    );

    let last_deleted = deleted_records.last().unwrap();
    assert_eq!(
        last_deleted["active"],
        serde_json::json!(false),
        "latest record for missing file must be active=false"
    );
    assert!(
        last_deleted["removed_at"].is_string(),
        "active=false record must include removed_at timestamp"
    );

    let kept_canonical = fs::canonicalize(&kept).unwrap();
    let last_kept = records
        .iter()
        .rfind(|r| r["path"].as_str().unwrap() == kept_canonical.to_str().unwrap())
        .unwrap();
    assert_eq!(last_kept["active"], serde_json::json!(true));
}

#[test]
fn bootstrap_without_prune_does_not_touch_missing_records() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("vanishes.png");
    fs::write(&img, b"x").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    fs::remove_file(&img).unwrap();

    let plain = assert_success(run(&config, &["bootstrap"]));
    assert!(plain.contains("ingested 0"));
    assert!(!plain.contains("pruned"));
    let records = read_jsonl(&config.join("images.jsonl"));
    assert_eq!(
        records.len(),
        1,
        "no prune happened, only original ingest record exists"
    );
}

#[test]
fn search_and_status_ignore_pruned_records() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let kept = images.join("findable.png");
    let pruned = images.join("ghost.png");
    fs::write(&kept, b"k").unwrap();
    fs::write(&pruned, b"g").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    fs::remove_file(&pruned).unwrap();
    assert_success(run(&config, &["bootstrap", "--prune"]));

    let status = assert_success(run(&config, &["status"]));
    assert!(
        status.contains("images: 1"),
        "status should report only active images, got: {status}"
    );

    let search_kept = assert_success(run(&config, &["search", "findable"]));
    assert!(search_kept.contains("findable.png"));

    let search_ghost = assert_success(run(&config, &["search", "ghost"]));
    assert!(
        !search_ghost.contains("ghost.png"),
        "pruned (active=false) records must not appear in search, got: {search_ghost}"
    );
}

fn inject_caption(
    config: &Path,
    image_path: &Path,
    image_id: &str,
    title: &str,
    filename_meaningful: Option<bool>,
) {
    let mut record = serde_json::json!({
        "image_id": image_id,
        "path": image_path,
        "title": title,
        "description": format!("test description for {title}"),
        "model": "test",
        "provider": "test",
        "created_at": "2026-05-04T00:00:00Z",
    });
    if let Some(b) = filename_meaningful {
        record["filename_meaningful"] = serde_json::Value::Bool(b);
    }
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

#[test]
fn rename_skips_meaningful_filenames_by_default() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("기아 승리 열차.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Some Caption Title", None);

    let dry = assert_success(run(&config, &["rename", "--dry-run"]));
    assert!(
        dry.contains("would skip"),
        "regex-meaningful Hangul name must trigger skip, got: {dry}"
    );
    assert!(
        !dry.contains("dry-run "),
        "no actual dry-run rename line should be emitted, got: {dry}"
    );
}

#[test]
fn rename_renames_generic_filenames_by_default() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("IMG_0034.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Cute Cat Doodle", None);

    let dry = assert_success(run(&config, &["rename", "--dry-run", "--style", "title"]));
    assert!(
        dry.contains("dry-run"),
        "generic stem should produce a dry-run rename, got: {dry}"
    );
    assert!(dry.contains("cute-cat-doodle"));
}

#[test]
fn rename_force_overrides_skip() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("기아 승리 열차.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Forced Title", None);

    let dry = assert_success(run(
        &config,
        &["rename", "--dry-run", "--force", "--style", "title"],
    ));
    assert!(
        dry.contains("dry-run"),
        "--force must rename even meaningful names, got: {dry}"
    );
    assert!(dry.contains("forced-title"));
}

#[test]
fn rename_explicit_file_overrides_skip() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("기아 승리 열차.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Explicit File Title", None);

    let dry = assert_success(run(
        &config,
        &[
            "rename",
            "--dry-run",
            "--file",
            canonical.to_str().unwrap(),
            "--style",
            "title",
        ],
    ));
    assert!(
        dry.contains("dry-run"),
        "--file must rename even meaningful names, got: {dry}"
    );
}

#[test]
fn rename_uses_cached_filename_meaningful_from_caption() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("test-image.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "From Cache False", Some(false));

    let dry = assert_success(run(&config, &["rename", "--dry-run", "--style", "title"]));
    assert!(
        dry.contains("dry-run"),
        "cached filename_meaningful=false must trigger rename, got: {dry}"
    );

    fs::write(config.join("captions.jsonl"), "").unwrap();
    inject_caption(&config, &canonical, &id, "From Cache True", Some(true));
    let dry2 = assert_success(run(&config, &["rename", "--dry-run", "--style", "title"]));
    assert!(
        dry2.contains("would skip"),
        "cached filename_meaningful=true must trigger skip, got: {dry2}"
    );
}

#[test]
fn caption_dry_run_skips_pruned_records() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let alive = images.join("alive.png");
    let dead = images.join("dead.png");
    fs::write(&alive, b"a").unwrap();
    fs::write(&dead, b"d").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    fs::remove_file(&dead).unwrap();
    assert_success(run(&config, &["bootstrap", "--prune"]));

    let output = assert_success(run(&config, &["caption", "--dry-run"]));
    assert!(output.contains("alive.png"));
    assert!(
        !output.contains("dead.png"),
        "pruned image must not be a caption target, got: {output}"
    );
}

#[test]
fn rename_apply_self_heals_when_source_already_missing() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("IMG_0042.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Synthetic Title", Some(false));
    fs::remove_file(&canonical).unwrap();

    let output = run(&config, &["rename", "--apply", "--style", "title"]);
    assert!(
        output.status.success(),
        "rename --apply must self-heal when source is missing\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains("missing source"),
        "stdout should announce a missing-source skip, got: {stdout}"
    );

    let records = read_jsonl(&config.join("images.jsonl"));
    let our_records: Vec<_> = records
        .iter()
        .filter(|r| r["id"].as_str().unwrap() == id)
        .collect();
    assert!(
        our_records
            .iter()
            .any(|r| r["active"] == serde_json::json!(false) && r["removed_at"].is_string()),
        "self-heal must append an active=false record for the vanished path, got: {our_records:#?}"
    );
}

#[test]
fn rename_apply_continues_after_per_image_failure() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let vanished = images.join("IMG_0001.png");
    let alive = images.join("IMG_0002.png");
    fs::write(&vanished, b"a").unwrap();
    fs::write(&alive, b"b").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));

    let records = read_jsonl(&config.join("images.jsonl"));
    let id_for = |name: &str| -> String {
        records
            .iter()
            .find(|r| r["path"].as_str().unwrap().ends_with(name))
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    inject_caption(
        &config,
        &vanished.canonicalize().unwrap(),
        &id_for("IMG_0001.png"),
        "ghost",
        Some(false),
    );
    inject_caption(
        &config,
        &alive.canonicalize().unwrap(),
        &id_for("IMG_0002.png"),
        "fresh-target",
        Some(false),
    );
    fs::remove_file(&vanished).unwrap();

    let output = run(&config, &["rename", "--apply", "--style", "title"]);
    assert!(
        output.status.success(),
        "batch must keep going past the missing-source image\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains("missing source"),
        "missing source must be reported per-image, got: {stdout}"
    );
    assert!(
        stdout.contains("renamed ") && stdout.contains("fresh-target.png"),
        "second image must still be renamed, got: {stdout}"
    );
}

#[test]
fn rename_dry_run_self_heals_when_source_missing() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let img = images.join("IMG_9999.png");
    fs::write(&img, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = img.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Title", Some(false));
    fs::remove_file(&canonical).unwrap();

    let stdout = assert_success(run(&config, &["rename", "--dry-run", "--style", "title"]));
    assert!(
        stdout.contains("missing source"),
        "dry-run must also report missing source skip, got: {stdout}"
    );
    let records = read_jsonl(&config.join("images.jsonl"));
    assert!(
        !records
            .iter()
            .any(|r| r.get("active") == Some(&serde_json::json!(false))),
        "dry-run must NEVER append active=false records, got: {records:#?}"
    );
}
