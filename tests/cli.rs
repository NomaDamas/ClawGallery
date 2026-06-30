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

fn run_with_env(config: &Path, args: &[&str], envs: &[(&str, &Path)]) -> Output {
    let mut command = Command::new(bin());
    command
        .env("CLAWGALLERY_CONFIG_DIR", config)
        .env_remove("OPENAI_API_KEY")
        .env("CODEX_HOME", config.join("codex-home"))
        .args(args);
    for (key, value) in envs {
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
fn daemon_install_status_and_uninstall_use_managed_service_file() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let daemon_dir = temp.path().join("daemon-services");
    assert_success(run(&config, &["init"]));

    let installed = assert_success(run_with_env(
        &config,
        &[
            "daemon",
            "install",
            "--interval",
            "5",
            "--caption",
            "--sync",
        ],
        &[("CLAWGALLERY_DAEMON_DIR", daemon_dir.as_path())],
    ));

    assert!(
        installed.contains("installed daemon service"),
        "got: {installed}"
    );
    let service_file = daemon_dir.join("com.clawgallery.poll.plist");
    assert!(
        service_file.exists(),
        "daemon install should write service file"
    );
    let service = fs::read_to_string(&service_file).unwrap();
    assert!(service.contains("daemon"));
    assert!(service.contains("run"));
    assert!(service.contains("--caption"));
    assert!(service.contains("--sync"));

    let status = assert_success(run_with_env(
        &config,
        &["daemon", "status"],
        &[("CLAWGALLERY_DAEMON_DIR", daemon_dir.as_path())],
    ));
    assert!(status.contains("installed: yes"), "got: {status}");
    assert!(status.contains("last_started: <never>"), "got: {status}");

    let uninstalled = assert_success(run_with_env(
        &config,
        &["daemon", "uninstall"],
        &[("CLAWGALLERY_DAEMON_DIR", daemon_dir.as_path())],
    ));
    assert!(uninstalled.contains("uninstalled daemon service"));
    assert!(
        !service_file.exists(),
        "daemon uninstall should remove service file"
    );
}

#[test]
fn daemon_status_reports_missing_service_cleanly() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let daemon_dir = temp.path().join("daemon-services");
    assert_success(run(&config, &["init"]));

    let status = assert_success(run_with_env(
        &config,
        &["daemon", "status"],
        &[("CLAWGALLERY_DAEMON_DIR", daemon_dir.as_path())],
    ));

    assert!(status.contains("installed: no"), "got: {status}");
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
fn caption_dry_run_accepts_concurrency_flag() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("first.png"), b"first").unwrap();
    fs::write(images.join("second.png"), b"second").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    let serial = assert_success(run(
        &config,
        &["caption", "--dry-run", "--concurrency", "1"],
    ));
    let parallel = assert_success(run(
        &config,
        &["caption", "--dry-run", "--concurrency", "2"],
    ));

    assert_eq!(serial, parallel);
    assert!(serial.contains("would caption "));
    assert!(serial.contains("first.png"));
    assert!(serial.contains("second.png"));
}

#[test]
fn bootstrap_ingests_heic_and_heif_files() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("iphone.heic"), b"fake heic").unwrap();
    fs::write(images.join("camera.heif"), b"fake heif").unwrap();

    assert_success(run(&config, &["init"]));
    let bootstrapped = assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    assert!(
        bootstrapped.contains("ingested 2"),
        "HEIC/HEIF files should be ingested, got: {bootstrapped}"
    );
    let search = assert_success(run(&config, &["search", "iphone"]));
    assert!(search.contains("iphone.heic"));
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

fn result_paths(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|line| line.ends_with(".png"))
        .collect()
}

#[test]
fn search_ranks_title_above_description() {
    let (_temp, config) = setup_search_fixture(&[
        (
            "desc.png",
            "Settings Panel",
            "Login workflow appears in a browser",
            "2026-05-04T00:00:00Z",
            true,
        ),
        (
            "title.png",
            "Login Dialog",
            "A generic modal",
            "2026-05-03T00:00:00Z",
            true,
        ),
    ]);
    let stdout = assert_success(run(&config, &["search", "login"]));
    let paths = result_paths(&stdout);
    assert!(paths[0].ends_with("title.png"), "got: {stdout}");
}

#[test]
fn search_smart_case_lowercase_query() {
    let (_temp, config) = setup_search_fixture(&[(
        "mixed.png",
        "Login Dialog",
        "Mixed case title",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "login"]));
    assert!(stdout.contains("mixed.png"));
}

#[test]
fn search_smart_case_uppercase_query_is_case_sensitive() {
    let (_temp, config) = setup_search_fixture(&[(
        "lower.png",
        "login dialog",
        "lowercase title",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "Login"]));
    assert!(stdout.trim().is_empty(), "got: {stdout}");
}

#[test]
fn search_fzf_dsl_negation() {
    let (_temp, config) = setup_search_fixture(&[
        (
            "good.png",
            "Login dialog",
            "production UI",
            "2026-05-04T00:00:00Z",
            true,
        ),
        (
            "bad.png",
            "Login dialog",
            "test fixture UI",
            "2026-05-05T00:00:00Z",
            true,
        ),
    ]);
    let stdout = assert_success(run(&config, &["search", "login", "!test"]));
    assert!(stdout.contains("good.png"));
    assert!(!stdout.contains("bad.png"));
}

#[test]
fn search_fzf_dsl_exact() {
    let (_temp, config) = setup_search_fixture(&[
        ("foo.png", "foo", "exact", "2026-05-04T00:00:00Z", true),
        ("fo.png", "fo", "short", "2026-05-05T00:00:00Z", true),
    ]);
    let stdout = assert_success(run(&config, &["search", "'foo"]));
    assert!(stdout.contains("foo.png"));
    assert!(!stdout.contains("fo.png"));
}

#[test]
fn search_fzf_dsl_prefix() {
    let (_temp, config) = setup_search_fixture(&[
        (
            "login.png",
            "login panel",
            "prefix",
            "2026-05-04T00:00:00Z",
            true,
        ),
        (
            "other.png",
            "panel login",
            "not prefix",
            "2026-05-05T00:00:00Z",
            true,
        ),
    ]);
    let stdout = assert_success(run(&config, &["search", "^login"]));
    assert!(stdout.contains("login.png"));
    assert!(!stdout.contains("other.png"));
}

#[test]
fn search_levenshtein_fallback_typo() {
    let (_temp, config) = setup_search_fixture(&[(
        "screen.png",
        "Screenshot capture",
        "desktop image",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "scrnshot"]));
    assert!(stdout.contains("falling back"), "got: {stdout}");
    assert!(stdout.contains("screen.png"));
}

#[test]
fn search_levenshtein_skipped_for_short_atom() {
    let (_temp, config) = setup_search_fixture(&[(
        "ux.png",
        "UX panel",
        "interface settings",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "ui"]));
    assert!(stdout.trim().is_empty(), "got: {stdout}");
}

#[test]
fn search_no_fuzzy_disables_dsl_and_fallback() {
    let (_temp, config) = setup_search_fixture(&[(
        "bang.png",
        "literal !foo marker",
        "not a negation",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "!foo", "--no-fuzzy"]));
    assert!(stdout.contains("bang.png"));
    assert!(!stdout.contains("score:"), "old output only: {stdout}");
}

#[test]
fn search_json_output_jsonl_one_per_line() {
    let (_temp, config) = setup_search_fixture(&[(
        "json.png",
        "Login JSON",
        "structured output",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "login", "--json"]));
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(value["source"], "fuzzy");
        assert!(value["score"].is_number());
    }
}

#[test]
fn search_case_sensitive_flag() {
    let (_temp, config) = setup_search_fixture(&[(
        "case.png",
        "foo panel",
        "lowercase only",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "Foo", "--case-sensitive"]));
    assert!(stdout.trim().is_empty(), "got: {stdout}");
}

#[test]
fn search_korean_nfc_normalization() {
    let nfd = "한글";
    let (_temp, config) = setup_search_fixture(&[(
        "korean.png",
        nfd,
        "decomposed Hangul caption",
        "2026-05-04T00:00:00Z",
        true,
    )]);
    let stdout = assert_success(run(&config, &["search", "한글"]));
    assert!(stdout.contains("korean.png"), "got: {stdout}");
}

#[test]
fn search_active_false_excluded() {
    let (_temp, config) = setup_search_fixture(&[
        (
            "active.png",
            "Login active",
            "kept",
            "2026-05-04T00:00:00Z",
            true,
        ),
        (
            "inactive.png",
            "Login inactive",
            "pruned",
            "2026-05-05T00:00:00Z",
            false,
        ),
    ]);
    let stdout = assert_success(run(&config, &["search", "login"]));
    assert!(stdout.contains("active.png"));
    assert!(!stdout.contains("inactive.png"));
}

#[test]
fn search_no_results_clean_exit() {
    let (_temp, config) =
        setup_search_fixture(&[("one.png", "Settings", "panel", "2026-05-04T00:00:00Z", true)]);
    let stdout = assert_success(run(&config, &["search", "nomatch"]));
    assert!(stdout.trim().is_empty(), "got: {stdout}");
}

#[test]
fn search_limit_truncates_after_sort() {
    let (_temp, config) = setup_search_fixture(&[
        (
            "path-login.png",
            "Other",
            "misc",
            "2026-05-05T00:00:00Z",
            true,
        ),
        (
            "desc.png",
            "Other",
            "login description",
            "2026-05-04T00:00:00Z",
            true,
        ),
        (
            "title-a.png",
            "Login Alpha",
            "best",
            "2026-05-03T00:00:00Z",
            true,
        ),
        (
            "title-b.png",
            "Login Beta",
            "best",
            "2026-05-06T00:00:00Z",
            true,
        ),
        ("other.png", "Login", "best", "2026-05-02T00:00:00Z", true),
    ]);
    let stdout = assert_success(run(&config, &["search", "login", "--limit", "2"]));
    let paths = result_paths(&stdout);
    assert_eq!(paths.len(), 2, "got: {stdout}");
    assert!(
        paths
            .iter()
            .all(|path| path.contains("title") || path.contains("other")),
        "got: {stdout}"
    );
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
fn poll_once_caption_sync_logs_stage_failures_without_stopping_poll() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("new.png"), b"not really png").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));

    let stdout = assert_success(run(
        &config,
        &[
            "poll",
            "--once",
            "--caption",
            "--sync",
            "--embedding-url",
            "http://127.0.0.1:9",
        ],
    ));
    assert!(stdout.contains("ingested 1 new image(s)"), "got: {stdout}");
    assert!(
        stdout.contains("caption stage failed"),
        "poll should report caption stage failure without aborting, got: {stdout}"
    );
    assert!(
        stdout.contains("vdr sync stage failed"),
        "poll should report vdr sync failure without aborting, got: {stdout}"
    );

    let errors = read_jsonl(&config.join("errors.jsonl"));
    assert!(
        errors
            .iter()
            .any(|record| record["context"] == "poll_caption"),
        "caption failure should be logged to errors.jsonl: {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|record| record["context"] == "poll_vdr_sync"),
        "vdr sync failure should be logged to errors.jsonl: {errors:#?}"
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

#[test]
fn forget_file_deactivates_without_deleting_and_removes_from_search_status() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let image = images.join("forget-me.png");
    fs::write(&image, b"still on disk").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));
    let canonical = image.canonicalize().unwrap();

    let before = assert_success(run(&config, &["search", "forget-me"]));
    assert!(before.contains("forget-me.png"));

    let forgot = assert_success(run(
        &config,
        &["forget", "--file", canonical.to_str().unwrap()],
    ));
    assert!(
        forgot.contains("forgot 1 image"),
        "forget should summarize the deactivation, got: {forgot}"
    );
    assert!(
        image.exists(),
        "forget without --delete must leave the file"
    );

    let status = assert_success(run(&config, &["status"]));
    assert!(
        status.contains("images: 0"),
        "status should exclude forgotten images, got: {status}"
    );
    let after = assert_success(run(&config, &["search", "forget-me"]));
    assert!(
        !after.contains("forget-me.png"),
        "forgotten image must not appear in search, got: {after}"
    );

    let records = read_jsonl(&config.join("images.jsonl"));
    let path_records: Vec<_> = records
        .iter()
        .filter(|record| record["path"].as_str() == Some(canonical.to_str().unwrap()))
        .collect();
    assert_eq!(
        path_records.len(),
        2,
        "forget must append a new image record"
    );
    let latest = path_records.last().unwrap();
    assert_eq!(latest["active"], false);
    assert!(latest["removed_at"].is_string());
}

#[test]
fn forget_delete_removes_disk_file_and_untracks_image() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let image = images.join("delete-me.png");
    fs::write(&image, b"remove from disk").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));
    let canonical = image.canonicalize().unwrap();

    let forgot = assert_success(run(
        &config,
        &["forget", "--file", canonical.to_str().unwrap(), "--delete"],
    ));
    assert!(forgot.contains("deleted"));
    assert!(!image.exists(), "forget --delete must remove the disk file");

    let status = assert_success(run(&config, &["status"]));
    assert!(status.contains("images: 0"));
}

#[test]
fn forget_missing_path_reports_clear_error() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));

    let missing = temp.path().join("missing.png");
    let output = run(&config, &["forget", "--file", missing.to_str().unwrap()]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no active image matched"),
        "missing path should report a clear error, got: {stderr}"
    );
    assert!(stderr.contains("missing.png"));
}

#[test]
fn dedup_exact_groups_active_images_by_sha_jsonl() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("copy-a.png"), b"same bytes").unwrap();
    fs::write(images.join("copy-b.png"), b"same bytes").unwrap();
    fs::write(images.join("unique.png"), b"different").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    let stdout = assert_success(run(&config, &["dedup", "--exact", "--json"]));
    let rows: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    assert_eq!(rows.len(), 1, "expected one duplicate group, got: {stdout}");
    assert_eq!(rows[0]["kind"], "exact");
    assert!(
        rows[0]["representative"]["path"]
            .as_str()
            .unwrap()
            .ends_with("copy-a.png")
    );
    let duplicates = rows[0]["duplicates"].as_array().unwrap();
    assert_eq!(duplicates.len(), 1);
    assert!(
        duplicates[0]["path"]
            .as_str()
            .unwrap()
            .ends_with("copy-b.png")
    );
    assert!(
        !stdout.contains("unique.png"),
        "unique image must not be reported, got: {stdout}"
    );
}

#[test]
fn dedup_exact_reports_no_groups_cleanly() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    fs::write(images.join("one.png"), b"one").unwrap();
    fs::write(images.join("two.png"), b"two").unwrap();

    assert_success(run(&config, &["init"]));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().unwrap()],
    ));

    let stdout = assert_success(run(&config, &["dedup", "--exact"]));
    assert_eq!(stdout.trim(), "no duplicate groups found");
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
fn rename_undo_last_restores_original_path_and_state() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let original = images.join("IMG_0042.png");
    fs::write(&original, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = original.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Restored Name", Some(false));
    assert_success(run(&config, &["rename", "--apply", "--style", "title"]));
    let renamed = images.join("restored-name.png");
    assert!(renamed.exists(), "rename --apply should create target");
    assert!(!original.exists(), "rename --apply should move original");

    let undo = assert_success(run(&config, &["rename", "--undo", "--last"]));

    assert!(undo.contains("undone 1"), "got: {undo}");
    assert!(original.exists(), "undo should restore original file");
    assert!(!renamed.exists(), "undo should remove renamed file");
    let status = assert_success(run(&config, &["status"]));
    assert!(status.contains("images: 1"), "got: {status}");
    let search_original = assert_success(run(&config, &["search", "IMG_0042"]));
    assert!(search_original.contains("IMG_0042.png"));
    let search_renamed = assert_success(run(&config, &["search", "restored-name"]));
    assert!(
        !search_renamed.contains("restored-name.png"),
        "undone path must not stay searchable, got: {search_renamed}"
    );
}

#[test]
fn rename_undo_last_skips_when_original_path_is_occupied() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let original = images.join("IMG_0043.png");
    fs::write(&original, b"x").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = original.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Collision Name", Some(false));
    assert_success(run(&config, &["rename", "--apply", "--style", "title"]));
    let renamed = images.join("collision-name.png");
    fs::write(&original, b"blocker").unwrap();

    let undo = assert_success(run(&config, &["rename", "--undo", "--last"]));

    assert!(undo.contains("skipped 1"), "got: {undo}");
    assert!(renamed.exists(), "collision skip must leave renamed file");
    assert_eq!(fs::read(&original).unwrap(), b"blocker");
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
fn rename_apply_existing_target_does_not_clobber() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).unwrap();
    let source = images.join("IMG_0100.png");
    let target = images.join("existing-title.png");
    fs::write(&source, b"source bytes").unwrap();
    fs::write(&target, b"target bytes").unwrap();
    assert_success(run(&config, &["init"]));
    assert_success(run(&config, &["folder", "add", images.to_str().unwrap()]));
    assert_success(run(&config, &["bootstrap"]));
    let canonical = source.canonicalize().unwrap();
    let id = first_image_id(&config);
    inject_caption(&config, &canonical, &id, "Existing Title", Some(false));

    let output = run(&config, &["rename", "--apply", "--style", "title"]);
    assert!(
        output.status.success(),
        "rename collision is a per-image failure, not a process failure\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains("existing-title-1.png"),
        "occupied target must be avoided via collision suffix, got: {stdout}"
    );
    assert!(
        !target.exists() || fs::read(&target).unwrap() == b"target bytes",
        "pre-existing target must never be overwritten"
    );
    assert_eq!(
        fs::read(images.join("existing-title-1.png")).unwrap(),
        b"source bytes"
    );
    assert!(!source.exists(), "source must be moved");
}

#[test]
fn final_error_masks_api_keys_in_stderr() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("state");
    assert_success(run(&config, &["init"]));
    let output = run(
        &config,
        &[
            "search",
            "--mode",
            "embedding",
            "--embedding-url",
            "http://127.0.0.1:9/embed?key=sk-1234567890abcdef&gemini=AIza1234567890abcdef1234567890abcdef1234",
            "needle",
        ],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("Error: "), "got: {stderr}");
    assert!(!stderr.contains("sk-"), "got: {stderr}");
    assert!(!stderr.contains("AIza"), "got: {stderr}");
    assert!(!stderr.contains("?key=sk-"), "got: {stderr}");
    assert!(!stderr.contains("key="), "got: {stderr}");
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
