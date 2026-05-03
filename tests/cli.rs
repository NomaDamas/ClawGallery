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
        "created_at": "2026-05-03T00:00:00Z"
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
