use std::{path::PathBuf, process::Command};

fn clawgallery_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_clawgallery"))
}

#[test]
fn vdr_serve_mlx_help_is_packaged_for_installed_cli() {
    // Given: the installed ClawGallery binary without repository script paths.
    let temp = tempfile::tempdir().expect("tempdir");

    // When: the mlx backend server help is requested through the VDR CLI.
    let output = Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", temp.path().join("state"))
        .env("CLAWGALLERY_VDR_MLX_FAKE", "1")
        .args(["vdr", "serve", "--backend", "mlx", "--help"])
        .output()
        .expect("clawgallery command should run");

    // Then: the packaged daemon surface is discoverable from the binary.
    assert!(
        output.status.success(),
        "serve help should succeed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--dimensions"), "got: {stdout}");
    assert!(stdout.contains("mlx"), "got: {stdout}");
}

#[test]
fn vdr_serve_help_lists_jina_mlx_backend() {
    // Given: the installed ClawGallery binary.
    let temp = tempfile::tempdir().expect("tempdir");

    // When: the managed VDR server help is requested.
    let output = Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", temp.path().join("state"))
        .args(["vdr", "serve", "--help"])
        .output()
        .expect("clawgallery command should run");

    // Then: the new packaged backend is discoverable as a supported value.
    assert!(output.status.success(), "serve help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("jina-mlx"), "got: {stdout}");
}

#[test]
fn vdr_serve_jina_mlx_rejects_incompatible_dimensions() {
    // Given: the Jina MLX backend with a ColQwen-sized vector request.
    let temp = tempfile::tempdir().expect("tempdir");

    // When: the server is started with 128 dimensions.
    let output = Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", temp.path().join("state"))
        .env("CLAWGALLERY_VDR_JINA_MLX_FAKE", "1")
        .args([
            "vdr",
            "serve",
            "--backend",
            "jina-mlx",
            "--dimensions",
            "128",
        ])
        .output()
        .expect("clawgallery command should run");

    // Then: validation fails before a model process can start.
    assert!(!output.status.success(), "invalid dimensions should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("requires 1024 dimensions"), "got: {stderr}");
}

#[test]
fn vdr_serve_mlx_rejects_remote_bind_without_allow_remote() {
    // Given: a request to expose the unauthenticated local-file embedding server.
    let temp = tempfile::tempdir().expect("tempdir");

    // When: the mlx backend is started on a non-loopback host without an opt-in.
    let output = Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", temp.path().join("state"))
        .env("CLAWGALLERY_VDR_MLX_FAKE", "1")
        .args([
            "vdr",
            "serve",
            "--backend",
            "mlx",
            "--host",
            "0.0.0.0",
            "--port",
            "8877",
        ])
        .output()
        .expect("clawgallery command should run");

    // Then: the launcher refuses the unsafe bind before starting the daemon.
    assert!(
        !output.status.success(),
        "remote bind should fail\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refusing to bind") && stderr.contains("non-loopback"),
        "expected non-loopback refusal, got: {stderr}"
    );
}

#[test]
fn vdr_serve_mlx_accepts_allow_remote_flag_for_python_launcher() {
    // Given: the launcher is asked to forward the explicit remote-bind opt-in.
    let temp = tempfile::tempdir().expect("tempdir");

    // When: help is requested with --allow-remote present.
    let output = Command::new(clawgallery_bin())
        .env("CLAWGALLERY_CONFIG_DIR", temp.path().join("state"))
        .env("CLAWGALLERY_VDR_MLX_FAKE", "1")
        .args([
            "vdr",
            "serve",
            "--backend",
            "mlx",
            "--allow-remote",
            "--help",
        ])
        .output()
        .expect("clawgallery command should run");

    // Then: clap accepts the flag before the Python process is launched.
    assert!(
        output.status.success(),
        "serve help with --allow-remote should succeed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
