#[path = "vdr_autosync_support/mod.rs"]
mod vdr_autosync_support;

use vdr_autosync_support::{
    FakeEmbeddingServer, assert_success, one_image_library, run_without_embedding_url,
};

#[test]
fn vdr_auto_start_embedding_url_uses_external_server_without_managed_child() {
    let server = FakeEmbeddingServer::start();
    let (temp, config) = one_image_library();
    let missing_python = temp.path().join("missing-python");

    let synced = assert_success(run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--embedding-url",
            server.url(),
            "--python",
            missing_python.to_str().expect("utf8"),
            "--model",
            "test-model",
            "--dimensions",
            "4",
        ],
        Some("CLAWGALLERY_VDR_MLX_FAKE"),
    ));

    assert!(synced.contains("indexed 1"), "got: {synced}");
    assert!(
        !synced.contains("starting managed mlx embedding server"),
        "external URL must not auto-start, got: {synced}"
    );
    assert_eq!(server.request_count(), 1);
}

#[test]
fn vdr_auto_start_no_auto_start_preserves_external_server_failure() {
    let (_temp, config) = one_image_library();

    let output = run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--no-auto-start",
            "--dimensions",
            "4",
            "--max-retries",
            "0",
        ],
        None,
    );

    assert!(
        !output.status.success(),
        "sync should fail without a server"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("127.0.0.1:8765"),
        "expected default embedding URL failure, got: {stderr}"
    );
}

#[test]
fn vdr_auto_start_missing_python_path_fails_cleanly() {
    let (temp, config) = one_image_library();
    let missing_python = temp.path().join("missing-python");

    let output = run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--auto-start",
            "--python",
            missing_python.to_str().expect("utf8"),
            "--dimensions",
            "4",
        ],
        Some("CLAWGALLERY_VDR_MLX_FAKE"),
    );

    assert!(!output.status.success(), "sync should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to start Python interpreter")
            && stderr.contains(missing_python.to_str().expect("utf8")),
        "expected missing interpreter in error, got: {stderr}"
    );
}
