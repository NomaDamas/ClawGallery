#[path = "vdr_autosync_support/mod.rs"]
mod vdr_autosync_support;

use vdr_autosync_support::{
    FakeEmbeddingServer, assert_success, one_image_library, run_without_embedding_url,
};

fn python_for_mlx_tests() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

#[test]
fn vdr_plain_sync_mlx_fake_auto_starts_without_external_server() {
    let (_temp, config) = one_image_library();
    let python = python_for_mlx_tests();

    let synced = assert_success(run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--backend",
            "mlx",
            "--python",
            &python,
            "--model",
            "test-model",
            "--dimensions",
            "4",
        ],
        true,
    ));
    let status = assert_success(run_without_embedding_url(
        &config,
        &["vdr", "status", "--json"],
        false,
    ));

    assert!(
        synced.contains("starting managed mlx embedding server at http://127.0.0.1:"),
        "managed server startup should be observable, got: {synced}"
    );
    assert!(synced.contains("indexed 1"), "got: {synced}");
    let status: serde_json::Value = serde_json::from_str(&status).expect("status json");
    assert_eq!(status["active_images"], 1);
    assert_eq!(status["active_vectors"], 1);
}

#[test]
fn vdr_auto_start_skips_managed_server_when_nothing_changed() {
    let server = FakeEmbeddingServer::start();
    let (temp, config) = one_image_library();
    let missing_python = temp.path().join("missing-python");
    assert_success(run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--embedding-url",
            server.url(),
            "--model",
            "test-model",
            "--dimensions",
            "4",
        ],
        false,
    ));

    let synced = assert_success(run_without_embedding_url(
        &config,
        &[
            "vdr",
            "sync",
            "--python",
            missing_python.to_str().expect("utf8"),
            "--model",
            "test-model",
            "--dimensions",
            "4",
        ],
        true,
    ));

    assert!(
        !synced.contains("starting managed mlx embedding server"),
        "no-op sync should not start Python, got: {synced}"
    );
    assert!(
        synced.contains("indexed 0 vector(s), skipped unchanged"),
        "expected no-op sync, got: {synced}"
    );
    assert_eq!(server.request_count(), 1);
}
