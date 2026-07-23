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
        Some("CLAWGALLERY_VDR_MLX_FAKE"),
    ));
    let status = assert_success(run_without_embedding_url(
        &config,
        &["vdr", "status", "--json"],
        None,
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
fn vdr_plain_sync_jina_mlx_fake_auto_starts_with_backend_defaults() {
    // Given: one indexed image and only the managed Jina backend selection.
    let (_temp, config) = one_image_library();
    let python = python_for_mlx_tests();

    // When: sync runs without an external embedding server or explicit model dimensions.
    let synced = assert_success(run_without_embedding_url(
        &config,
        &["vdr", "sync", "--backend", "jina-mlx", "--python", &python],
        Some("CLAWGALLERY_VDR_JINA_MLX_FAKE"),
    ));
    let status = assert_success(run_without_embedding_url(
        &config,
        &["vdr", "status", "--json"],
        None,
    ));

    // Then: the managed Jina runtime starts and its exact index contract is persisted.
    assert!(
        synced.contains("starting managed jina-mlx embedding server at http://127.0.0.1:"),
        "managed Jina server startup should be observable, got: {synced}"
    );
    assert!(synced.contains("indexed 1"), "got: {synced}");
    let status: serde_json::Value = serde_json::from_str(&status).expect("status json");
    assert_eq!(
        status["model"],
        "jinaai/jina-embeddings-v5-omni-small-retrieval-mlx"
    );
    assert_eq!(status["dimensions"], 1024);
}

#[test]
fn embedding_search_auto_starts_jina_backend_from_index_model() {
    // Given: an image index created by the managed Jina MLX backend.
    let (_temp, config) = one_image_library();
    let python = python_for_mlx_tests();
    assert_success(run_without_embedding_url(
        &config,
        &["vdr", "sync", "--backend", "jina-mlx", "--python", &python],
        Some("CLAWGALLERY_VDR_JINA_MLX_FAKE"),
    ));

    // When: embedding search runs without restating a backend or endpoint.
    let search = assert_success(run_without_embedding_url(
        &config,
        &["search", "--mode", "embedding", "dog", "--json"],
        Some("CLAWGALLERY_VDR_JINA_MLX_FAKE"),
    ));

    // Then: the persisted model selects Jina MLX and returns the indexed image.
    assert!(search.contains("dog.png"), "got: {search}");
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
        None,
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
        Some("CLAWGALLERY_VDR_MLX_FAKE"),
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
