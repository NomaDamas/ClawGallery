#[path = "vdr_autosync_support/mod.rs"]
mod vdr_autosync_support;

use vdr_autosync_support::{
    FakeEmbeddingServer, assert_success, image_library, run_without_embedding_url,
};

#[test]
fn vdr_sync_batches_large_pending_sets_into_single_input_requests() {
    let server = FakeEmbeddingServer::start();
    let (_temp, config) = image_library(&["dog.png", "cat.png"]);

    let synced = assert_success(run_without_embedding_url(
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
    let status = assert_success(run_without_embedding_url(
        &config,
        &["vdr", "status", "--json"],
        false,
    ));

    assert!(synced.contains("indexed 2"), "got: {synced}");
    assert_eq!(server.request_count(), 2);
    let status: serde_json::Value = serde_json::from_str(&status).expect("status json");
    assert_eq!(status["active_images"], 2);
    assert_eq!(status["active_vectors"], 2);
}
