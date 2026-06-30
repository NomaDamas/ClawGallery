use std::{fs, io::Write};

#[path = "vdr_support/mod.rs"]
mod vdr_support;

use vdr_support::{FakeEmbeddingServer, assert_success, image_id_for, run, write_caption};

#[test]
fn vdr_sync_retries_transient_429_then_succeeds() {
    let server = FakeEmbeddingServer::start_with_statuses(vec![429, 200]);
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("retry.png"), b"retry image bytes").expect("write image");

    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));

    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    assert!(synced.contains("indexed 1"), "got: {synced}");
    assert_eq!(server.request_count(), 2);
}

#[test]
fn vdr_late_interaction_ranks_with_multivector_maxsim() {
    // Given: a multi-vector (late-interaction) embedding server and two images.
    let server = FakeEmbeddingServer::start_multivector();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    fs::write(images.join("cat.png"), b"cat image bytes").expect("write cat image");

    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    let (cat_id, cat_path) = image_id_for(&config, "cat.png");
    write_caption(
        &config,
        &dog_id,
        &dog_path,
        "Dog Park",
        "puppy playing outside",
    );
    write_caption(
        &config,
        &cat_id,
        &cat_path,
        "Cat Sofa",
        "kitten sleeping indoors",
    );

    // When: sync stores multi-vectors and a multi-token query searches.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    assert!(
        synced.contains("indexed 4"),
        "expected paired vectors, got: {synced}"
    );
    let search = assert_success(run(
        &config,
        &[
            "search",
            "--mode",
            "embedding",
            "puppy playing",
            "--json",
            "--limit",
            "2",
        ],
        server.url(),
    ));

    // Then: MaxSim over token vectors ranks the dog image first with the
    // average of per-token maxima, proving real late interaction.
    let rows: Vec<serde_json::Value> = search
        .lines()
        .map(|line| serde_json::from_str(line).expect("json result"))
        .collect();
    assert_eq!(rows.len(), 2, "expected two rows, got: {search}");
    assert!(
        rows[0]["path"].as_str().expect("path").ends_with("dog.png"),
        "dog should rank first, got: {search}"
    );
    let top = rows[0]["score"].as_f64().expect("score");
    let bottom = rows[1]["score"].as_f64().expect("score");
    // Query "puppy playing" = [dog-axis, other-axis]; the dog caption doc has
    // both axes so MaxSim = (1.0 + 1.0) / 2 = 1.0, cat doc = (0.0 + 1.0) / 2.
    assert!((top - 1.0).abs() < 1e-6, "expected maxsim 1.0, got {top}");
    assert!(
        (bottom - 0.5).abs() < 1e-6,
        "expected maxsim 0.5, got {bottom}"
    );
}

#[test]
fn vdr_embedding_search_matches_image_or_caption_embeddings() {
    // Given: two tracked images with captions whose text embeddings are distinct.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    fs::write(images.join("cat.png"), b"cat image bytes").expect("write cat image");

    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["folder", "add", images.to_str().expect("utf8")],
        server.url(),
    ));
    assert_success(run(&config, &["bootstrap"], server.url()));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    let (cat_id, cat_path) = image_id_for(&config, "cat.png");
    write_caption(
        &config,
        &dog_id,
        &dog_path,
        "Dog Park",
        "puppy playing outside",
    );
    write_caption(
        &config,
        &cat_id,
        &cat_path,
        "Cat Sofa",
        "kitten sleeping indoors",
    );

    // When: VDR sync indexes both image and caption vectors, then embedding search queries dog.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    assert!(
        synced.contains("indexed 4"),
        "expected paired vectors, got: {synced}"
    );
    let search = assert_success(run(
        &config,
        &[
            "search",
            "--mode",
            "embedding",
            "dog",
            "--json",
            "--limit",
            "2",
        ],
        server.url(),
    ));

    // Then: either the image vector or caption vector can match the dog image first.
    let rows: Vec<serde_json::Value> = search
        .lines()
        .map(|line| serde_json::from_str(line).expect("json result"))
        .collect();
    assert_eq!(
        rows.len(),
        2,
        "expected two embedding search rows, got: {search}"
    );
    assert_eq!(rows[0]["source"], "embedding");
    assert!(
        rows[0]["path"].as_str().expect("path").ends_with("dog.png"),
        "dog should rank first, got: {search}"
    );
    assert!(
        rows[0]["matched_field"] == "embedding_image"
            || rows[0]["matched_field"] == "embedding_caption",
        "matched field should identify the embedding modality, got: {search}"
    );
}

#[test]
fn vdr_sync_prunes_deleted_and_indexes_added_images() {
    // Given: one indexed image, then a filesystem deletion and a newly added image.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    let old = images.join("old-dog.png");
    fs::write(&old, b"dog image bytes").expect("write old image");

    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["folder", "add", images.to_str().expect("utf8")],
        server.url(),
    ));
    assert_success(run(&config, &["bootstrap"], server.url()));
    let (old_id, old_path) = image_id_for(&config, "old-dog.png");
    write_caption(
        &config,
        &old_id,
        &old_path,
        "Old Dog",
        "puppy should disappear",
    );
    assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));

    fs::remove_file(&old).expect("delete old image");
    fs::write(images.join("new.png"), b"new fresh image").expect("write new image");
    assert_success(run(&config, &["bootstrap", "--prune"], server.url()));
    let (new_id, new_path) = image_id_for(&config, "new.png");
    write_caption(
        &config,
        &new_id,
        &new_path,
        "Fresh New",
        "new searchable image",
    );

    // When: the incremental sync runs with pruning.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--prune", "--dimensions", "4"],
        server.url(),
    ));
    let status = assert_success(run(&config, &["vdr", "status", "--json"], server.url()));
    let search = assert_success(run(
        &config,
        &["search", "--mode", "embedding", "new", "--json"],
        server.url(),
    ));
    let old_search = assert_success(run(
        &config,
        &["search", "--mode", "embedding", "dog", "--json"],
        server.url(),
    ));

    // Then: new vectors are added and pruned image vectors are no longer active/searchable.
    assert!(
        synced.contains("indexed 2"),
        "new image + caption indexed, got: {synced}"
    );
    let status: serde_json::Value = serde_json::from_str(&status).expect("status json");
    assert_eq!(status["active_images"], 1);
    assert_eq!(status["active_vectors"], 2);
    assert!(
        search.contains("new.png"),
        "new image should be searchable, got: {search}"
    );
    assert!(
        !old_search.contains("old-dog.png"),
        "deleted image should not be searchable, got: {old_search}"
    );
}

#[test]
fn forget_deactivates_vdr_vectors_for_image() {
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    let image = images.join("dog.png");
    fs::write(&image, b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");
    assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));

    let forgot = assert_success(run(
        &config,
        &["forget", "--file", dog_path.to_str().expect("utf8")],
        server.url(),
    ));
    let status = assert_success(run(&config, &["vdr", "status", "--json"], server.url()));

    assert!(forgot.contains("forgot 1 image"), "got: {forgot}");
    let status: serde_json::Value = serde_json::from_str(&status).expect("status json");
    assert_eq!(status["active_images"], 0);
    assert_eq!(status["active_vectors"], 0);
}

#[test]
fn vdr_sync_second_run_skips_already_indexed_images() {
    // Given: one image has already been synced.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");

    assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    let requests_after_first_sync = server.request_count();

    // When: sync runs again with unchanged sha/model/dimensions.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));

    // Then: no new vectors are written and no extra embedding request is needed.
    assert!(
        synced.contains("indexed 0"),
        "second sync should skip, got: {synced}"
    );
    assert_eq!(server.request_count(), requests_after_first_sync);
}

#[test]
fn vdr_sync_fails_when_server_returns_different_model() {
    // Given: a server that reports a different model than the sync request.
    let server = FakeEmbeddingServer::start_with_response_model("different-model");
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));

    // When: sync requests the normal test model/dimensions.
    let output = run(
        &config,
        &[
            "vdr",
            "sync",
            "--model",
            "requested-model",
            "--dimensions",
            "4",
        ],
        server.url(),
    );

    // Then: the mismatch is rejected before any misleading metadata is stored.
    assert!(
        !output.status.success(),
        "sync should fail on model mismatch"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "embedding server returned model different-model but requested-model was requested"
        ),
        "expected model mismatch, got: {stderr}"
    );
    assert!(
        stderr.contains("pass --model/--dimensions matching the running server"),
        "expected remediation hint, got: {stderr}"
    );
}

#[test]
fn vdr_sync_updates_vector_path_after_image_record_path_changes() {
    // Given: an indexed image whose latest image record keeps the id/sha but changes path.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    let new_path = images.join("renamed-dog.png");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");
    assert_success(run(
        &config,
        &["vdr", "sync", "--model", "test-model", "--dimensions", "4"],
        server.url(),
    ));
    let requests_after_first_sync = server.request_count();

    let raw_images = fs::read_to_string(config.join("images.jsonl")).expect("images jsonl");
    let mut renamed_record: serde_json::Value = raw_images
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("image record"))
        .find(|record: &serde_json::Value| record["id"] == dog_id)
        .expect("dog record");
    renamed_record["path"] = serde_json::Value::String(new_path.to_string_lossy().to_string());
    let mut images_jsonl = fs::OpenOptions::new()
        .append(true)
        .open(config.join("images.jsonl"))
        .expect("open images jsonl");
    writeln!(images_jsonl, "{renamed_record}").expect("append renamed image record");
    write_caption(&config, &dog_id, &new_path, "Dog", "puppy");

    // When: sync sees the same id/sha/content under a different latest path.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--model", "test-model", "--dimensions", "4"],
        server.url(),
    ));
    let search = assert_success(run(
        &config,
        &["search", "--mode", "embedding", "dog", "--json"],
        server.url(),
    ));

    // Then: no re-embedding is needed, but search reports the current image path.
    assert!(
        synced.contains("indexed 0"),
        "path-only update should not re-index, got: {synced}"
    );
    assert_eq!(server.request_count(), requests_after_first_sync + 1);
    assert!(
        search.contains("renamed-dog.png"),
        "embedding search should return latest path, got: {search}"
    );
    let rows: Vec<serde_json::Value> = search
        .lines()
        .map(|line| serde_json::from_str(line).expect("json result"))
        .collect();
    assert!(
        rows.iter()
            .all(|row| !row["path"].as_str().expect("path").ends_with("/dog.png")),
        "embedding search should not return stale path, got: {search}"
    );
}

#[test]
fn vdr_sync_reindexes_when_file_sha_changes() {
    // Given: an indexed image whose bytes change without changing its path.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    let image = images.join("dog.png");
    fs::write(&image, b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");
    assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));

    fs::write(&image, b"dog image bytes changed").expect("modify image bytes");

    // When: VDR sync runs again without a path change.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));

    // Then: image and caption vectors are refreshed for the new file sha.
    assert!(
        synced.contains("indexed 2"),
        "changed same-path image should be re-indexed, got: {synced}"
    );
}

#[test]
fn vdr_sync_reindexes_when_caption_changes() {
    // Given: an indexed image whose latest caption record changes.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");
    assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    write_caption(
        &config,
        &dog_id,
        &dog_path,
        "Fresh New",
        "new searchable image",
    );

    // When: only the caption content changes.
    let synced = assert_success(run(
        &config,
        &["vdr", "sync", "--dimensions", "4"],
        server.url(),
    ));
    let search = assert_success(run(
        &config,
        &["search", "--mode", "embedding", "new", "--json"],
        server.url(),
    ));

    // Then: the caption vector is refreshed and can satisfy the new query.
    assert!(
        synced.contains("indexed 1"),
        "caption-only change should index one caption vector, got: {synced}"
    );
    assert!(
        search.contains("dog.png"),
        "updated caption should be searchable, got: {search}"
    );
}

#[test]
fn embedding_search_uses_latest_index_model_and_dimensions() {
    // Given: VDR sync used a non-default model and dimensions.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("dog.png"), b"dog image bytes").expect("write dog image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (dog_id, dog_path) = image_id_for(&config, "dog.png");
    write_caption(&config, &dog_id, &dog_path, "Dog", "puppy");
    assert_success(run(
        &config,
        &["vdr", "sync", "--model", "custom-jina", "--dimensions", "4"],
        server.url(),
    ));

    // When: embedding search runs without restating those sync options.
    let search = assert_success(run(
        &config,
        &["search", "--mode", "embedding", "dog", "--json"],
        server.url(),
    ));

    // Then: search uses the latest active index config instead of default dimensions.
    assert!(search.contains("dog.png"), "got: {search}");
}

#[test]
fn keyword_search_without_embedding_preserves_fuzzy_ranking() {
    // Given: normal caption metadata and no VDR index/server requirement.
    let server = FakeEmbeddingServer::start();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = temp.path().join("state");
    let images = temp.path().join("images");
    fs::create_dir_all(&images).expect("create images");
    fs::write(images.join("login.png"), b"login").expect("write image");
    assert_success(run(&config, &["init"], server.url()));
    assert_success(run(
        &config,
        &["bootstrap", "--path", images.to_str().expect("utf8")],
        server.url(),
    ));
    let (id, path) = image_id_for(&config, "login.png");
    write_caption(&config, &id, &path, "Login Dialog", "A settings screen");

    // When: search runs in the default keyword mode.
    let stdout = assert_success(run(&config, &["search", "login", "--json"], server.url()));

    // Then: existing fuzzy JSON output is preserved.
    let first: serde_json::Value =
        serde_json::from_str(stdout.lines().next().expect("one keyword search result"))
            .expect("json result");
    assert_eq!(first["source"], "fuzzy");
    assert!(first["path"].as_str().expect("path").ends_with("login.png"));
    assert_eq!(
        server.request_count(),
        0,
        "keyword search must not call embedding server"
    );
}
