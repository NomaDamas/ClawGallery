use crate::{AppPaths, ImageRecord};
use anyhow::Result;
use clap::{Args, Subcommand};
use std::{collections::HashMap, path::PathBuf};

mod client;
mod index;
mod search;
mod serve;
mod store;

pub(crate) use client::DEFAULT_MAX_RETRIES;
pub(crate) use search::cmd_embedding_search;

const DEFAULT_EMBEDDING_URL: &str = "http://127.0.0.1:8765";
pub(crate) const DEFAULT_VDR_MODEL: &str = "vidore/colqwen2-v1.0";
pub(crate) const DEFAULT_DIMENSIONS: usize = 128;
const DEFAULT_MLX_MODEL: &str = "qnguyen3/colqwen2.5-v0.2-mlx";
const DEFAULT_MLX_DIMENSIONS: usize = 128;

#[derive(Debug, Args)]
pub(crate) struct VdrArgs {
    #[command(subcommand)]
    command: VdrCommand,
}

#[derive(Debug, Subcommand)]
enum VdrCommand {
    Sync(VdrSyncArgs),
    Status(VdrStatusArgs),
    Serve(VdrServeArgs),
}

#[derive(Debug, Args)]
pub(crate) struct VdrSyncArgs {
    #[arg(long)]
    pub(crate) prune: bool,
    #[arg(long)]
    pub(crate) embedding_url: Option<String>,
    #[arg(long, default_value = DEFAULT_VDR_MODEL)]
    pub(crate) model: String,
    #[arg(long, default_value_t = DEFAULT_DIMENSIONS)]
    pub(crate) dimensions: usize,
    #[arg(long, default_value_t = client::DEFAULT_MAX_RETRIES)]
    pub(crate) max_retries: usize,
}

#[derive(Debug, Args)]
struct VdrStatusArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct VdrServeArgs {
    #[arg(long, value_enum, default_value_t = serve::ServeBackend::Mlx)]
    backend: serve::ServeBackend,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8765)]
    port: u16,
    #[arg(long, default_value = DEFAULT_MLX_MODEL)]
    model: String,
    #[arg(long, default_value_t = DEFAULT_MLX_DIMENSIONS)]
    dimensions: usize,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long)]
    python: Option<PathBuf>,
    #[arg(long)]
    allow_remote: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingKind {
    Image,
    Caption,
}

impl EmbeddingKind {
    const fn as_str(self) -> &'static str {
        match self {
            EmbeddingKind::Image => "image",
            EmbeddingKind::Caption => "caption",
        }
    }

    const fn matched_field(self) -> &'static str {
        match self {
            EmbeddingKind::Image => "embedding_image",
            EmbeddingKind::Caption => "embedding_caption",
        }
    }
}

#[derive(Debug)]
struct PendingEmbedding {
    image_id: String,
    path: PathBuf,
    sha256: String,
    content_hash: String,
    kind: EmbeddingKind,
    value: String,
}

#[derive(Debug)]
pub(crate) struct SimilarImageDuplicate {
    pub(crate) image_id: String,
    pub(crate) score: f64,
}

#[derive(Debug)]
pub(crate) struct SimilarImageGroup {
    pub(crate) representative_id: String,
    pub(crate) duplicates: Vec<SimilarImageDuplicate>,
}

pub(crate) fn cmd_vdr(paths: &AppPaths, args: VdrArgs) -> Result<()> {
    paths.ensure()?;
    match args.command {
        VdrCommand::Sync(args) => cmd_sync(paths, args),
        VdrCommand::Status(args) => cmd_status(paths, args),
        VdrCommand::Serve(args) => cmd_serve(args),
    }
}

pub(crate) fn deactivate_image_vectors(paths: &AppPaths, image_id: &str) -> Result<()> {
    if !paths.vdr_db.exists() {
        return Ok(());
    }
    let conn = store::open_store(paths)?;
    store::deactivate_image_vectors(&conn, image_id)
}

pub(crate) fn similar_image_groups(
    paths: &AppPaths,
    images: &[ImageRecord],
    threshold: f64,
) -> Result<Vec<SimilarImageGroup>> {
    let conn = store::open_store(paths)?;
    let index_config =
        index::latest_active_index_config(&conn)?.unwrap_or_else(|| index::ActiveIndexConfig {
            model: client::default_model().to_string(),
            dimensions: DEFAULT_DIMENSIONS,
        });
    let active_images: HashMap<String, ImageRecord> = images
        .iter()
        .map(|image| (image.id.clone(), image.clone()))
        .collect();
    let mut vectors: Vec<_> = store::active_vectors(
        &conn,
        &active_images,
        &index_config.model,
        index_config.dimensions,
    )?
    .into_iter()
    .filter(|stored| stored.kind == EmbeddingKind::Image)
    .collect();
    vectors.sort_by(|left, right| {
        active_images[&left.image_id]
            .path
            .cmp(&active_images[&right.image_id].path)
    });
    let mut parents: Vec<usize> = (0..vectors.len()).collect();
    let mut scores: HashMap<(usize, usize), f64> = HashMap::new();
    for left in 0..vectors.len() {
        for right in (left + 1)..vectors.len() {
            let forward =
                search::late_interaction_score(&vectors[left].vectors, &vectors[right].vectors)?;
            let backward =
                search::late_interaction_score(&vectors[right].vectors, &vectors[left].vectors)?;
            let score = (forward + backward) / 2.0;
            if score >= threshold {
                union(&mut parents, left, right);
                scores.insert((left, right), score);
            }
        }
    }
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..vectors.len() {
        components
            .entry(find(&mut parents, index))
            .or_default()
            .push(index);
    }
    let mut groups = Vec::new();
    for component in components.values_mut() {
        if component.len() < 2 {
            continue;
        }
        component.sort_by(|left, right| {
            active_images[&vectors[*left].image_id]
                .path
                .cmp(&active_images[&vectors[*right].image_id].path)
        });
        let representative = component[0];
        let duplicates = component[1..]
            .iter()
            .map(|index| SimilarImageDuplicate {
                image_id: vectors[*index].image_id.clone(),
                score: best_component_score(*index, component, &scores),
            })
            .collect();
        groups.push(SimilarImageGroup {
            representative_id: vectors[representative].image_id.clone(),
            duplicates,
        });
    }
    groups.sort_by(|left, right| {
        active_images[&left.representative_id]
            .path
            .cmp(&active_images[&right.representative_id].path)
    });
    Ok(groups)
}

fn best_component_score(
    target: usize,
    component: &[usize],
    scores: &HashMap<(usize, usize), f64>,
) -> f64 {
    component
        .iter()
        .filter(|index| **index != target)
        .filter_map(|index| {
            let key = if *index < target {
                (*index, target)
            } else {
                (target, *index)
            };
            scores.get(&key).copied()
        })
        .fold(0.0, f64::max)
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find(parents, left);
    let right_root = find(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn find(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = find(parents, parents[index]);
    }
    parents[index]
}

fn cmd_serve(args: VdrServeArgs) -> Result<()> {
    serve::serve(serve::ServeArgs {
        backend: args.backend,
        host: args.host,
        port: args.port,
        model: args.model,
        dimensions: args.dimensions,
        device: args.device,
        python: args.python,
        allow_remote: args.allow_remote,
    })
}

pub(crate) fn cmd_sync(paths: &AppPaths, args: VdrSyncArgs) -> Result<()> {
    let conn = store::open_store(paths)?;
    let captions = crate::latest_captions_by_path(paths)?;
    let (images, refreshed_files) = crate::latest_images_refreshing_changed_files(paths)?;
    if args.prune || refreshed_files {
        store::prune_inactive_vectors(&conn, &images)?;
    }
    store::update_active_vector_paths(&conn, &images, &args.model, args.dimensions)?;
    let pending =
        store::pending_embeddings(&conn, images, &captions, &args.model, args.dimensions)?;
    if pending.is_empty() {
        println!("indexed 0 vector(s), skipped unchanged");
        return Ok(());
    }
    let inputs = pending
        .iter()
        .map(|item| client::EmbedInput {
            kind: item.kind.as_str().to_string(),
            role: "document".to_string(),
            value: item.value.clone(),
        })
        .collect();
    let url = client::resolve_embedding_url(args.embedding_url.as_deref());
    let response =
        client::embed_with_retries(&url, &args.model, args.dimensions, inputs, args.max_retries)?;
    if response.embeddings.len() != pending.len() {
        anyhow::bail!(
            "embedding server returned {} embedding(s) for {} input(s)",
            response.embeddings.len(),
            pending.len()
        );
    }
    let indexed = response.embeddings.len();
    for (item, vector) in pending.into_iter().zip(response.embeddings) {
        store::deactivate_existing_kind(&conn, &item.image_id, item.kind)?;
        store::insert_vector(&conn, &item, &response.model, args.dimensions, &vector)?;
    }
    println!("indexed {indexed} vector(s)");
    Ok(())
}

fn cmd_status(paths: &AppPaths, args: VdrStatusArgs) -> Result<()> {
    let conn = store::open_store(paths)?;
    let status = index::status(paths, &conn)?;
    if args.json {
        println!("{}", serde_json::to_string(&status)?);
    } else {
        let value = serde_json::to_value(&status)?;
        println!("vdr_db: {}", paths.vdr_db.display());
        println!("active_images: {}", value["active_images"]);
        println!("active_vectors: {}", value["active_vectors"]);
        if let Some(model) = value["model"].as_str() {
            println!("model: {model}");
        }
        if let Some(dimensions) = value["dimensions"].as_u64() {
            println!("dimensions: {dimensions}");
        }
    }
    Ok(())
}
