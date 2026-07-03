use crate::{AppPaths, CaptionRecord, ImageRecord};
use anyhow::Result;
use clap::{Args, Subcommand};
use clawgallery_vdr::{
    CaptionDocument, ImageDocument, SearchConfig, SyncConfig, SyncOutcome,
    deactivate_image_vectors as deactivate_library_vectors, embedding_search,
    similar_image_groups as library_similar_image_groups, status as library_status, sync,
};
use std::{collections::HashMap, path::PathBuf};

mod serve;

pub(crate) use clawgallery_vdr::SimilarImageGroup;
pub(crate) use clawgallery_vdr::{DEFAULT_DIMENSIONS, DEFAULT_MAX_RETRIES, DEFAULT_VDR_MODEL};

const DEFAULT_MLX_MODEL: &str = "qnguyen3/colqwen2.5-v0.2-mlx";
const DEFAULT_MLX_DIMENSIONS: usize = 128;

#[derive(Debug, Args)]
pub(crate) struct VdrArgs {
    #[command(subcommand)]
    command: VdrCommand,
}

#[derive(Debug, Subcommand)]
enum VdrCommand {
    /// Index active images and captions into the local VDR store.
    Sync(VdrSyncArgs),
    /// Show local VDR index metadata and counts.
    Status(VdrStatusArgs),
    /// Run a local embedding HTTP server for VDR sync and search.
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
    #[arg(long, default_value_t = DEFAULT_MAX_RETRIES)]
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

pub(crate) fn cmd_vdr(paths: &AppPaths, args: VdrArgs) -> Result<()> {
    paths.ensure()?;
    match args.command {
        VdrCommand::Sync(args) => cmd_sync(paths, args),
        VdrCommand::Status(args) => cmd_status(paths, args),
        VdrCommand::Serve(args) => cmd_serve(args),
    }
}

pub(crate) fn deactivate_image_vectors(paths: &AppPaths, image_id: &str) -> Result<()> {
    deactivate_library_vectors(&paths.vdr_db, image_id)
}

pub(crate) fn similar_image_groups(
    paths: &AppPaths,
    images: &[ImageRecord],
    threshold: f64,
) -> Result<Vec<SimilarImageGroup>> {
    library_similar_image_groups(&paths.vdr_db, &image_documents(images), threshold)
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
    let captions = crate::latest_captions_by_path(paths)?;
    let (images, refreshed_files) = crate::latest_images_refreshing_changed_files(paths)?;
    let outcome = sync(
        &SyncConfig {
            db_path: paths.vdr_db.clone(),
            model: args.model,
            dimensions: args.dimensions,
            embedding_url: args.embedding_url,
            max_retries: args.max_retries,
            prune: args.prune || refreshed_files,
        },
        image_documents(&images),
        caption_documents_from_map(captions),
    )?;
    print_sync_outcome(outcome);
    Ok(())
}

fn cmd_status(paths: &AppPaths, args: VdrStatusArgs) -> Result<()> {
    let active_images = crate::latest_images(paths)?.len();
    let status = library_status(&paths.vdr_db, active_images)?;
    if args.json {
        println!("{}", serde_json::to_string(&status)?);
    } else {
        println!("vdr_db: {}", paths.vdr_db.display());
        println!("active_images: {}", status.active_images);
        println!("active_vectors: {}", status.active_vectors);
        if let Some(model) = status.model {
            println!("model: {model}");
        }
        if let Some(dimensions) = status.dimensions {
            println!("dimensions: {dimensions}");
        }
    }
    Ok(())
}

pub(crate) fn cmd_embedding_search(
    paths: &AppPaths,
    query: &str,
    limit: usize,
    json_output: bool,
    embedding_url: Option<&str>,
    images: Vec<ImageRecord>,
    captions: HashMap<PathBuf, CaptionRecord>,
) -> Result<()> {
    let hits = embedding_search(
        &SearchConfig {
            db_path: paths.vdr_db.clone(),
            model: None,
            dimensions: None,
            embedding_url: embedding_url.map(str::to_string),
            limit,
        },
        query,
        image_documents(&images),
        caption_documents_from_map(captions),
    )?;
    for hit in hits {
        if json_output {
            println!("{}", serde_json::to_string(&hit)?);
        } else {
            println!(
                "{}\n  title: {}\n  caption: {}\n  score: {:.4}\n  matches: {} ({})",
                hit.path.display(),
                hit.title,
                hit.description,
                hit.score,
                hit.matched_field,
                query
            );
        }
    }
    Ok(())
}

fn print_sync_outcome(outcome: SyncOutcome) {
    if outcome.indexed_vectors == 0 {
        println!("indexed 0 vector(s), skipped unchanged");
    } else {
        println!("indexed {} vector(s)", outcome.indexed_vectors);
    }
}

fn image_documents(images: &[ImageRecord]) -> Vec<ImageDocument> {
    images
        .iter()
        .map(|image| ImageDocument {
            id: image.id.clone(),
            path: image.path.clone(),
            sha256: image.sha256.clone(),
        })
        .collect()
}

fn caption_documents_from_map(captions: HashMap<PathBuf, CaptionRecord>) -> Vec<CaptionDocument> {
    captions
        .into_values()
        .map(|caption| CaptionDocument {
            image_id: caption.image_id,
            path: caption.path,
            title: caption.title,
            description: caption.description,
        })
        .collect()
}
