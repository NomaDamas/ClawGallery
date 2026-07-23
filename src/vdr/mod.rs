use crate::{AppPaths, CaptionRecord, ImageRecord};
use anyhow::Result;
use clap::{Args, Subcommand};
use clawgallery_vdr::{
    CaptionDocument, EmbeddingSearchHit, ImageDocument, SearchConfig, SyncConfig, SyncOutcome,
    deactivate_image_vectors as deactivate_library_vectors, embedding_search,
    pending_embedding_count, similar_image_groups as library_similar_image_groups,
    status as library_status, sync,
};
use std::{collections::HashMap, env, path::PathBuf};

mod backend;
mod serve;

pub(crate) use backend::ServeBackend;
use backend::{DEFAULT_MANAGED_HOST, DEFAULT_MLX_DIMENSIONS, DEFAULT_MLX_MODEL, resolve_backend};
pub(crate) use clawgallery_vdr::SimilarImageGroup;
pub(crate) use clawgallery_vdr::{DEFAULT_DIMENSIONS, DEFAULT_MAX_RETRIES, DEFAULT_VDR_MODEL};

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
    #[arg(long)]
    pub(crate) model: Option<String>,
    #[arg(long)]
    pub(crate) dimensions: Option<usize>,
    #[arg(long, default_value_t = DEFAULT_MAX_RETRIES)]
    pub(crate) max_retries: usize,
    #[arg(long, conflicts_with = "no_auto_start")]
    pub(crate) auto_start: bool,
    #[arg(long)]
    pub(crate) no_auto_start: bool,
    #[arg(long, value_enum)]
    pub(crate) backend: Option<ServeBackend>,
    #[arg(long, default_value = DEFAULT_MANAGED_HOST)]
    pub(crate) host: String,
    #[arg(long, default_value_t = 0)]
    pub(crate) port: u16,
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
    #[arg(long)]
    pub(crate) python: Option<PathBuf>,
    #[arg(long)]
    pub(crate) allow_remote: bool,
}

#[derive(Debug, Args)]
struct VdrStatusArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct VdrServeArgs {
    #[arg(long, value_enum)]
    backend: Option<ServeBackend>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8765)]
    port: u16,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    dimensions: Option<usize>,
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
    let backend = resolve_backend(args.backend, args.model.as_deref(), args.dimensions)?;
    serve::serve(serve::ServeArgs {
        backend: backend.backend,
        host: args.host,
        port: args.port,
        model: backend.model,
        dimensions: backend.dimensions,
        device: args.device,
        python: args.python,
        allow_remote: args.allow_remote,
    })
}

pub(crate) fn cmd_sync(paths: &AppPaths, args: VdrSyncArgs) -> Result<()> {
    let backend = resolve_backend(args.backend, args.model.as_deref(), args.dimensions)?;
    let captions = crate::latest_captions_by_path(paths)?;
    let (images, refreshed_files) = crate::latest_images_refreshing_changed_files(paths)?;
    let config = SyncConfig {
        db_path: paths.vdr_db.clone(),
        model: backend.model.clone(),
        dimensions: backend.dimensions,
        embedding_url: args.embedding_url.clone(),
        max_retries: args.max_retries,
        prune: args.prune || refreshed_files,
    };
    let image_documents = image_documents(&images);
    let caption_documents = caption_documents_from_map(captions);
    let should_auto_start = should_auto_start(&args)
        && pending_embedding_count(&config, image_documents.clone(), caption_documents.clone())?
            > 0;
    let managed_server = should_auto_start.then(|| {
        serve::ManagedServer::start(&serve::ServeArgs {
            backend: backend.backend,
            host: args.host.clone(),
            port: args.port,
            model: backend.model.clone(),
            dimensions: backend.dimensions,
            device: args.device.clone(),
            python: args.python.clone(),
            allow_remote: args.allow_remote,
        })
    });
    let managed_server = managed_server.transpose()?;
    let embedding_url = args.embedding_url.or_else(|| {
        managed_server
            .as_ref()
            .map(|server| server.url().to_string())
    });
    let config = SyncConfig {
        embedding_url,
        ..config
    };
    let outcome = sync(&config, image_documents, caption_documents)?;
    print_sync_outcome(outcome);
    Ok(())
}

fn should_auto_start(args: &VdrSyncArgs) -> bool {
    if args.embedding_url.is_some() || args.no_auto_start {
        return false;
    }
    args.auto_start || env::var_os("CLAWGALLERY_VDR_EMBEDDING_URL").is_none()
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

pub(crate) fn embedding_search_hits(
    paths: &AppPaths,
    query: &str,
    limit: usize,
    embedding_url: Option<&str>,
    skip_empty_index: bool,
    images: Vec<ImageRecord>,
    captions: HashMap<PathBuf, CaptionRecord>,
) -> Result<Vec<EmbeddingSearchHit>> {
    let status = library_status(&paths.vdr_db, images.len())?;
    if status.active_vectors == 0 && skip_empty_index {
        return Ok(Vec::new());
    }
    let model = status
        .model
        .unwrap_or_else(|| DEFAULT_MLX_MODEL.to_string());
    let dimensions = status.dimensions.unwrap_or(DEFAULT_MLX_DIMENSIONS);
    let backend = resolve_backend(None, Some(&model), Some(dimensions))?;
    let env_embedding_url = env::var("CLAWGALLERY_VDR_EMBEDDING_URL").ok();
    let managed_server = if embedding_url.is_none() && env_embedding_url.is_none() {
        Some(serve::ManagedServer::start_quiet(&serve::ServeArgs {
            backend: backend.backend,
            host: DEFAULT_MANAGED_HOST.to_string(),
            port: 0,
            model: model.clone(),
            dimensions,
            device: "auto".to_string(),
            python: None,
            allow_remote: false,
        })?)
    } else {
        None
    };
    let embedding_url = embedding_url
        .map(str::to_string)
        .or(env_embedding_url)
        .or_else(|| {
            managed_server
                .as_ref()
                .map(|server| server.url().to_string())
        });
    embedding_search(
        &SearchConfig {
            db_path: paths.vdr_db.clone(),
            model: Some(model),
            dimensions: Some(dimensions),
            embedding_url,
            limit,
        },
        query,
        image_documents(&images),
        caption_documents_from_map(captions),
    )
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
