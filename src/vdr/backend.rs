use anyhow::{Result, bail};
use clap::ValueEnum;

pub(super) const DEFAULT_MLX_MODEL: &str = "qnguyen3/colqwen2.5-v0.2-mlx";
pub(super) const DEFAULT_MLX_DIMENSIONS: usize = 128;
pub(super) const DEFAULT_MANAGED_HOST: &str = "127.0.0.1";
const JINA_MLX_MODEL: &str = "jinaai/jina-embeddings-v5-omni-small-retrieval-mlx";
const JINA_MLX_DIMENSIONS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeBackend {
    Mlx,
    JinaMlx,
}

pub(super) struct BackendConfig {
    pub(super) backend: ServeBackend,
    pub(super) model: String,
    pub(super) dimensions: usize,
}

pub(super) fn resolve_backend(
    backend: Option<ServeBackend>,
    model: Option<&str>,
    dimensions: Option<usize>,
) -> Result<BackendConfig> {
    let backend = backend.unwrap_or_else(|| {
        if model == Some(JINA_MLX_MODEL) {
            ServeBackend::JinaMlx
        } else {
            ServeBackend::Mlx
        }
    });
    let (default_model, default_dimensions) = match backend {
        ServeBackend::Mlx => (DEFAULT_MLX_MODEL, DEFAULT_MLX_DIMENSIONS),
        ServeBackend::JinaMlx => (JINA_MLX_MODEL, JINA_MLX_DIMENSIONS),
    };
    let model = model.unwrap_or(default_model);
    let dimensions = dimensions.unwrap_or(default_dimensions);
    if backend == ServeBackend::JinaMlx && model != JINA_MLX_MODEL {
        bail!("jina-mlx requires model {JINA_MLX_MODEL}");
    }
    if backend == ServeBackend::JinaMlx && dimensions != JINA_MLX_DIMENSIONS {
        bail!("jina-mlx requires {JINA_MLX_DIMENSIONS} dimensions");
    }
    Ok(BackendConfig {
        backend,
        model: model.to_string(),
        dimensions,
    })
}
