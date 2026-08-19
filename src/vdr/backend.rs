use anyhow::{Result, bail};
use clap::ValueEnum;

pub(super) const DEFAULT_MLX_MODEL: &str = "qnguyen3/colqwen2.5-v0.2-mlx";
pub(super) const DEFAULT_MLX_DIMENSIONS: usize = 128;
pub(super) const DEFAULT_MANAGED_HOST: &str = "127.0.0.1";
const JINA_MLX_MODEL: &str = "jinaai/jina-embeddings-v5-omni-small-retrieval-mlx";
const JINA_MLX_DIMENSIONS: usize = 1024;
pub(super) const DEFAULT_VSPLADE_MODEL: &str = clawgallery_vdr::DEFAULT_VSPLADE_MODEL;
pub(super) const DEFAULT_VSPLADE_DIMENSIONS: usize = clawgallery_vdr::DEFAULT_VSPLADE_DIMENSIONS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeBackend {
    Mlx,
    JinaMlx,
    Vsplade,
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
        } else if model.is_some_and(|value| value.contains("v-splade") || value.contains("vsplade"))
        {
            ServeBackend::Vsplade
        } else {
            ServeBackend::Mlx
        }
    });
    let (default_model, default_dimensions) = match backend {
        ServeBackend::Mlx => (DEFAULT_MLX_MODEL, DEFAULT_MLX_DIMENSIONS),
        ServeBackend::JinaMlx => (JINA_MLX_MODEL, JINA_MLX_DIMENSIONS),
        ServeBackend::Vsplade => (DEFAULT_VSPLADE_MODEL, DEFAULT_VSPLADE_DIMENSIONS),
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
