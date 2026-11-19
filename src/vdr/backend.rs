use anyhow::{Result, bail};
use clap::ValueEnum;

pub(super) const DEFAULT_MLX_MODEL: &str = "qnguyen3/colqwen2.5-v0.2-mlx";
pub(super) const DEFAULT_MLX_DIMENSIONS: usize = 128;
pub(super) const DEFAULT_MANAGED_HOST: &str = "127.0.0.1";
pub(super) const DEFAULT_COLQWEN_MODEL: &str = "vidore/colqwen2-v1.0";
pub(super) const DEFAULT_COLQWEN_DIMENSIONS: usize = 128;
const JINA_MLX_MODEL: &str = "jinaai/jina-embeddings-v5-omni-small-retrieval-mlx";
const JINA_MLX_DIMENSIONS: usize = 1024;
pub(super) const DEFAULT_INL_MODEL: &str = clawgallery_vdr::DEFAULT_MULTIMODAL_MODEL;
pub(super) const DEFAULT_INL_DIMENSIONS: usize = clawgallery_vdr::DEFAULT_MULTIMODAL_DIMENSIONS;
pub(super) const INL_MISSING_ERROR: &str = "the inl backend requires the im-not-llm `inl` binary on PATH; install it with `cargo install inl-server --locked`";
pub(super) const DEFAULT_VSPLADE_MODEL: &str = clawgallery_vdr::DEFAULT_VSPLADE_MODEL;
pub(super) const DEFAULT_VSPLADE_DIMENSIONS: usize = clawgallery_vdr::DEFAULT_VSPLADE_DIMENSIONS;
#[cfg(windows)]
const WINDOWS_VSPLADE_MODEL: &str = "naver/v-splade-efficient";
pub(super) const APPLE_ONLY_BACKEND_ERROR: &str = "the mlx and jina-mlx backends require Apple Silicon MLX; on Windows use --backend colqwen with a PyTorch/colpali-engine environment";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeBackend {
    Mlx,
    JinaMlx,
    Colqwen,
    Vsplade,
    /// Embed through the im-not-llm (`inl`) multimodal daemon.
    Inl,
}

pub(super) struct BackendConfig {
    pub(super) backend: ServeBackend,
    pub(super) model: String,
    pub(super) dimensions: usize,
}

pub(crate) fn default_dense_backend() -> ServeBackend {
    if cfg!(windows) {
        ServeBackend::Colqwen
    } else {
        ServeBackend::Mlx
    }
}

pub(super) fn resolve_backend(
    backend: Option<ServeBackend>,
    model: Option<&str>,
    dimensions: Option<usize>,
) -> Result<BackendConfig> {
    let backend = backend.unwrap_or_else(|| {
        if model == Some(JINA_MLX_MODEL) {
            ServeBackend::JinaMlx
        } else if model.is_some_and(|value| value.contains("embeddinggemma")) {
            ServeBackend::Inl
        } else if model.is_some_and(|value| value.contains("v-splade") || value.contains("vsplade"))
        {
            ServeBackend::Vsplade
        } else if model.is_some_and(|value| value.contains("colqwen") && !value.contains("mlx")) {
            ServeBackend::Colqwen
        } else {
            default_dense_backend()
        }
    });
    let (default_model, default_dimensions) = match backend {
        ServeBackend::Mlx => (DEFAULT_MLX_MODEL, DEFAULT_MLX_DIMENSIONS),
        ServeBackend::JinaMlx => (JINA_MLX_MODEL, JINA_MLX_DIMENSIONS),
        ServeBackend::Colqwen => (DEFAULT_COLQWEN_MODEL, DEFAULT_COLQWEN_DIMENSIONS),
        ServeBackend::Inl => (DEFAULT_INL_MODEL, DEFAULT_INL_DIMENSIONS),
        ServeBackend::Vsplade => {
            #[cfg(windows)]
            {
                (WINDOWS_VSPLADE_MODEL, DEFAULT_VSPLADE_DIMENSIONS)
            }
            #[cfg(not(windows))]
            {
                (DEFAULT_VSPLADE_MODEL, DEFAULT_VSPLADE_DIMENSIONS)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::{
        APPLE_ONLY_BACKEND_ERROR, DEFAULT_COLQWEN_DIMENSIONS, DEFAULT_COLQWEN_MODEL,
        DEFAULT_INL_DIMENSIONS, DEFAULT_INL_MODEL, ServeBackend, default_dense_backend,
        resolve_backend,
    };

    #[test]
    fn apple_only_error_names_colqwen() {
        assert!(APPLE_ONLY_BACKEND_ERROR.contains("colqwen"));
        assert!(APPLE_ONLY_BACKEND_ERROR.contains("Apple Silicon"));
    }

    #[test]
    fn colqwen_model_selects_colqwen_backend() {
        let config = resolve_backend(None, Some("vidore/colqwen2-v1.0"), None).expect("backend");
        assert_eq!(config.backend, ServeBackend::Colqwen);
        assert_eq!(config.model, DEFAULT_COLQWEN_MODEL);
        assert_eq!(config.dimensions, DEFAULT_COLQWEN_DIMENSIONS);
    }

    #[test]
    fn inl_backend_defaults_to_embeddinggemma() {
        let config = resolve_backend(Some(ServeBackend::Inl), None, None).expect("backend");
        assert_eq!(config.model, DEFAULT_INL_MODEL);
        assert_eq!(config.dimensions, DEFAULT_INL_DIMENSIONS);
        assert_eq!(DEFAULT_INL_MODEL, "google/embeddinggemma-2");
        assert_eq!(DEFAULT_INL_DIMENSIONS, 768);
    }

    #[test]
    fn embeddinggemma_model_selects_inl_backend() {
        let config = resolve_backend(None, Some(DEFAULT_INL_MODEL), Some(DEFAULT_INL_DIMENSIONS))
            .expect("backend");
        assert_eq!(config.backend, ServeBackend::Inl);
    }

    #[test]
    fn default_dense_backend_is_platform_specific() {
        assert_eq!(
            default_dense_backend(),
            if cfg!(windows) {
                ServeBackend::Colqwen
            } else {
                ServeBackend::Mlx
            }
        );
    }
}
