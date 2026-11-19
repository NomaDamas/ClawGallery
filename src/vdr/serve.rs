use super::backend::{APPLE_ONLY_BACKEND_ERROR, INL_MISSING_ERROR, ServeBackend};
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::json;
use std::{
    env,
    io::ErrorKind,
    net::{IpAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const MLX_SERVER: &str = include_str!("../../scripts/mlx_embeddings_server.py");
const JINA_MLX_SERVER: &str = include_str!("../../scripts/jina_mlx_embeddings_server.py");
const COLQWEN_SERVER: &str = include_str!("../../scripts/colqwen2_server.py");
const VSPLADE_SERVER: &str = include_str!("../../scripts/vsplade_server.py");
const VSPLADE_TORCH_SERVER: &str = include_str!("../../scripts/vsplade_torch_server.py");
const MANAGED_STARTUP_TIMEOUT: Duration = Duration::from_secs(20 * 60);
/// How long the `inl` daemon ClawGallery starts stays resident after the last
/// request. It is stopped with this process, so the timeout only bounds a
/// launcher that outlives its ClawGallery.
const INL_IDLE_TIMEOUT_SECS: u64 = 300;

impl ServeBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::JinaMlx => "jina-mlx",
            Self::Colqwen => "colqwen",
            Self::Vsplade => "vsplade",
            Self::Inl => "inl",
        }
    }

    fn script(self) -> &'static str {
        match self {
            Self::Mlx => MLX_SERVER,
            Self::JinaMlx => JINA_MLX_SERVER,
            Self::Colqwen => COLQWEN_SERVER,
            Self::Vsplade => {
                if cfg!(windows) {
                    VSPLADE_TORCH_SERVER
                } else {
                    VSPLADE_SERVER
                }
            }
            Self::Inl => unreachable!("the inl backend runs no Python script"),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ServeArgs {
    pub(crate) backend: ServeBackend,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) model: String,
    pub(crate) dimensions: usize,
    pub(crate) device: String,
    pub(crate) python: Option<PathBuf>,
    pub(crate) allow_remote: bool,
}

pub(crate) fn serve(args: ServeArgs) -> Result<()> {
    validate_bind_host(&args.host, args.allow_remote)?;
    if args.backend == ServeBackend::Inl {
        return run_inl_server(&args);
    }
    run_python_server(&args)
}

pub(crate) struct ManagedServer {
    /// `None` when an already running server was adopted instead of started.
    child: Option<Child>,
    url: String,
}

impl ManagedServer {
    pub(crate) fn start(args: &ServeArgs) -> Result<Self> {
        start_managed_server(args, true)
    }

    pub(crate) fn start_quiet(args: &ServeArgs) -> Result<Self> {
        start_managed_server(args, false)
    }

    pub(crate) fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for ManagedServer {
    fn drop(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn start_managed_server(args: &ServeArgs, announce: bool) -> Result<ManagedServer> {
    validate_bind_host(&args.host, args.allow_remote)?;
    if args.backend == ServeBackend::Inl {
        return start_inl_server(args, announce);
    }
    start_python_server(args, announce)
}

fn run_python_server(args: &ServeArgs) -> Result<()> {
    let python = resolve_python(args.python.as_ref());
    check_python_runtime(args.backend, &python)?;
    let mut command = python_command(args, &python, args.port);
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to start Python interpreter {}", python.display()))?;
    if !status.success() {
        bail!(
            "{} embedding server exited with {status}",
            args.backend.name()
        );
    }
    Ok(())
}

fn start_python_server(args: &ServeArgs, announce: bool) -> Result<ManagedServer> {
    let python = resolve_python(args.python.as_ref());
    check_python_runtime(args.backend, &python)?;
    let port = if args.port == 0 {
        choose_available_port(&args.host)?
    } else {
        args.port
    };
    let url = format!("http://{}:{port}", args.host);
    if announce {
        println!(
            "starting managed {} embedding server at {url}",
            args.backend.name()
        );
    }
    let stderr = if announce {
        Stdio::inherit()
    } else {
        Stdio::null()
    };
    let mut child = python_command(args, &python, port)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
        .with_context(|| format!("failed to start Python interpreter {}", python.display()))?;
    if let Err(err) = wait_until_embed_reachable(&mut child, &url, &args.model, args.dimensions) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(err);
    }
    Ok(ManagedServer {
        child: Some(child),
        url,
    })
}

/// Run `inl serve` for the requested multimodal model in the foreground,
/// letting it own its loopback address and lifetime.
fn run_inl_server(args: &ServeArgs) -> Result<()> {
    let status = inl_command(args, 0)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(inl_start_error)?;
    if !status.success() {
        bail!("inl embedding server exited with {status}");
    }
    Ok(())
}

/// Adopt a running `inl` multimodal server for `model`, or start one and wait
/// for it to answer `/health`. Only a server ClawGallery started is stopped on
/// drop; an adopted one is left alone.
fn start_inl_server(args: &ServeArgs, announce: bool) -> Result<ManagedServer> {
    if let Some(url) = discover_inl_server(&args.model) {
        if announce {
            println!(
                "adopted running inl multimodal server for {} at {url}",
                args.model
            );
        }
        return Ok(ManagedServer { child: None, url });
    }
    if announce {
        println!(
            "starting managed inl multimodal server for {} from PATH",
            args.model
        );
    }
    let stderr = if announce {
        Stdio::inherit()
    } else {
        Stdio::null()
    };
    let child = inl_command(args, INL_IDLE_TIMEOUT_SECS)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
        .map_err(inl_start_error)?;
    let mut child = child;
    wait_until_inl_discoverable(&mut child, &args.model)?;
    let url = discover_inl_server(&args.model).ok_or_else(|| {
        anyhow!(
            "inl reports serving {} but discovery found no live server",
            args.model
        )
    })?;
    Ok(ManagedServer {
        child: Some(child),
        url,
    })
}

fn inl_command(args: &ServeArgs, idle_timeout_secs: u64) -> Command {
    let mut command = Command::new("inl");
    command
        .arg("serve")
        .arg("--multimodal")
        .arg("--model")
        .arg(&args.model)
        .arg("--host")
        .arg(&args.host)
        .arg("--port")
        .arg(args.port.to_string())
        .arg("--device")
        .arg(&args.device)
        .arg("--idle-timeout")
        .arg(idle_timeout_secs.to_string());
    command
}

fn inl_start_error(error: std::io::Error) -> anyhow::Error {
    if error.kind() == ErrorKind::NotFound {
        anyhow!("{INL_MISSING_ERROR}")
    } else {
        anyhow::Error::new(error).context("failed to start `inl serve`")
    }
}

fn wait_until_inl_discoverable(child: &mut Child, model: &str) -> Result<()> {
    let deadline = Instant::now() + MANAGED_STARTUP_TIMEOUT;
    loop {
        if discover_inl_server(model).is_some() {
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            // A zero exit means `inl serve` adopted an existing server (which
            // may still be loading); a failure means it never got there.
            if !status.success() {
                bail!(
                    "inl serve for {model} exited before its server became reachable with {status}"
                );
            }
        }
        if Instant::now() >= deadline {
            bail!("inl multimodal server for {model} did not become discoverable");
        }
        thread::sleep(Duration::from_millis(250));
    }
}

/// The `inl` health subset discovery matches on.
#[derive(Debug, Deserialize)]
struct InlHealth {
    status: String,
    #[serde(default)]
    kind: String,
    model_id: String,
}

/// The `inl` serve root for this user, matching `inl_core::device::cache_root`
/// (`$HOME/.cache/im-not-llm/serve`, `%LOCALAPPDATA%\im-not-llm\serve`).
fn inl_serve_root() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let root = env::var_os("LOCALAPPDATA")?;
        Some(PathBuf::from(root).join("im-not-llm").join("serve"))
    }
    #[cfg(not(windows))]
    {
        let root = env::var_os("HOME")?;
        Some(
            PathBuf::from(root)
                .join(".cache")
                .join("im-not-llm")
                .join("serve"),
        )
    }
}

/// Find a live `inl` server under the serve root that serves `model` as a
/// multimodal engine, by reading each instance's `port`/`host` and matching
/// the answering `/health`. Returns its loopback URL.
fn discover_inl_server(model: &str) -> Option<String> {
    let root = inl_serve_root()?;
    let entries = std::fs::read_dir(&root).ok()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .no_proxy()
        .build()
        .ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let Some(port) =
            read_trimmed(&dir.join("port")).and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        let host = read_trimmed(&dir.join("host")).unwrap_or_else(|| "127.0.0.1".to_string());
        if !is_loopback_host(&host) {
            continue;
        }
        let url = format!("http://{host}:{port}");
        let Ok(response) = client.get(format!("{url}/health")).send() else {
            continue;
        };
        let Ok(health) = response.json::<InlHealth>() else {
            continue;
        };
        if health.status == "ok"
            && health.kind == "multimodal"
            && health.model_id.eq_ignore_ascii_case(model)
        {
            return Some(url);
        }
    }
    None
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn python_command(args: &ServeArgs, python: &PathBuf, port: u16) -> Command {
    let mut command = Command::new(python);
    command
        .arg("-c")
        .arg(args.backend.script())
        .arg("--host")
        .arg(&args.host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--model")
        .arg(&args.model)
        .arg("--dimensions")
        .arg(args.dimensions.to_string())
        .arg("--device")
        .arg(&args.device);
    if args.allow_remote {
        command.arg("--allow-remote");
    }
    command
}

fn choose_available_port(host: &str) -> Result<u16> {
    let listener = TcpListener::bind((host, 0))
        .with_context(|| format!("failed to choose local port for {host}"))?;
    Ok(listener.local_addr()?.port())
}

fn wait_until_embed_reachable(
    child: &mut Child,
    url: &str,
    model: &str,
    dimensions: usize,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let endpoint = format!("{}/embed", url.trim_end_matches('/'));
    let deadline = Instant::now() + MANAGED_STARTUP_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            bail!("managed embedding server exited before it became reachable with {status}");
        }
        if Instant::now() >= deadline {
            bail!("managed embedding server at {url} did not become reachable");
        }
        let response = client
            .post(&endpoint)
            .json(&json!({
                "model": model,
                "dimensions": dimensions,
                "inputs": [],
            }))
            .send();
        if matches!(response, Ok(response) if response.status().is_success()) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn resolve_python(explicit: Option<&PathBuf>) -> PathBuf {
    explicit.cloned().unwrap_or_else(|| {
        if let Some(python) = env::var_os("CLAWGALLERY_PYTHON") {
            return PathBuf::from(python);
        }
        if let Some(virtual_env) = env::var_os("VIRTUAL_ENV")
            && let Some(python) = virtual_env_python(PathBuf::from(virtual_env))
        {
            return python;
        }
        default_python()
    })
}

fn virtual_env_python(virtual_env: PathBuf) -> Option<PathBuf> {
    let python = if cfg!(windows) {
        virtual_env.join("Scripts").join("python.exe")
    } else {
        virtual_env.join("bin").join("python")
    };
    python.is_file().then_some(python)
}

#[cfg(windows)]
fn default_python() -> PathBuf {
    PathBuf::from("python")
}

#[cfg(not(windows))]
fn default_python() -> PathBuf {
    PathBuf::from("python3")
}

fn fake_env_enabled(name: &str) -> bool {
    env::var_os(name).is_some_and(|value| value == "1")
}

fn check_python_runtime(backend: ServeBackend, python: &PathBuf) -> Result<()> {
    let apple_only_fake = match backend {
        ServeBackend::Mlx => fake_env_enabled("CLAWGALLERY_VDR_MLX_FAKE"),
        ServeBackend::JinaMlx => fake_env_enabled("CLAWGALLERY_VDR_JINA_MLX_FAKE"),
        ServeBackend::Colqwen | ServeBackend::Vsplade => false,
        ServeBackend::Inl => unreachable!("the inl backend checks no Python runtime"),
    };
    if cfg!(windows)
        && matches!(backend, ServeBackend::Mlx | ServeBackend::JinaMlx)
        && !apple_only_fake
    {
        bail!("{APPLE_ONLY_BACKEND_ERROR}");
    }
    let (import, fake) = match backend {
        ServeBackend::Colqwen => (
            "import colpali_engine, torch, PIL",
            fake_env_enabled("CLAWGALLERY_VDR_COLQWEN_FAKE"),
        ),
        ServeBackend::Vsplade => (
            if cfg!(windows) {
                "import torch, transformers, PIL"
            } else {
                "import splade_mlx"
            },
            fake_env_enabled("CLAWGALLERY_VDR_VSPLADE_FAKE"),
        ),
        ServeBackend::Mlx | ServeBackend::JinaMlx => return Ok(()),
        ServeBackend::Inl => unreachable!("the inl backend checks no Python imports"),
    };
    if fake {
        return Ok(());
    }
    if backend == ServeBackend::Vsplade
        && cfg!(windows)
        && env::var_os("CLAWGALLERY_VSPLADE_REPO").is_none()
    {
        bail!(
            "V-SPLADE Windows runtime is not configured: set \
CLAWGALLERY_VSPLADE_REPO to a checkout of https://github.com/naver/v-splade \
before retrying with --python {} or CLAWGALLERY_PYTHON={}",
            python.display(),
            python.display()
        );
    }
    let output = Command::new(python)
        .args(["-c", import])
        .output()
        .with_context(|| {
            format!(
                "failed to inspect {} Python runtime {}",
                backend.name(),
                python.display()
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    if backend == ServeBackend::Colqwen {
        bail!(
            "ColQwen runtime is unavailable in {}: could not import colpali_engine, torch, and PIL. \
Install the Windows PyTorch runtime in this environment, then retry with \
--python {} or CLAWGALLERY_PYTHON={}. \
Example: {} -m pip install colpali-engine torch transformers pillow huggingface_hub",
            python.display(),
            python.display(),
            python.display(),
            python.display()
        );
    }
    bail!(
        "V-SPLADE runtime is unavailable in {}: the selected interpreter cannot \
load the supported backend dependencies. Install the Windows PyTorch runtime \
or SPLADE-mlx as appropriate, then retry with \
--python {} or CLAWGALLERY_PYTHON={}. \
Example: {} -m pip install torch transformers pillow",
        python.display(),
        python.display(),
        python.display(),
        python.display()
    );
}

fn validate_bind_host(host: &str, allow_remote: bool) -> Result<()> {
    if allow_remote || is_loopback_host(host) {
        return Ok(());
    }
    bail!(
        "refusing to bind unauthenticated embedding server to non-loopback host {host:?} without --allow-remote"
    );
}

fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    host.parse::<IpAddr>()
        .map(|address| address.is_loopback())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::{ServeBackend, check_python_runtime};
    use super::{default_python, virtual_env_python};
    use std::fs;
    #[cfg(windows)]
    use std::path::PathBuf;

    #[test]
    fn virtual_env_python_uses_platform_layout() {
        let temp = tempfile::tempdir().expect("tempdir");
        let python = if cfg!(windows) {
            temp.path().join("Scripts").join("python.exe")
        } else {
            temp.path().join("bin").join("python")
        };
        fs::create_dir_all(python.parent().expect("python parent")).expect("python directory");
        fs::write(&python, b"").expect("python file");

        assert_eq!(virtual_env_python(temp.path().to_path_buf()), Some(python));
    }

    #[test]
    fn default_python_uses_platform_name() {
        assert_eq!(
            default_python().file_name().expect("python filename"),
            if cfg!(windows) { "python" } else { "python3" }
        );
    }

    #[cfg(windows)]
    #[test]
    fn colqwen_runtime_check_does_not_require_vsplade_repo() {
        let python = PathBuf::from("python");
        let err = check_python_runtime(ServeBackend::Colqwen, &python)
            .expect_err("colqwen check should fail on missing colpali imports, not vsplade repo");
        let msg = format!("{err:#}");
        assert!(
            !msg.contains("CLAWGALLERY_VSPLADE_REPO"),
            "colqwen must not demand V-SPLADE_REPO, got: {msg}"
        );
        assert!(
            msg.contains("ColQwen") || msg.contains("colpali"),
            "expected ColQwen import diagnostic, got: {msg}"
        );
    }
}
