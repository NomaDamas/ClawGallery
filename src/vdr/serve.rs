use super::backend::ServeBackend;
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::{
    env,
    net::{IpAddr, TcpListener},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const MLX_SERVER: &str = include_str!("../../scripts/mlx_embeddings_server.py");
const JINA_MLX_SERVER: &str = include_str!("../../scripts/jina_mlx_embeddings_server.py");
const MANAGED_STARTUP_TIMEOUT: Duration = Duration::from_secs(20 * 60);

impl ServeBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::JinaMlx => "jina-mlx",
        }
    }

    const fn script(self) -> &'static str {
        match self {
            Self::Mlx => MLX_SERVER,
            Self::JinaMlx => JINA_MLX_SERVER,
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
    run_python_server(&args)
}

pub(crate) struct ManagedServer {
    child: Child,
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
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_managed_server(args: &ServeArgs, announce: bool) -> Result<ManagedServer> {
    validate_bind_host(&args.host, args.allow_remote)?;
    start_python_server(args, announce)
}

fn run_python_server(args: &ServeArgs) -> Result<()> {
    let python = resolve_python(args.python.as_ref());
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
    Ok(ManagedServer { child, url })
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
        env::var_os("CLAWGALLERY_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python3"))
    })
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
