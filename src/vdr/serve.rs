use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use std::{
    env,
    net::IpAddr,
    path::PathBuf,
    process::{Command, Stdio},
};

const MLX_SERVER: &str = include_str!("../../scripts/mlx_embeddings_server.py");

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeBackend {
    Mlx,
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
    match args.backend {
        ServeBackend::Mlx => run_python_server(&args),
    }
}

fn run_python_server(args: &ServeArgs) -> Result<()> {
    let python = resolve_python(args.python.as_ref());
    let mut command = Command::new(&python);
    command
        .arg("-c")
        .arg(MLX_SERVER)
        .arg("--host")
        .arg(&args.host)
        .arg("--port")
        .arg(args.port.to_string())
        .arg("--model")
        .arg(&args.model)
        .arg("--dimensions")
        .arg(args.dimensions.to_string())
        .arg("--device")
        .arg(&args.device);
    if args.allow_remote {
        command.arg("--allow-remote");
    }
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to start Python interpreter {}", python.display()))?;
    if !status.success() {
        bail!("mlx embedding server exited with {status}");
    }
    Ok(())
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
        "refusing to bind unauthenticated mlx embedding server to non-loopback host {host:?} without --allow-remote"
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
