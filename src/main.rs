use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use nucleo_matcher::{
    Config, Matcher, Utf32Str,
    pattern::{CaseMatching, Normalization, Pattern},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    thread,
    time::Duration,
};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;
use walkdir::WalkDir;

mod vdr;

const APP_DIR_NAME: &str = "clawgallery";
const DEFAULT_MODEL: &str = "gpt-4.1-mini";
const DEFAULT_FILENAME_LIMIT_BYTES: usize = 240;
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "avif", "gif"];

#[derive(Debug, Parser)]
#[command(
    name = "clawgallery",
    version,
    about = "Agent-native screenshot gallery manager"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create the config directory and default config file.
    Init,
    /// Manage tracked folders.
    Folder {
        #[command(subcommand)]
        command: FolderCommand,
    },
    /// Bootstrap existing images from registered folders or a one-off path.
    Bootstrap(IngestArgs),
    /// Poll for new images once or continuously.
    Poll(PollArgs),
    /// Caption/title images through the configured visual model.
    Caption(CaptionArgs),
    /// Safely rename images from generated titles/captions.
    Rename(RenameArgs),
    /// Search local JSONL metadata by keyword.
    Search(SearchArgs),
    /// Manage local visual document retrieval embeddings.
    Vdr(vdr::VdrArgs),
    /// Show state and configuration summary.
    Status,
    /// Print or locate the bundled Vercel Agent Skill.
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },
}

#[derive(Debug, Subcommand)]
enum FolderCommand {
    /// Add a folder to the tracking registry.
    Add(FolderAddArgs),
    /// Remove/deactivate a folder by id or path.
    Remove(FolderRemoveArgs),
    /// List active tracked folders.
    List,
}

#[derive(Debug, Args)]
struct FolderAddArgs {
    path: PathBuf,
    #[arg(long, default_value_t = true)]
    recursive: bool,
}

#[derive(Debug, Args)]
struct FolderRemoveArgs {
    id_or_path: String,
}

#[derive(Debug, Args, Clone)]
struct IngestArgs {
    /// Limit ingestion to a registered folder id.
    #[arg(long)]
    folder: Option<String>,
    /// Bootstrap a one-off path without registering it first.
    #[arg(long)]
    path: Option<PathBuf>,
    /// Mark previously ingested images that are no longer on disk as inactive.
    #[arg(long)]
    prune: bool,
}

#[derive(Debug, Args)]
struct PollArgs {
    #[command(flatten)]
    ingest: IngestArgs,
    /// Run a single scan and exit.
    #[arg(long)]
    once: bool,
    /// Poll interval in seconds for continuous mode.
    #[arg(long, default_value_t = 10)]
    interval: u64,
}

#[derive(Debug, Args)]
struct CaptionArgs {
    /// Caption only records that do not already have a caption.
    #[arg(long, default_value_t = true)]
    missing: bool,
    /// Caption one explicit image path, recording it if needed.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Print target files without calling the model.
    #[arg(long)]
    dry_run: bool,
    /// Override model for this run.
    #[arg(long)]
    model: Option<String>,
    /// Override provider for this run (openai-compatible or gemini).
    #[arg(long)]
    provider: Option<String>,
}

#[derive(Debug, Args)]
struct RenameArgs {
    /// Apply renames. Without this flag, rename is a dry-run.
    #[arg(long)]
    apply: bool,
    /// Explicitly request dry-run mode. This is also the default when --apply is absent.
    #[arg(long)]
    dry_run: bool,
    /// Rename one explicit image path only.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Filename style.
    #[arg(long, value_enum, default_value_t = RenameStyle::DateTitle)]
    style: RenameStyle,
    /// Rename even when the current filename looks human-meaningful.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RenameStyle {
    Title,
    Caption,
    DateTitle,
}

#[derive(Debug, Args)]
struct SearchArgs {
    /// Search backend to use.
    #[arg(long, value_enum, default_value_t = SearchMode::Keyword)]
    mode: SearchMode,
    /// Query terms (combined as fzf-style query). See README for syntax.
    keywords: Vec<String>,
    /// Maximum number of results to print.
    #[arg(long, default_value_t = 20)]
    limit: usize,
    /// Emit one JSON object per result (JSONL).
    #[arg(long)]
    json: bool,
    /// Force case-sensitive matching (overrides smart-case).
    #[arg(long)]
    case_sensitive: bool,
    /// Disable fuzzy matching and Levenshtein fallback. Old-style substring AND.
    #[arg(long)]
    no_fuzzy: bool,
    /// Override the local VDR embedding server URL.
    #[arg(long)]
    embedding_url: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchMode {
    Keyword,
    Embedding,
}

#[derive(Debug, Subcommand)]
enum SkillCommand {
    /// Print the path to the bundled skill.
    Path,
    /// Print the bundled skill instructions.
    Print,
}

#[derive(Debug, Clone)]
pub(crate) struct AppPaths {
    pub(crate) root: PathBuf,
    config: PathBuf,
    folders: PathBuf,
    images: PathBuf,
    captions: PathBuf,
    renames: PathBuf,
    errors: PathBuf,
    pub(crate) vdr_db: PathBuf,
}

impl AppPaths {
    fn resolve() -> Result<Self> {
        let root = if let Ok(path) = env::var("CLAWGALLERY_CONFIG_DIR") {
            PathBuf::from(path)
        } else {
            dirs::config_dir()
                .ok_or_else(|| anyhow!("could not resolve user config directory"))?
                .join(APP_DIR_NAME)
        };
        Ok(Self {
            config: root.join("config.json"),
            folders: root.join("folders.jsonl"),
            images: root.join("images.jsonl"),
            captions: root.join("captions.jsonl"),
            renames: root.join("renames.jsonl"),
            errors: root.join("errors.jsonl"),
            vdr_db: root.join("vdr.sqlite3"),
            root,
        })
    }

    pub(crate) fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AppConfig {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_provider_str")]
    provider: String,
    #[serde(default = "default_filename_limit")]
    filename_limit_bytes: usize,
}

fn default_model() -> String {
    env::var("CLAWGALLERY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

fn default_provider_str() -> String {
    "openai-compatible".to_string()
}

fn default_filename_limit() -> usize {
    DEFAULT_FILENAME_LIMIT_BYTES
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            provider: default_provider_str(),
            filename_limit_bytes: default_filename_limit(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FolderRecord {
    id: String,
    path: PathBuf,
    recursive: bool,
    active: bool,
    created_at: DateTime<Utc>,
    removed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ImageRecord {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    original_path: PathBuf,
    pub(crate) sha256: String,
    size: u64,
    modified_at: Option<DateTime<Utc>>,
    pub(crate) discovered_at: DateTime<Utc>,
    extension: String,
    #[serde(default = "default_active")]
    pub(crate) active: bool,
    #[serde(default)]
    removed_at: Option<DateTime<Utc>>,
}

fn default_active() -> bool {
    true
}

impl ImageRecord {
    pub(crate) fn has_same_file_fingerprint(&self, other: &Self) -> bool {
        self.sha256 == other.sha256
            && self.size == other.size
            && self.modified_at == other.modified_at
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct CaptionRecord {
    pub(crate) image_id: String,
    pub(crate) path: PathBuf,
    pub(crate) title: String,
    pub(crate) description: String,
    model: String,
    provider: String,
    created_at: DateTime<Utc>,
    #[serde(default)]
    filename_meaningful: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct RenameRecord {
    image_id: Option<String>,
    from: PathBuf,
    to: PathBuf,
    applied: bool,
    reason: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ErrorRecord {
    context: String,
    message: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug)]
struct CaptionOutput {
    title: String,
    description: String,
}

enum Provider {
    OpenAiCompat(OpenAiCompatProvider),
    Gemini(GeminiProvider),
}

impl Provider {
    fn caption_image(&self, path: &Path) -> Result<CaptionOutput> {
        match self {
            Provider::OpenAiCompat(p) => p.caption_image(path),
            Provider::Gemini(p) => p.caption_image(path),
        }
    }

    fn classify_stem(&self, stem: &str) -> Result<bool> {
        match self {
            Provider::OpenAiCompat(p) => p.classify_stem(stem),
            Provider::Gemini(p) => p.classify_stem(stem),
        }
    }
}

fn build_provider(
    config: &AppConfig,
    cli_provider: Option<String>,
    cli_model: Option<String>,
) -> Result<Provider> {
    let provider_name = cli_provider.unwrap_or_else(|| config.provider.clone());
    let model = cli_model.unwrap_or_else(|| config.model.clone());
    match provider_name.as_str() {
        "gemini" => {
            let api_key = env::var("GEMINI_API_KEY")
                .with_context(|| "missing GEMINI_API_KEY environment variable")?;
            Ok(Provider::Gemini(GeminiProvider::new(api_key, model)))
        }
        _ => {
            let auth = Auth::discover()?;
            let base_url = env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            Ok(Provider::OpenAiCompat(OpenAiCompatProvider::new(
                auth, model, base_url,
            )))
        }
    }
}

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("Error: {}", mask_api_keys(&format!("{err:#}")));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::resolve()?;
    match cli.command {
        Command::Init => cmd_init(&paths),
        Command::Folder { command } => match command {
            FolderCommand::Add(args) => cmd_folder_add(&paths, args),
            FolderCommand::Remove(args) => cmd_folder_remove(&paths, args),
            FolderCommand::List => cmd_folder_list(&paths),
        },
        Command::Bootstrap(args) => cmd_bootstrap(&paths, &args).map(|stats| {
            println!("ingested {} new image(s)", stats.ingested);
            if args.prune {
                println!("pruned {} missing image(s)", stats.pruned);
            }
        }),
        Command::Poll(args) => cmd_poll(&paths, args),
        Command::Caption(args) => cmd_caption(&paths, args),
        Command::Rename(args) => cmd_rename(&paths, args),
        Command::Search(args) => cmd_search(&paths, args),
        Command::Vdr(args) => vdr::cmd_vdr(&paths, args),
        Command::Status => cmd_status(&paths),
        Command::Skill { command } => match command {
            SkillCommand::Path => cmd_skill_path(&paths),
            SkillCommand::Print => {
                print!("{}", include_str!("../skills/clawgallery/SKILL.md"));
                Ok(())
            }
        },
    }
}

fn cmd_init(paths: &AppPaths) -> Result<()> {
    paths.ensure()?;
    if !paths.config.exists() {
        write_json_pretty(&paths.config, &AppConfig::default())?;
    }
    for path in [
        &paths.folders,
        &paths.images,
        &paths.captions,
        &paths.renames,
        &paths.errors,
    ] {
        if !path.exists() {
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        }
    }
    println!("initialized {}", paths.root.display());
    Ok(())
}

fn cmd_folder_add(paths: &AppPaths, args: FolderAddArgs) -> Result<()> {
    cmd_init(paths)?;
    let canonical = canonicalize_existing_dir(&args.path)?;
    if active_folders(paths)?
        .iter()
        .any(|folder| folder.path == canonical)
    {
        println!("folder already tracked: {}", canonical.display());
        return Ok(());
    }
    let record = FolderRecord {
        id: Uuid::new_v4().to_string(),
        path: canonical,
        recursive: args.recursive,
        active: true,
        created_at: Utc::now(),
        removed_at: None,
    };
    append_jsonl(&paths.folders, &record)?;
    println!("added {} {}", record.id, record.path.display());
    Ok(())
}

fn cmd_folder_remove(paths: &AppPaths, args: FolderRemoveArgs) -> Result<()> {
    paths.ensure()?;
    let mut matched = false;
    for folder in active_folders(paths)? {
        if folder.id == args.id_or_path || folder.path.to_string_lossy() == args.id_or_path {
            let mut removed = folder.clone();
            removed.active = false;
            removed.removed_at = Some(Utc::now());
            append_jsonl(&paths.folders, &removed)?;
            println!("removed {} {}", removed.id, removed.path.display());
            matched = true;
        }
    }
    if !matched {
        bail!("no active folder matched '{}'", args.id_or_path);
    }
    Ok(())
}

fn cmd_folder_list(paths: &AppPaths) -> Result<()> {
    for folder in active_folders(paths)? {
        println!(
            "{}\t{}\trecursive={}",
            folder.id,
            folder.path.display(),
            folder.recursive
        );
    }
    Ok(())
}

struct BootstrapStats {
    ingested: usize,
    pruned: usize,
}

fn cmd_bootstrap(paths: &AppPaths, args: &IngestArgs) -> Result<BootstrapStats> {
    paths.ensure()?;
    if !paths.config.exists() {
        write_json_pretty(&paths.config, &AppConfig::default())?;
    }
    let existing = latest_images_by_path(paths)?;
    let mut seen_paths: HashSet<PathBuf> = HashSet::new();
    let mut ingested = 0;
    for image_path in candidate_image_paths(paths, args)? {
        let canonical = fs::canonicalize(&image_path).unwrap_or(image_path.clone());
        if seen_paths.contains(&canonical) {
            continue;
        }
        seen_paths.insert(canonical.clone());
        match build_image_record(&canonical) {
            Ok(record) => {
                if let Some(previous) = existing.get(&canonical)
                    && previous.has_same_file_fingerprint(&record)
                {
                    continue;
                }
                append_jsonl(&paths.images, &record)?;
                ingested += 1;
            }
            Err(err) => log_error(paths, "ingest", err),
        }
    }
    let pruned = if args.prune {
        prune_missing(paths, &existing)?
    } else {
        0
    };
    Ok(BootstrapStats { ingested, pruned })
}

fn prune_missing(paths: &AppPaths, active_images: &HashMap<PathBuf, ImageRecord>) -> Result<usize> {
    let mut pruned = 0;
    let now = Utc::now();
    for image in active_images.values() {
        if image.path.exists() {
            continue;
        }
        let mut deactivated = image.clone();
        deactivated.active = false;
        deactivated.removed_at = Some(now);
        append_jsonl(&paths.images, &deactivated)?;
        pruned += 1;
    }
    Ok(pruned)
}

fn cmd_poll(paths: &AppPaths, args: PollArgs) -> Result<()> {
    loop {
        let stats = cmd_bootstrap(paths, &args.ingest)?;
        println!(
            "{}: ingested {} new image(s)",
            Utc::now().to_rfc3339(),
            stats.ingested
        );
        if args.ingest.prune {
            println!("pruned {} missing image(s)", stats.pruned);
        }
        if args.once {
            break;
        }
        thread::sleep(Duration::from_secs(args.interval.max(1)));
    }
    Ok(())
}

fn resolve_provider(cli_provider: Option<&str>, config_provider: &str) -> String {
    cli_provider
        .map(str::to_string)
        .unwrap_or_else(|| config_provider.to_string())
}

fn resolve_model(cli_model: Option<&str>, config_model: &str) -> String {
    cli_model
        .map(str::to_string)
        .unwrap_or_else(|| config_model.to_string())
}

fn cmd_caption(paths: &AppPaths, args: CaptionArgs) -> Result<()> {
    paths.ensure()?;
    let config = read_config(paths)?;
    let effective_provider = resolve_provider(args.provider.as_deref(), &config.provider);
    let effective_model = resolve_model(args.model.as_deref(), &config.model);
    let mut images = latest_images(paths)?;
    if let Some(file) = args.file {
        let canonical = fs::canonicalize(&file).unwrap_or(file);
        if !images.iter().any(|image| image.path == canonical) {
            images.push(build_image_record(&canonical)?);
        }
        images.retain(|image| image.path == canonical);
    }
    if args.missing {
        let captioned: HashSet<String> = latest_captions(paths)?
            .into_iter()
            .map(|cap| cap.image_id)
            .collect();
        images.retain(|image| !captioned.contains(&image.id));
    }
    if images.is_empty() {
        println!("no images need captions");
        return Ok(());
    }
    if args.dry_run {
        for image in images {
            println!("would caption {}", image.path.display());
        }
        return Ok(());
    }
    let provider = build_provider(&config, args.provider, args.model.clone())?;
    for image in images {
        let stem = image.path.file_stem().and_then(OsStr::to_str).unwrap_or("");
        let resolved_meaningful = match classify_filename(stem) {
            NameClassification::Generic => Some(false),
            NameClassification::NeedsModel => match provider.classify_stem(stem) {
                Ok(b) => Some(b),
                Err(err) => {
                    log_error(paths, "classify_stem", err);
                    None
                }
            },
        };
        match provider.caption_image(&image.path) {
            Ok(output) => {
                let record = CaptionRecord {
                    image_id: image.id.clone(),
                    path: image.path.clone(),
                    title: output.title,
                    description: output.description,
                    model: effective_model.clone(),
                    provider: effective_provider.clone(),
                    created_at: Utc::now(),
                    filename_meaningful: resolved_meaningful,
                };
                append_jsonl(&paths.captions, &record)?;
                println!("captioned {}", image.path.display());
            }
            Err(err) => {
                log_error(paths, "caption", err);
            }
        }
    }
    Ok(())
}

fn cmd_rename(paths: &AppPaths, args: RenameArgs) -> Result<()> {
    paths.ensure()?;
    if args.apply && args.dry_run {
        bail!("--apply and --dry-run cannot be used together");
    }
    let config = read_config(paths)?;
    let captions = latest_captions_by_path(paths)?;
    let mut images = latest_images(paths)?;
    let explicit_file = args.file.is_some();
    if let Some(file) = args.file {
        let canonical = fs::canonicalize(&file).unwrap_or(file);
        images.retain(|image| image.path == canonical);
        if images.is_empty() && canonical.exists() {
            images.push(build_image_record(&canonical)?);
        }
    }
    let mut renamed = 0_usize;
    let mut skipped = 0_usize;
    let mut failed = 0_usize;
    for image in images {
        let Some(caption) = captions.get(&image.path) else {
            continue;
        };
        let stem = image.path.file_stem().and_then(OsStr::to_str).unwrap_or("");
        let decision =
            rename_decision(stem, caption.filename_meaningful, explicit_file, args.force);
        if decision == RenameDecision::Skip {
            skipped += 1;
            if !args.apply {
                println!("would skip (meaningful filename) {}", image.path.display());
            }
            continue;
        }
        if !image.path.exists() {
            println!("would skip (missing source) {}", image.path.display());
            if args.apply {
                deactivate_image_record(paths, &image)?;
            }
            continue;
        }
        let title = match args.style {
            RenameStyle::Title => caption.title.clone(),
            RenameStyle::Caption => caption.description.clone(),
            RenameStyle::DateTitle => format!(
                "{} {}",
                image.discovered_at.format("%Y-%m-%d"),
                caption.title
            ),
        };
        let target = match rename_candidate(&image.path, &title, config.filename_limit_bytes) {
            Ok(t) => t,
            Err(err) => {
                log_error(paths, "rename", err);
                failed += 1;
                continue;
            }
        };
        let record = RenameRecord {
            image_id: Some(image.id.clone()),
            from: image.path.clone(),
            to: target.clone(),
            applied: args.apply,
            reason: format!("style={:?}", args.style),
            created_at: Utc::now(),
        };
        if args.apply {
            if let Err(err) = rename_no_clobber(&image.path, &target).with_context(|| {
                format!(
                    "failed to rename {} to {}",
                    image.path.display(),
                    target.display()
                )
            }) {
                log_error(paths, "rename", err);
                failed += 1;
                continue;
            }
            if let Err(err) = append_jsonl(&paths.renames, &record) {
                eprintln!(
                    "warning: renamed on disk but state update failed for {}; run bootstrap to reconcile",
                    target.display()
                );
                log_error(paths, "rename", err);
                failed += 1;
                continue;
            }
            let mut updated = image.clone();
            updated.path = fs::canonicalize(&target).unwrap_or(target.clone());
            if let Err(err) = append_jsonl(&paths.images, &updated) {
                eprintln!(
                    "warning: renamed on disk but state update failed for {}; run bootstrap to reconcile",
                    target.display()
                );
                log_error(paths, "rename", err);
                failed += 1;
                continue;
            }
            println!("renamed {} -> {}", image.path.display(), target.display());
        } else {
            if let Err(err) = append_jsonl(&paths.renames, &record) {
                log_error(paths, "rename", err);
                failed += 1;
                continue;
            }
            println!("dry-run {} -> {}", image.path.display(), target.display());
        }
        renamed += 1;
    }
    if args.apply {
        println!(
            "renamed {renamed}, skipped {skipped} meaningful-looking name(s), failed {failed}"
        );
    } else if skipped > 0 {
        println!("(would skip {skipped} meaningful-looking name(s); use --force to override)");
    }
    Ok(())
}

fn rename_no_clobber(source: &Path, target: &Path) -> io::Result<()> {
    match fs::hard_link(source, target) {
        Ok(()) => fs::remove_file(source),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("refusing to overwrite existing file {}", target.display()),
        )),
        Err(err) if err.kind() == io::ErrorKind::CrossesDevices => {
            if target.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("refusing to overwrite existing file {}", target.display()),
                ));
            }
            fs::rename(source, target)
        }
        Err(err) => Err(err),
    }
}

fn deactivate_image_record(paths: &AppPaths, image: &ImageRecord) -> Result<()> {
    let mut deactivated = image.clone();
    deactivated.active = false;
    deactivated.removed_at = Some(Utc::now());
    append_jsonl(&paths.images, &deactivated)
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    path_raw: PathBuf,
    path_nfc: String,
    title_nfc: String,
    description_nfc: String,
    discovered_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct SearchHit {
    candidate_idx: usize,
    score: f64,
    pattern_score: u32,
    matched_field: MatchedField,
    matched_atoms: Vec<String>,
    source: HitSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum HitSource {
    Fuzzy,
    Levenshtein,
    NoFuzzy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MatchedField {
    Title,
    Description,
    Path,
    Multiple,
}

impl MatchedField {
    fn as_str(self) -> &'static str {
        match self {
            MatchedField::Title => "title",
            MatchedField::Description => "description",
            MatchedField::Path => "path",
            MatchedField::Multiple => "multiple",
        }
    }
}

fn nfc(s: &str) -> String {
    s.nfc().collect()
}

fn case_fold_for_search(s: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        s.to_string()
    } else {
        s.to_lowercase()
    }
}

fn smart_case_sensitive(query: &str, case_sensitive: bool) -> bool {
    case_sensitive || query.chars().any(|ch| ch.is_uppercase())
}

fn extract_atom_payloads(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter_map(|raw| {
            let mut atom = raw;
            if atom.starts_with('!') {
                return None;
            }
            if let Some(stripped) = atom.strip_prefix(r"\!") {
                atom = stripped;
            }
            if let Some(stripped) = atom.strip_prefix('^') {
                atom = stripped;
            }
            if let Some(stripped) = atom.strip_prefix('"') {
                atom = stripped;
            }
            if let Some(stripped) = atom.strip_prefix('\'') {
                atom = stripped;
            }
            if let Some(stripped) = atom.strip_suffix('$') {
                atom = stripped;
            }
            let payload = atom
                .replace(r"\ ", " ")
                .replace(r"\^", "^")
                .replace(r"\'", "'")
                .replace(r"\$", "$")
                .replace(r"\!", "!");
            (!payload.trim().is_empty()).then(|| payload.trim().to_string())
        })
        .collect()
}

fn build_candidates(
    images: Vec<ImageRecord>,
    captions: &HashMap<PathBuf, CaptionRecord>,
) -> Vec<SearchCandidate> {
    images
        .into_iter()
        .map(|image| {
            let cap = captions.get(&image.path);
            SearchCandidate {
                path_nfc: nfc(&image.path.display().to_string()),
                title_nfc: nfc(cap.map(|c| c.title.as_str()).unwrap_or("<missing>")),
                description_nfc: nfc(cap.map(|c| c.description.as_str()).unwrap_or("<missing>")),
                discovered_at: image.discovered_at,
                path_raw: image.path,
            }
        })
        .collect()
}

fn score_field(
    pattern: &Pattern,
    haystack: &str,
    weight: f64,
    matcher: &mut Matcher,
    buf: &mut Vec<char>,
) -> Option<(u32, f64)> {
    pattern
        .score(Utf32Str::new(haystack, buf), matcher)
        .map(|score| (score, f64::from(score) * weight))
}

fn best_matched_field(scores: &[(MatchedField, u32, f64)]) -> MatchedField {
    if scores.len() > 1 {
        let best = scores
            .iter()
            .map(|(_, _, weighted)| *weighted)
            .fold(0.0, f64::max);
        let tied = scores
            .iter()
            .filter(|(_, _, weighted)| (*weighted - best).abs() < f64::EPSILON)
            .count();
        if tied > 1 {
            return MatchedField::Multiple;
        }
    }
    scores
        .iter()
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(field, _, _)| *field)
        .unwrap_or(MatchedField::Multiple)
}

fn search_pattern_pass(
    candidates: &[SearchCandidate],
    query: &str,
    matcher: &mut Matcher,
    case_sensitive: bool,
) -> Vec<SearchHit> {
    let case = if case_sensitive {
        CaseMatching::Respect
    } else {
        CaseMatching::Smart
    };
    let pattern = Pattern::parse(query, case, Normalization::Smart);
    let atoms = extract_atom_payloads(query);
    let smart_sensitive = smart_case_sensitive(query, case_sensitive);
    let atoms_cmp: Vec<String> = atoms
        .iter()
        .map(|atom| case_fold_for_search(atom, smart_sensitive))
        .collect();
    let mut buf = Vec::new();
    let mut hits = Vec::new();

    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        let mut scores = Vec::new();
        if let Some((raw, weighted)) =
            score_field(&pattern, &candidate.title_nfc, 3.0, matcher, &mut buf)
        {
            scores.push((MatchedField::Title, raw, weighted));
        }
        if let Some((raw, weighted)) =
            score_field(&pattern, &candidate.description_nfc, 1.5, matcher, &mut buf)
        {
            scores.push((MatchedField::Description, raw, weighted));
        }
        if let Some((raw, weighted)) =
            score_field(&pattern, &candidate.path_nfc, 1.0, matcher, &mut buf)
        {
            scores.push((MatchedField::Path, raw, weighted));
        }
        if scores.is_empty() {
            continue;
        }

        let combined_cmp = case_fold_for_search(
            &format!(
                "{} {} {}",
                candidate.title_nfc, candidate.description_nfc, candidate.path_nfc
            ),
            smart_sensitive,
        );
        if query.split_whitespace().any(|atom| {
            atom.strip_prefix('!')
                .filter(|payload| !payload.is_empty())
                .map(|payload| {
                    combined_cmp.contains(&case_fold_for_search(payload, smart_sensitive))
                })
                .unwrap_or(false)
        }) {
            continue;
        }

        let matched_field = best_matched_field(&scores);
        let (pattern_score, weighted_score) = scores
            .iter()
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(_, raw, weighted)| (*raw, *weighted))
            .unwrap_or((0, 0.0));
        let title_cmp = case_fold_for_search(&candidate.title_nfc, smart_sensitive);
        let desc_cmp = case_fold_for_search(&candidate.description_nfc, smart_sensitive);
        let path_cmp = case_fold_for_search(&candidate.path_nfc, smart_sensitive);
        let stem_cmp = candidate
            .path_raw
            .file_stem()
            .and_then(OsStr::to_str)
            .map(nfc)
            .map(|stem| case_fold_for_search(&stem, smart_sensitive))
            .unwrap_or_default();
        let matched_atoms: Vec<String> = atoms
            .iter()
            .zip(atoms_cmp.iter())
            .filter(|(_, atom)| {
                title_cmp.contains(atom.as_str())
                    || desc_cmp.contains(atom.as_str())
                    || path_cmp.contains(atom.as_str())
            })
            .map(|(atom, _)| atom.clone())
            .collect();
        if !atoms.is_empty() && matched_atoms.is_empty() {
            continue;
        }
        let mut bonus = 0.0;
        if atoms_cmp
            .iter()
            .any(|atom| title_cmp.contains(atom.as_str()))
        {
            bonus += 50.0;
        }
        if atoms_cmp
            .first()
            .is_some_and(|atom| title_cmp.starts_with(atom.as_str()))
        {
            bonus += 30.0;
        }
        if atoms_cmp
            .iter()
            .any(|atom| stem_cmp.contains(atom.as_str()))
        {
            bonus += 10.0;
        }

        hits.push(SearchHit {
            candidate_idx,
            score: weighted_score + bonus,
            pattern_score,
            matched_field,
            matched_atoms,
            source: HitSource::Fuzzy,
        });
    }

    hits
}

fn fallback_threshold(payload: &str) -> Option<f64> {
    match payload.chars().count() {
        0..=2 => None,
        3..=8 => Some(0.75),
        _ => Some(0.70),
    }
}

fn best_window_similarity(atom: &str, text: &str, case_sensitive: bool) -> f64 {
    let atom_cmp = case_fold_for_search(atom, case_sensitive);
    let atom_token_count = atom_cmp.split_whitespace().count().max(1);
    let text_cmp = case_fold_for_search(text, case_sensitive);
    let tokens: Vec<&str> = text_cmp.split_whitespace().collect();
    if tokens.is_empty() {
        return 0.0;
    }
    let mut best = 0.0;
    for size in [
        atom_token_count.saturating_sub(1),
        atom_token_count,
        atom_token_count + 1,
    ]
    .into_iter()
    .filter(|size| *size > 0)
    {
        if size > tokens.len() {
            continue;
        }
        for window in tokens.windows(size) {
            let joined = window.join(" ");
            let similarity = strsim::normalized_damerau_levenshtein(&atom_cmp, &joined);
            if similarity > best {
                best = similarity;
            }
        }
    }
    best
}

fn search_levenshtein_fallback(
    candidates: &[SearchCandidate],
    query_atoms: &[String],
    case_sensitive: bool,
) -> Vec<SearchHit> {
    let fallback_atoms: Vec<&String> = query_atoms
        .iter()
        .filter(|atom| fallback_threshold(atom).is_some())
        .collect();
    if fallback_atoms.is_empty() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        let mut similarities = Vec::new();
        let mut matched_atoms = Vec::new();
        for atom in &fallback_atoms {
            let threshold = fallback_threshold(atom).unwrap_or(1.0);
            let title_similarity =
                best_window_similarity(atom, &candidate.title_nfc, case_sensitive);
            let desc_similarity =
                best_window_similarity(atom, &candidate.description_nfc, case_sensitive);
            let similarity = title_similarity.max(desc_similarity);
            if similarity < threshold {
                similarities.clear();
                break;
            }
            similarities.push(similarity);
            matched_atoms.push((*atom).clone());
        }
        if similarities.len() == fallback_atoms.len() {
            let avg = similarities.iter().sum::<f64>() / similarities.len() as f64;
            hits.push(SearchHit {
                candidate_idx,
                score: (avg * 1000.0).floor(),
                pattern_score: 0,
                matched_field: MatchedField::Multiple,
                matched_atoms,
                source: HitSource::Levenshtein,
            });
        }
    }
    hits
}

fn print_old_search_result(candidate: &SearchCandidate) {
    println!(
        "{}\n  title: {}\n  caption: {}",
        candidate.path_raw.display(),
        candidate.title_nfc,
        candidate.description_nfc
    );
}

fn print_text_result(hit: &SearchHit, candidate: &SearchCandidate) {
    println!(
        "{}\n  title: {}\n  caption: {}\n  score: {:.1}\n  matches: {} ({})",
        candidate.path_raw.display(),
        candidate.title_nfc,
        candidate.description_nfc,
        hit.score,
        hit.matched_field.as_str(),
        hit.matched_atoms.join(", ")
    );
}

fn print_json_result(hit: &SearchHit, candidate: &SearchCandidate) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(&json!({
            "path": candidate.path_raw,
            "title": candidate.title_nfc,
            "description": candidate.description_nfc,
            "score": hit.score,
            "matched_field": hit.matched_field,
            "matched_atoms": hit.matched_atoms,
            "source": hit.source,
        }))?
    );
    Ok(())
}

fn cmd_search(paths: &AppPaths, args: SearchArgs) -> Result<()> {
    if args.keywords.is_empty() {
        bail!("provide at least one keyword");
    }
    let query = nfc(&args.keywords.join(" "));
    let captions = latest_captions_by_path(paths)?;
    let images = latest_images(paths)?;
    if matches!(args.mode, SearchMode::Embedding) {
        return vdr::cmd_embedding_search(
            paths,
            &query,
            args.limit,
            args.json,
            args.embedding_url.as_deref(),
            images,
            captions,
        );
    }
    let candidates = build_candidates(images, &captions);

    if args.no_fuzzy {
        let needle: Vec<String> = args.keywords.iter().map(|k| k.to_lowercase()).collect();
        let mut printed = 0;
        for candidate in &candidates {
            let haystack = format!(
                "{} {} {}",
                candidate.path_raw.display(),
                candidate.title_nfc,
                candidate.description_nfc
            )
            .to_lowercase();
            if !needle.iter().all(|keyword| haystack.contains(keyword)) {
                continue;
            }
            let hit = SearchHit {
                candidate_idx: 0,
                score: 0.0,
                pattern_score: 0,
                matched_field: MatchedField::Multiple,
                matched_atoms: Vec::new(),
                source: HitSource::NoFuzzy,
            };
            let _ = hit;
            print_old_search_result(candidate);
            printed += 1;
            if printed >= args.limit {
                break;
            }
        }
        return Ok(());
    }

    let mut matcher = Matcher::new(Config::DEFAULT);
    let mut hits = search_pattern_pass(&candidates, &query, &mut matcher, args.case_sensitive);
    let query_atoms = extract_atom_payloads(&query);
    let mut used_fallback = false;
    if hits.is_empty()
        && !query_atoms.is_empty()
        && !smart_case_sensitive(&query, args.case_sensitive)
        && query_atoms
            .iter()
            .any(|atom| fallback_threshold(atom).is_some())
    {
        hits = search_levenshtein_fallback(&candidates, &query_atoms, args.case_sensitive);
        used_fallback = !hits.is_empty();
    }

    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.pattern_score.cmp(&a.pattern_score))
            .then_with(|| {
                candidates[b.candidate_idx]
                    .discovered_at
                    .cmp(&candidates[a.candidate_idx].discovered_at)
            })
            .then_with(|| {
                candidates[a.candidate_idx]
                    .path_raw
                    .cmp(&candidates[b.candidate_idx].path_raw)
            })
    });

    if used_fallback && !args.json {
        println!("(no fuzzy matches; falling back to typo-tolerant search)");
    }
    for hit in hits.iter().take(args.limit) {
        let candidate = &candidates[hit.candidate_idx];
        if args.json {
            print_json_result(hit, candidate)?;
        } else {
            print_text_result(hit, candidate);
        }
    }
    Ok(())
}

fn cmd_skill_path(paths: &AppPaths) -> Result<()> {
    let skill_path = paths
        .root
        .join("skills")
        .join("clawgallery")
        .join("SKILL.md");
    if let Some(parent) = skill_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&skill_path, include_str!("../skills/clawgallery/SKILL.md"))?;
    println!("{}", skill_path.display());
    Ok(())
}

fn cmd_status(paths: &AppPaths) -> Result<()> {
    let config = read_config(paths).unwrap_or_default();
    println!("config_dir: {}", paths.root.display());
    println!("provider: {}", config.provider);
    println!("model: {}", config.model);
    println!("folders: {}", active_folders(paths)?.len());
    println!("images: {}", latest_images(paths)?.len());
    println!("captions: {}", latest_captions(paths)?.len());
    Ok(())
}

fn read_config(paths: &AppPaths) -> Result<AppConfig> {
    if paths.config.exists() {
        let raw = fs::read_to_string(&paths.config)?;
        Ok(serde_json::from_str(&raw)?)
    } else {
        Ok(AppConfig::default())
    }
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let mut records = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str(&line) {
            records.push(record);
        }
    }
    Ok(records)
}

fn active_folders(paths: &AppPaths) -> Result<Vec<FolderRecord>> {
    let mut by_id: HashMap<String, FolderRecord> = HashMap::new();
    for folder in read_jsonl::<FolderRecord>(&paths.folders)? {
        by_id.insert(folder.id.clone(), folder);
    }
    let mut folders: Vec<_> = by_id.into_values().filter(|folder| folder.active).collect();
    folders.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(folders)
}

pub(crate) fn latest_images(paths: &AppPaths) -> Result<Vec<ImageRecord>> {
    let mut sequence: HashMap<PathBuf, usize> = HashMap::new();
    for (index, image) in read_jsonl::<ImageRecord>(&paths.images)?
        .into_iter()
        .enumerate()
    {
        sequence.insert(image.path.clone(), index);
    }
    let mut images: Vec<ImageRecord> = latest_images_by_path(paths)?.into_values().collect();
    images.sort_by_key(|image| sequence[&image.path]);
    Ok(images)
}

pub(crate) fn latest_images_refreshing_changed_files(
    paths: &AppPaths,
) -> Result<(Vec<ImageRecord>, bool)> {
    let mut images = latest_images(paths)?;
    let mut refreshed = false;
    for image in &mut images {
        if !image.path.exists() {
            continue;
        }
        let current = build_image_record(&image.path)?;
        if image.has_same_file_fingerprint(&current) {
            continue;
        }
        append_jsonl(&paths.images, &current)?;
        *image = current;
        refreshed = true;
    }
    Ok((images, refreshed))
}

fn latest_images_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, ImageRecord>> {
    Ok(all_latest_images_by_path(paths)?
        .into_iter()
        .filter(|(_, image)| image.active)
        .collect())
}

fn all_latest_images_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, ImageRecord>> {
    let mut images = HashMap::new();
    for image in read_jsonl::<ImageRecord>(&paths.images)? {
        images.insert(image.path.clone(), image);
    }
    Ok(images)
}

fn latest_captions(paths: &AppPaths) -> Result<Vec<CaptionRecord>> {
    Ok(latest_captions_by_path(paths)?.into_values().collect())
}

pub(crate) fn latest_captions_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, CaptionRecord>> {
    let mut captions = HashMap::new();
    for caption in read_jsonl::<CaptionRecord>(&paths.captions)? {
        captions.insert(caption.path.clone(), caption);
    }
    Ok(captions)
}

fn candidate_image_paths(paths: &AppPaths, args: &IngestArgs) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    if let Some(path) = &args.path {
        roots.push((path.clone(), true));
    } else {
        for folder in active_folders(paths)? {
            if args.folder.as_ref().is_none_or(|id| id == &folder.id) {
                roots.push((folder.path, folder.recursive));
            }
        }
    }
    let mut images = Vec::new();
    for (root, recursive) in roots {
        if root.is_file() {
            if is_image_path(&root) {
                images.push(root);
            }
            continue;
        }
        let walker = if recursive {
            WalkDir::new(&root)
        } else {
            WalkDir::new(&root).max_depth(1)
        };
        for entry in walker
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.into_path();
            if is_image_path(&path) {
                images.push(path);
            }
        }
    }
    images.sort();
    Ok(images)
}

fn build_image_record(path: &Path) -> Result<ImageRecord> {
    let metadata =
        fs::metadata(path).with_context(|| format!("metadata failed for {}", path.display()))?;
    let modified_at = metadata.modified().ok().map(DateTime::<Utc>::from);
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_lowercase();
    Ok(ImageRecord {
        id: Uuid::new_v4().to_string(),
        path: path.to_path_buf(),
        original_path: path.to_path_buf(),
        sha256: sha256_file(path)?,
        size: metadata.len(),
        modified_at,
        discovered_at: Utc::now(),
        extension,
        active: true,
        removed_at: None,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn canonicalize_existing_dir(path: &Path) -> Result<PathBuf> {
    let canonical =
        fs::canonicalize(path).with_context(|| format!("{} does not exist", path.display()))?;
    if !canonical.is_dir() {
        bail!("{} is not a directory", canonical.display());
    }
    Ok(canonical)
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

#[derive(Debug)]
struct Auth {
    bearer: String,
}

impl Auth {
    fn discover() -> Result<Self> {
        if let Ok(key) = env::var("OPENAI_API_KEY")
            && !key.trim().is_empty()
        {
            return Ok(Self { bearer: key });
        }
        let auth_path = codex_home().join("auth.json");
        if auth_path.exists() {
            let raw = fs::read_to_string(&auth_path)
                .with_context(|| format!("failed to read Codex auth at {}", auth_path.display()))?;
            let value: Value = serde_json::from_str(&raw)?;
            if let Some(key) = value
                .get("OPENAI_API_KEY")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                return Ok(Self {
                    bearer: key.to_string(),
                });
            }
            if let Some(token) = value
                .get("tokens")
                .and_then(|tokens| tokens.get("access_token"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                return Ok(Self {
                    bearer: token.to_string(),
                });
            }
        }
        bail!(
            "missing visual model credentials: set OPENAI_API_KEY or login with Codex so CODEX_HOME/auth.json contains OPENAI_API_KEY/tokens.access_token"
        )
    }
}

fn codex_home() -> PathBuf {
    env::var("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".codex")
        })
}

struct OpenAiCompatProvider {
    auth: Auth,
    model: String,
    base_url: String,
}

impl OpenAiCompatProvider {
    fn new(auth: Auth, model: String, base_url: String) -> Self {
        Self {
            auth,
            model,
            base_url,
        }
    }

    fn caption_image(&self, path: &Path) -> Result<CaptionOutput> {
        let image_data =
            fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let data_url = format!(
            "data:{};base64,{}",
            mime_for_path(path),
            base64::engine::general_purpose::STANDARD.encode(image_data)
        );
        let request = json!({
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": caption_prompt()},
                    {"type": "input_image", "image_url": data_url}
                ]
            }],
            "max_output_tokens": 500
        });
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let response: Value = reqwest::blocking::Client::new()
            .post(url)
            .bearer_auth(&self.auth.bearer)
            .json(&request)
            .send()?
            .error_for_status()?
            .json()?;
        parse_caption_response(&response)
    }

    fn classify_stem(&self, stem: &str) -> Result<bool> {
        let request = json!({
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": stem_classify_prompt(stem)}
                ]
            }],
            "max_output_tokens": 50
        });
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let response: Value = reqwest::blocking::Client::new()
            .post(url)
            .bearer_auth(&self.auth.bearer)
            .json(&request)
            .send()?
            .error_for_status()?
            .json()?;
        parse_stem_classification(&response)
    }
}

struct GeminiProvider {
    api_key: String,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    fn caption_image(&self, path: &Path) -> Result<CaptionOutput> {
        let image_data =
            fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(image_data);
        let request = json!({
            "contents": [{
                "parts": [
                    {"text": caption_prompt()},
                    {"inline_data": {"mime_type": mime_for_path(path), "data": b64}}
                ]
            }]
        });
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let response: Value = reqwest::blocking::Client::new()
            .post(&url)
            .json(&request)
            .send()?
            .error_for_status()?
            .json()?;
        let text = gemini_text(&response)?;
        parse_caption_text(&text)
    }

    fn classify_stem(&self, stem: &str) -> Result<bool> {
        let request = json!({
            "contents": [{
                "parts": [{"text": stem_classify_prompt(stem)}]
            }]
        });
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let response: Value = reqwest::blocking::Client::new()
            .post(&url)
            .json(&request)
            .send()?
            .error_for_status()?
            .json()?;
        let text = gemini_text(&response)?;
        parse_stem_classification_text(&text)
    }
}

fn gemini_text(response: &Value) -> Result<String> {
    response
        .get("candidates")
        .and_then(|c| c.as_array()?.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array()?.first())
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("Gemini response did not include text"))
}

fn caption_prompt() -> &'static str {
    "You are ClawGallery. Analyze this screenshot/image and return compact JSON only: \
     {\"title\":\"kebab or spaced concise filename title under 80 chars\",\
     \"description\":\"detailed searchable caption with visible text, app/site, UI state, entities, and likely context\"}."
}

fn stem_classify_prompt(stem: &str) -> String {
    format!(
        "You are classifying a single filename stem. \
         Return compact JSON only: {{\"meaningful\": <true|false>}}. \
         Do not look at, request, or imagine any image content. \
         Decide purely from the filename text. \
         Set meaningful=false ONLY if the stem looks auto-generated by a camera, screenshot tool, \
         messenger, browser download, or platform (e.g. IMG_0034, DSC04551, PXL_20240316_080000123, \
         Screenshot 2025-11-01 at 14.32.55, WhatsApp Image 2024-03-16 at 08.00.00, \
         KakaoTalk_20231109_221206834, image (1), Untitled, 1696862563748, 20230822_120055). \
         Set meaningful=true for any stem a human likely chose deliberately, including descriptive \
         English, names of people/teams/places, slang, project codenames, or non-Latin scripts \
         (Korean Hangul, Japanese, Chinese, Cyrillic, Arabic, etc.). When uncertain, prefer meaningful=true. \
         Stem to classify: {stem:?}"
    )
}

fn parse_caption_response(response: &Value) -> Result<CaptionOutput> {
    let text = response
        .get("output_text")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| collect_response_text(response));
    let text = text.ok_or_else(|| anyhow!("model response did not include output_text"))?;
    parse_caption_text(&text)
}

fn parse_caption_text(text: &str) -> Result<CaptionOutput> {
    let value: Value = serde_json::from_str(text.trim()).or_else(|_| {
        let start = text
            .find('{')
            .ok_or_else(|| anyhow!("caption response was not JSON"))?;
        let end = text
            .rfind('}')
            .ok_or_else(|| anyhow!("caption response was not JSON"))?;
        serde_json::from_str(&text[start..=end]).map_err(anyhow::Error::from)
    })?;
    Ok(CaptionOutput {
        title: value
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("untitled screenshot")
            .trim()
            .to_string(),
        description: value
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
    })
}

fn parse_stem_classification(response: &Value) -> Result<bool> {
    let text = response
        .get("output_text")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| collect_response_text(response));
    let text = text.ok_or_else(|| anyhow!("model response did not include output_text"))?;
    parse_stem_classification_text(&text)
}

fn parse_stem_classification_text(text: &str) -> Result<bool> {
    let value: Value = serde_json::from_str(text.trim()).or_else(|_| {
        let start = text
            .find('{')
            .ok_or_else(|| anyhow!("stem classification response was not JSON"))?;
        let end = text
            .rfind('}')
            .ok_or_else(|| anyhow!("stem classification response was not JSON"))?;
        serde_json::from_str(&text[start..=end]).map_err(anyhow::Error::from)
    })?;
    value
        .get("meaningful")
        .and_then(Value::as_bool)
        .ok_or_else(|| anyhow!("stem classification response did not include meaningful"))
}

fn collect_response_text(response: &Value) -> Option<String> {
    let mut parts = Vec::new();
    for item in response.get("output")?.as_array()? {
        for content in item.get("content")?.as_array()? {
            if let Some(text) = content.get("text").and_then(Value::as_str) {
                parts.push(text);
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn mime_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(OsStr::to_str)
        .map(|s| s.to_lowercase())
    {
        Some(ext) if ext == "jpg" || ext == "jpeg" => "image/jpeg",
        Some(ext) if ext == "webp" => "image/webp",
        Some(ext) if ext == "avif" => "image/avif",
        Some(ext) if ext == "gif" => "image/gif",
        _ => "image/png",
    }
}

fn rename_candidate(path: &Path, title: &str, limit_bytes: usize) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let extension = path.extension().and_then(OsStr::to_str).unwrap_or("png");
    let extension_part = format!(".{extension}");
    let suffix_budget = 10;
    let max_stem_bytes = limit_bytes
        .saturating_sub(extension_part.len())
        .saturating_sub(suffix_budget)
        .max(16);
    let stem = truncate_utf8_bytes(&sanitize_filename(title), max_stem_bytes);
    for index in 0..10_000 {
        let candidate_name = if index == 0 {
            format!("{stem}{extension_part}")
        } else {
            format!("{stem}-{index}{extension_part}")
        };
        let candidate = parent.join(candidate_name);
        if candidate == path || !candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not find non-colliding filename for {}",
        path.display()
    )
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum NameClassification {
    Generic,
    NeedsModel,
}

fn strip_copy_and_sequence_suffix(stem: &str) -> &str {
    let copy_re = regex::Regex::new(r"(?i)(?:[ _-](?:copy|복사본|사본)(?:[ _-]?\d+)?| \(\d+\))+$")
        .expect("copy suffix regex compiles");
    let trimmed = copy_re.replace(stem, "");
    let len = trimmed.len();
    &stem[..len.min(stem.len())]
}

fn is_generic_filename(stem: &str) -> bool {
    let stem = strip_copy_and_sequence_suffix(stem.trim()).trim();
    if stem.is_empty() {
        return true;
    }

    let patterns: &[&str] = &[
        r"^\d+$",
        r"(?i)^(?:image|download|img|photo|picture|untitled)$",
        r"(?i)^(?:image|download|img|photo|picture)\s*\(\d+\)$",
        r"^(?:Screenshot|Screen Shot|Annotation|Captura de pantalla|Снимок экрана|스크린샷|화면 캡처)\s\d{4}-\d{2}-\d{2}(?:[ T]| at | a las )\d{1,2}[.: -]\d{2}[.: -]\d{2}(?:\s?(?:AM|PM))?$",
        r"^WhatsApp\s(?:Image|Video)\s\d{4}-\d{2}-\d{2}\sat\s\d{1,2}[.: ]\d{2}[.: ]\d{2}$",
        r"^KakaoTalk_\d{8}_\d{6,9}$",
        r"^PXL_\d{8}_\d{9}(?:~\d+)?$",
        r"^(?:IMG|VID)_\d{8}_\d{6}(?:_\d{1,3})?$",
        r"^(?:IMG|VID)[-_]\d{8}[-_]?WA\d{2,6}$",
        r"^(?:DSC|DSCF|DSCN)\d{4,8}(?:_\d{1,6}px)?$",
        r"^IMG_\d{4,5}(?:[ _]Medium| Large| Small| HEIC)?$",
        r"^\d{8}_\d{6}(?:_\d{1,4})?$",
    ];
    for pat in patterns {
        let re = regex::Regex::new(pat).expect("classifier regex compiles");
        if re.is_match(stem) {
            return true;
        }
    }
    false
}

fn classify_filename(stem: &str) -> NameClassification {
    if is_generic_filename(stem) {
        NameClassification::Generic
    } else {
        NameClassification::NeedsModel
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum RenameDecision {
    Rename,
    Skip,
}

fn rename_decision(
    stem: &str,
    cached_filename_meaningful: Option<bool>,
    explicit_file: bool,
    force: bool,
) -> RenameDecision {
    if explicit_file || force {
        return RenameDecision::Rename;
    }
    if let Some(meaningful) = cached_filename_meaningful {
        return if meaningful {
            RenameDecision::Skip
        } else {
            RenameDecision::Rename
        };
    }
    match classify_filename(stem) {
        NameClassification::Generic => RenameDecision::Rename,
        NameClassification::NeedsModel => RenameDecision::Skip,
    }
}

fn sanitize_filename(input: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in input.trim().chars() {
        let replacement = if ch.is_ascii_alphanumeric() || matches!(ch, '가'..='힣') {
            Some(ch.to_ascii_lowercase())
        } else if ch.is_whitespace()
            || matches!(
                ch,
                '-' | '_' | '.' | ':' | '/' | '\\' | '|' | '?' | '*' | '"' | '<' | '>'
            )
        {
            Some('-')
        } else if ch.is_control() {
            None
        } else {
            Some(ch)
        };
        if let Some(ch) = replacement {
            if ch == '-' {
                if !last_dash {
                    out.push(ch);
                }
                last_dash = true;
            } else {
                out.push(ch);
                last_dash = false;
            }
        }
    }
    let cleaned = out.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "untitled-screenshot".to_string()
    } else {
        cleaned
    }
}

fn truncate_utf8_bytes(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }
    let mut end = 0;
    for (idx, ch) in input.char_indices() {
        let next = idx + ch.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }
    input[..end].trim_end_matches('-').to_string()
}

fn log_error(paths: &AppPaths, context: &str, err: anyhow::Error) {
    let message = mask_api_keys(&err.to_string());
    let record = ErrorRecord {
        context: context.to_string(),
        message: message.clone(),
        created_at: Utc::now(),
    };
    let _ = append_jsonl(&paths.errors, &record);
    eprintln!("{context}: {message}");
}

fn mask_api_keys(input: &str) -> String {
    let key_query = regex::Regex::new(r"(?i)([?&]key=)[^\s&)]+").expect("key= regex compiles");
    let bearer =
        regex::Regex::new(r"(?i)(Bearer\s+)[A-Za-z0-9_\-\.]+").expect("bearer regex compiles");
    let openai = regex::Regex::new(r"sk-[A-Za-z0-9_\-]{16,}").expect("openai regex compiles");
    let gemini = regex::Regex::new(r"AIza[0-9A-Za-z_\-]{35}").expect("gemini regex compiles");
    let step1 = key_query.replace_all(input, "${1}REDACTED");
    let step2 = bearer.replace_all(&step1, "${1}REDACTED");
    let step3 = openai.replace_all(&step2, "REDACTED");
    let step4 = gemini.replace_all(&step3, "REDACTED");
    step4.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_images_case_insensitively() {
        assert!(is_image_path(Path::new("Screen.PNG")));
        assert!(is_image_path(Path::new("photo.avif")));
        assert!(!is_image_path(Path::new("notes.txt")));
    }

    #[test]
    fn sanitizes_and_truncates_filename() {
        assert_eq!(
            sanitize_filename(" Hello / World: Screenshot? "),
            "hello-world-screenshot"
        );
        assert_eq!(truncate_utf8_bytes("가나다abc", 7), "가나");
    }

    #[test]
    fn rename_candidate_preserves_extension_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.png");
        fs::write(&path, b"x").unwrap();
        let candidate =
            rename_candidate(&path, "Very Long Screenshot Title With Spaces", 32).unwrap();
        let name = candidate.file_name().unwrap().to_string_lossy();
        assert!(name.ends_with(".png"));
        assert!(name.len() <= 32);
    }

    #[test]
    fn jsonl_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let folder = FolderRecord {
            id: "one".into(),
            path: PathBuf::from("/tmp/screens"),
            recursive: true,
            active: true,
            created_at: Utc::now(),
            removed_at: None,
        };
        append_jsonl(&path, &folder).unwrap();
        let records = read_jsonl::<FolderRecord>(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "one");
    }

    #[test]
    fn parses_openai_output_text_json() {
        let response = json!({"output_text":"{\"title\":\"Settings screen\",\"description\":\"A macOS settings screenshot\"}"});
        let parsed = parse_caption_response(&response).unwrap();
        assert_eq!(parsed.title, "Settings screen");
        assert!(parsed.description.contains("macOS"));
    }

    #[test]
    fn parses_stem_classification_true() {
        let json = r#"{"meaningful": true}"#;
        assert!(parse_stem_classification_text(json).unwrap());
    }

    #[test]
    fn parses_stem_classification_false_with_padding() {
        let wrapped = "Sure: {\"meaningful\": false} -- done.";
        assert!(!parse_stem_classification_text(wrapped).unwrap());
    }

    #[test]
    fn rejects_stem_classification_without_field() {
        let json = r#"{"foo": true}"#;
        assert!(parse_stem_classification_text(json).is_err());
    }

    #[test]
    fn parses_caption_text_directly() {
        let text = r#"{"title":"Gemini result","description":"A Gemini-generated description"}"#;
        let parsed = parse_caption_text(text).unwrap();
        assert_eq!(parsed.title, "Gemini result");
        assert!(parsed.description.contains("Gemini"));
    }

    #[test]
    fn parses_gemini_response_text() {
        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "{\"title\":\"Gemini screen\",\"description\":\"A Gemini vision result\"}"}]
                }
            }]
        });
        let text = response
            .get("candidates")
            .and_then(|c| c.as_array()?.first())
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array()?.first())
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .unwrap();
        let parsed = parse_caption_text(text).unwrap();
        assert_eq!(parsed.title, "Gemini screen");
        assert!(parsed.description.contains("vision"));
    }

    #[test]
    fn cli_provider_overrides_config_provider() {
        assert_eq!(
            resolve_provider(Some("gemini"), "openai-compatible"),
            "gemini"
        );
    }

    #[test]
    fn config_provider_used_when_cli_absent() {
        assert_eq!(
            resolve_provider(None, "openai-compatible"),
            "openai-compatible"
        );
    }

    #[test]
    fn cli_model_overrides_config_model() {
        assert_eq!(
            resolve_model(Some("gemini-2.5-flash"), "gpt-4.1-mini"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn config_model_used_when_cli_absent() {
        assert_eq!(resolve_model(None, "gpt-4.1-mini"), "gpt-4.1-mini");
    }

    #[test]
    fn is_generic_filename_detects_known_machine_patterns() {
        let generic = [
            "IMG_0034",
            "IMG_3963",
            "IMG_0621 Medium",
            "DSC04551",
            "DSCF1234",
            "DSCN0001",
            "PXL_20240316_080000123",
            "PXL_20240316_080000123~2",
            "Screenshot 2025-11-01 at 14.32.55",
            "Screen Shot 2025-11-01 at 2.32.55 PM",
            "Captura de pantalla 2024-03-16 a las 8.00.00",
            "WhatsApp Image 2024-03-16 at 08.00.00",
            "WhatsApp Image 2024-03-16 at 08.00.00 (1)",
            "KakaoTalk_20231109_221206834",
            "KakaoTalk_20231109_221206834 (2)",
            "IMG-20231124-WA0001",
            "IMG_20230915_123456",
            "VID_20230915_123456_001",
            "20230822_120055",
            "20230822_120055_001",
            "1696862563748",
            "1000010690",
            "image (1)",
            "image (12)",
            "download (3)",
            "Untitled",
            "untitled",
            "IMG_0034 copy",
            "IMG_0034 copy 2",
            "image (1) copy",
        ];
        for stem in generic {
            assert!(
                is_generic_filename(stem),
                "expected generic, but classifier said meaningful: {stem:?}"
            );
        }
    }

    #[test]
    fn is_generic_filename_does_not_match_human_authored_names() {
        let not_generic = [
            "Eva-William",
            "기아 승리 열차",
            "마데이라_노마드",
            "김동규_jeffrey_AWS_발표",
            "홍창기",
            "꼴데",
            "수능_국어_상위_5%_차트",
            "screenshot-payment-flow",
            "DSC_2024_summer_trip",
            "IMG_0034_beach",
            "image-final-cover",
            "report-q3",
            "test-image",
        ];
        for stem in not_generic {
            assert!(
                !is_generic_filename(stem),
                "regex must not match human-authored stems: {stem:?}"
            );
        }
    }

    #[test]
    fn pure_numeric_stems_are_generic() {
        for stem in ["2024", "12345", "1000010690", "1696862563748"] {
            assert!(
                is_generic_filename(stem),
                "pure-numeric stem {stem:?} must be generic per user policy"
            );
        }
    }

    #[test]
    fn classify_filename_only_generic_or_needs_model() {
        assert_eq!(classify_filename("IMG_0034"), NameClassification::Generic);
        assert_eq!(
            classify_filename("1696862563748"),
            NameClassification::Generic
        );
        for ambiguous in [
            "기아 승리 열차",
            "마데이라_노마드",
            "Eva-William",
            "test-image",
            "DSC_2024_summer_trip",
            "김동규_jeffrey_AWS_발표",
        ] {
            assert_eq!(
                classify_filename(ambiguous),
                NameClassification::NeedsModel,
                "anything not regex-generic must be delegated to the model: {ambiguous:?}"
            );
        }
    }

    #[test]
    fn rename_decision_renames_generic_by_default() {
        assert_eq!(
            rename_decision("IMG_0034", None, false, false),
            RenameDecision::Rename
        );
    }

    #[test]
    fn rename_decision_skips_unknown_stems_without_cached_answer() {
        assert_eq!(
            rename_decision("기아 승리 열차", None, false, false),
            RenameDecision::Skip
        );
        assert_eq!(
            rename_decision("Eva-William", None, false, false),
            RenameDecision::Skip
        );
    }

    #[test]
    fn rename_decision_force_overrides_skip() {
        assert_eq!(
            rename_decision("기아 승리 열차", None, false, true),
            RenameDecision::Rename
        );
    }

    #[test]
    fn rename_decision_explicit_file_overrides_skip() {
        assert_eq!(
            rename_decision("Eva-William", None, true, false),
            RenameDecision::Rename
        );
    }

    #[test]
    fn rename_decision_uses_cached_model_answer() {
        assert_eq!(
            rename_decision("Eva-William", Some(true), false, false),
            RenameDecision::Skip,
            "cached meaningful=true should win over local heuristic"
        );
        assert_eq!(
            rename_decision("Eva-William", Some(false), false, false),
            RenameDecision::Rename,
            "cached meaningful=false should win over local heuristic"
        );
    }

    #[test]
    fn rename_decision_skips_needs_model_when_no_cache() {
        assert_eq!(
            rename_decision("test-image", None, false, false),
            RenameDecision::Skip,
            "ambiguous names without a cached model answer must be skipped conservatively"
        );
    }

    #[test]
    fn mask_api_keys_redacts_query_string_keys() {
        let s = "404 for url (https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyAoIIGoqlaB0OMD5I958MYdJ1TCcd5JgYA)";
        let masked = mask_api_keys(s);
        assert!(!masked.contains("AIzaSyAoIIGoqlaB0OMD5I958MYdJ1TCcd5JgYA"));
        assert!(masked.contains("key=REDACTED"));
    }

    #[test]
    fn mask_api_keys_redacts_bearer_tokens() {
        let s = "Authorization: Bearer sk-proj-uLjyl6YxDb_vbQbHi0vPR3hyfJGwSVeoYIEwddMkMPCF7OjUi8iN8UafXklvRARqGYow2DiuFST3BlbkFJiDpamU";
        let masked = mask_api_keys(s);
        assert!(!masked.contains("sk-proj-uLjyl6YxDb"));
        assert!(masked.contains("REDACTED"));
    }

    #[test]
    fn mask_api_keys_redacts_loose_openai_keys() {
        let s = "leak: sk-proj-uLjyl6YxDb_vbQbHi0vPR3hyfJGwSVeoYIEwddMkMPCF7OjUi8iN8UafXklvRARqGYow2DiuFST3BlbkFJiDpamU somewhere in middle";
        let masked = mask_api_keys(s);
        assert!(!masked.contains("sk-proj-uLjyl6YxDb"));
    }

    #[test]
    fn mask_api_keys_redacts_loose_gemini_keys() {
        let s = "leak: AIzaSyAoIIGoqlaB0OMD5I958MYdJ1TCcd5JgYA inline";
        let masked = mask_api_keys(s);
        assert!(!masked.contains("AIzaSyAoIIGoqlaB0OMD5I958MYdJ1TCcd5JgYA"));
    }

    #[test]
    fn mask_api_keys_is_idempotent_on_clean_text() {
        let s = "no secrets here, just plain text";
        assert_eq!(mask_api_keys(s), s);
    }

    #[test]
    fn auth_reads_codex_auth_json_api_key() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            r#"{"OPENAI_API_KEY":"sk-test"}"#,
        )
        .unwrap();
        unsafe {
            env::remove_var("OPENAI_API_KEY");
            env::set_var("CODEX_HOME", dir.path());
        }
        let auth = Auth::discover().unwrap();
        assert_eq!(auth.bearer, "sk-test");
    }
}
