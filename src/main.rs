use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use uuid::Uuid;
use walkdir::WalkDir;

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
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RenameStyle {
    Title,
    Caption,
    DateTitle,
}

#[derive(Debug, Args)]
struct SearchArgs {
    keywords: Vec<String>,
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Debug, Subcommand)]
enum SkillCommand {
    /// Print the path to the bundled skill.
    Path,
    /// Print the bundled skill instructions.
    Print,
}

#[derive(Debug, Clone)]
struct AppPaths {
    root: PathBuf,
    config: PathBuf,
    folders: PathBuf,
    images: PathBuf,
    captions: PathBuf,
    renames: PathBuf,
    errors: PathBuf,
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
            root,
        })
    }

    fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AppConfig {
    model: String,
    provider: String,
    filename_limit_bytes: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: env::var("CLAWGALLERY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            provider: "openai-compatible".to_string(),
            filename_limit_bytes: DEFAULT_FILENAME_LIMIT_BYTES,
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
struct ImageRecord {
    id: String,
    path: PathBuf,
    original_path: PathBuf,
    sha256: String,
    size: u64,
    modified_at: Option<DateTime<Utc>>,
    discovered_at: DateTime<Utc>,
    extension: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CaptionRecord {
    image_id: String,
    path: PathBuf,
    title: String,
    description: String,
    model: String,
    provider: String,
    created_at: DateTime<Utc>,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::resolve()?;
    match cli.command {
        Command::Init => cmd_init(&paths),
        Command::Folder { command } => match command {
            FolderCommand::Add(args) => cmd_folder_add(&paths, args),
            FolderCommand::Remove(args) => cmd_folder_remove(&paths, args),
            FolderCommand::List => cmd_folder_list(&paths),
        },
        Command::Bootstrap(args) => cmd_bootstrap(&paths, &args).map(|count| {
            println!("ingested {count} new image(s)");
        }),
        Command::Poll(args) => cmd_poll(&paths, args),
        Command::Caption(args) => cmd_caption(&paths, args),
        Command::Rename(args) => cmd_rename(&paths, args),
        Command::Search(args) => cmd_search(&paths, args),
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

fn cmd_bootstrap(paths: &AppPaths, args: &IngestArgs) -> Result<usize> {
    paths.ensure()?;
    if !paths.config.exists() {
        write_json_pretty(&paths.config, &AppConfig::default())?;
    }
    let existing = latest_images_by_path(paths)?;
    let mut seen_paths: HashSet<PathBuf> = existing.keys().cloned().collect();
    let mut new_count = 0;
    for image_path in candidate_image_paths(paths, args)? {
        let canonical = fs::canonicalize(&image_path).unwrap_or(image_path.clone());
        if seen_paths.contains(&canonical) {
            continue;
        }
        match build_image_record(&canonical) {
            Ok(record) => {
                append_jsonl(&paths.images, &record)?;
                seen_paths.insert(record.path.clone());
                new_count += 1;
            }
            Err(err) => log_error(paths, "ingest", err),
        }
    }
    Ok(new_count)
}

fn cmd_poll(paths: &AppPaths, args: PollArgs) -> Result<()> {
    loop {
        let count = cmd_bootstrap(paths, &args.ingest)?;
        println!("{}: ingested {count} new image(s)", Utc::now().to_rfc3339());
        if args.once {
            break;
        }
        thread::sleep(Duration::from_secs(args.interval.max(1)));
    }
    Ok(())
}

fn cmd_caption(paths: &AppPaths, args: CaptionArgs) -> Result<()> {
    paths.ensure()?;
    let config = read_config(paths)?;
    let model = args.model.unwrap_or(config.model);
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
    let auth = Auth::discover()?;
    let client = OpenAiCompatClient::new(auth, model.clone());
    for image in images {
        match client.caption_image(&image.path) {
            Ok(output) => {
                let record = CaptionRecord {
                    image_id: image.id.clone(),
                    path: image.path.clone(),
                    title: output.title,
                    description: output.description,
                    model: model.clone(),
                    provider: "openai-compatible".to_string(),
                    created_at: Utc::now(),
                };
                append_jsonl(&paths.captions, &record)?;
                println!("captioned {} -> {}", image.path.display(), record.title);
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
    if let Some(file) = args.file {
        let canonical = fs::canonicalize(&file).unwrap_or(file);
        images.retain(|image| image.path == canonical);
        if images.is_empty() && canonical.exists() {
            images.push(build_image_record(&canonical)?);
        }
    }
    for image in images {
        let Some(caption) = captions.get(&image.path) else {
            continue;
        };
        let title = match args.style {
            RenameStyle::Title => caption.title.clone(),
            RenameStyle::Caption => caption.description.clone(),
            RenameStyle::DateTitle => format!(
                "{} {}",
                image.discovered_at.format("%Y-%m-%d"),
                caption.title
            ),
        };
        let target = rename_candidate(&image.path, &title, config.filename_limit_bytes)?;
        let record = RenameRecord {
            image_id: Some(image.id.clone()),
            from: image.path.clone(),
            to: target.clone(),
            applied: args.apply,
            reason: format!("style={:?}", args.style),
            created_at: Utc::now(),
        };
        if args.apply {
            if target.exists() {
                bail!("refusing to overwrite existing file {}", target.display());
            }
            fs::rename(&image.path, &target).with_context(|| {
                format!(
                    "failed to rename {} to {}",
                    image.path.display(),
                    target.display()
                )
            })?;
            append_jsonl(&paths.renames, &record)?;
            let mut updated = image.clone();
            updated.path = fs::canonicalize(&target).unwrap_or(target.clone());
            append_jsonl(&paths.images, &updated)?;
            println!("renamed {} -> {}", image.path.display(), target.display());
        } else {
            append_jsonl(&paths.renames, &record)?;
            println!("dry-run {} -> {}", image.path.display(), target.display());
        }
    }
    Ok(())
}

fn cmd_search(paths: &AppPaths, args: SearchArgs) -> Result<()> {
    if args.keywords.is_empty() {
        bail!("provide at least one keyword");
    }
    let needle: Vec<String> = args.keywords.iter().map(|k| k.to_lowercase()).collect();
    let captions = latest_captions_by_path(paths)?;
    let images = latest_images(paths)?;
    let mut printed = 0;
    for image in images {
        let cap = captions.get(&image.path);
        let haystack = format!(
            "{} {} {}",
            image.path.display(),
            cap.map(|c| c.title.as_str()).unwrap_or_default(),
            cap.map(|c| c.description.as_str()).unwrap_or_default()
        )
        .to_lowercase();
        if needle.iter().all(|keyword| haystack.contains(keyword)) {
            println!(
                "{}\n  title: {}\n  caption: {}",
                image.path.display(),
                cap.map(|c| c.title.as_str()).unwrap_or("<missing>"),
                cap.map(|c| c.description.as_str()).unwrap_or("<missing>")
            );
            printed += 1;
            if printed >= args.limit {
                break;
            }
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

fn latest_images(paths: &AppPaths) -> Result<Vec<ImageRecord>> {
    Ok(latest_images_by_path(paths)?.into_values().collect())
}

fn latest_images_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, ImageRecord>> {
    let mut images = HashMap::new();
    for image in read_jsonl::<ImageRecord>(&paths.images)? {
        images.insert(image.path.clone(), image);
    }
    Ok(images)
}

fn latest_captions(paths: &AppPaths) -> Result<Vec<CaptionRecord>> {
    Ok(latest_captions_by_path(paths)?.into_values().collect())
}

fn latest_captions_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, CaptionRecord>> {
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
    base_url: String,
}

impl Auth {
    fn discover() -> Result<Self> {
        let base_url =
            env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        if let Ok(key) = env::var("OPENAI_API_KEY")
            && !key.trim().is_empty()
        {
            return Ok(Self {
                bearer: key,
                base_url,
            });
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
                    base_url,
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
                    base_url,
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

struct OpenAiCompatClient {
    auth: Auth,
    model: String,
}

impl OpenAiCompatClient {
    fn new(auth: Auth, model: String) -> Self {
        Self { auth, model }
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
                    {"type": "input_text", "text": "You are ClawGallery. Analyze this screenshot/image and return compact JSON only: {\"title\":\"kebab or spaced concise filename title under 80 chars\",\"description\":\"detailed searchable caption with visible text, app/site, UI state, entities, and likely context\"}."},
                    {"type": "input_image", "image_url": data_url}
                ]
            }],
            "max_output_tokens": 500
        });
        let url = format!("{}/responses", self.auth.base_url.trim_end_matches('/'));
        let response: Value = reqwest::blocking::Client::new()
            .post(url)
            .bearer_auth(&self.auth.bearer)
            .json(&request)
            .send()?
            .error_for_status()?
            .json()?;
        parse_caption_response(&response)
    }
}

fn parse_caption_response(response: &Value) -> Result<CaptionOutput> {
    let text = response
        .get("output_text")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| collect_response_text(response));
    let text = text.ok_or_else(|| anyhow!("model response did not include output_text"))?;
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
    let record = ErrorRecord {
        context: context.to_string(),
        message: err.to_string(),
        created_at: Utc::now(),
    };
    let _ = append_jsonl(&paths.errors, &record);
    eprintln!("{context}: {err}");
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
