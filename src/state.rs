use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

use crate::{APP_DIR_NAME, DEFAULT_FILENAME_LIMIT_BYTES, DEFAULT_MODEL};

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "avif", "gif", "heic", "heif"];

#[derive(Debug, Clone)]
pub(crate) struct AppPaths {
    pub(crate) root: PathBuf,
    pub(crate) config: PathBuf,
    pub(crate) folders: PathBuf,
    pub(crate) images: PathBuf,
    pub(crate) captions: PathBuf,
    pub(crate) renames: PathBuf,
    pub(crate) errors: PathBuf,
    pub(crate) vdr_db: PathBuf,
}

impl AppPaths {
    pub(crate) fn resolve() -> Result<Self> {
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
pub(crate) struct AppConfig {
    #[serde(default = "default_model")]
    pub(crate) model: String,
    #[serde(default = "default_provider_str")]
    pub(crate) provider: String,
    #[serde(default = "default_filename_limit")]
    pub(crate) filename_limit_bytes: usize,
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
pub(crate) struct FolderRecord {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) recursive: bool,
    pub(crate) active: bool,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) removed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ImageRecord {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) original_path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) size: u64,
    pub(crate) modified_at: Option<DateTime<Utc>>,
    pub(crate) discovered_at: DateTime<Utc>,
    pub(crate) extension: String,
    #[serde(default = "default_active")]
    pub(crate) active: bool,
    #[serde(default)]
    pub(crate) removed_at: Option<DateTime<Utc>>,
}

fn default_active() -> bool {
    true
}

pub(crate) enum ImageRefresh {
    Unchanged,
    MetadataOnly(ImageRecord),
    ContentChanged(ImageRecord),
    New(ImageRecord),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct CaptionRecord {
    pub(crate) image_id: String,
    pub(crate) path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_sha256: Option<String>,
    pub(crate) title: String,
    pub(crate) description: String,
    pub(crate) model: String,
    pub(crate) provider: String,
    pub(crate) created_at: DateTime<Utc>,
    #[serde(default)]
    pub(crate) filename_meaningful: Option<bool>,
}

pub(crate) fn read_config(paths: &AppPaths) -> Result<AppConfig> {
    if paths.config.exists() {
        let raw = fs::read_to_string(&paths.config)?;
        Ok(serde_json::from_str(&raw)?)
    } else {
        Ok(AppConfig::default())
    }
}

pub(crate) fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

pub(crate) fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

pub(crate) fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse {} line {}", path.display(), index + 1))?;
        records.push(record);
    }
    Ok(records)
}

pub(crate) fn active_folders(paths: &AppPaths) -> Result<Vec<FolderRecord>> {
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

pub(crate) fn latest_images_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, ImageRecord>> {
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

pub(crate) fn latest_captions(paths: &AppPaths) -> Result<Vec<CaptionRecord>> {
    Ok(latest_captions_by_path(paths)?.into_values().collect())
}

pub(crate) fn latest_captions_by_path(paths: &AppPaths) -> Result<HashMap<PathBuf, CaptionRecord>> {
    let images = latest_images_by_path(paths)?;
    Ok(all_latest_captions_by_path(paths)?
        .into_iter()
        .filter_map(|(path, mut caption)| {
            let image = images.get(&path)?;
            let is_current = caption.source_sha256.as_deref().map_or_else(
                || caption.image_id == image.id,
                |sha256| sha256 == image.sha256,
            );
            if !is_current {
                return None;
            }
            caption.image_id.clone_from(&image.id);
            Some((path, caption))
        })
        .collect())
}

pub(crate) fn all_latest_captions_by_path(
    paths: &AppPaths,
) -> Result<HashMap<PathBuf, CaptionRecord>> {
    let mut captions = HashMap::new();
    for caption in read_jsonl::<CaptionRecord>(&paths.captions)? {
        captions.insert(caption.path.clone(), caption);
    }
    Ok(captions)
}

pub(crate) fn build_image_record(path: &Path) -> Result<ImageRecord> {
    match refresh_image_record(path, None)? {
        ImageRefresh::New(record) => Ok(record),
        ImageRefresh::Unchanged
        | ImageRefresh::MetadataOnly(_)
        | ImageRefresh::ContentChanged(_) => Err(anyhow!(
            "new image unexpectedly resolved as an existing record"
        )),
    }
}

pub(crate) fn refresh_image_record(
    path: &Path,
    previous: Option<&ImageRecord>,
) -> Result<ImageRefresh> {
    let metadata =
        fs::metadata(path).with_context(|| format!("metadata failed for {}", path.display()))?;
    let modified_at = metadata.modified().ok().map(DateTime::<Utc>::from);
    if previous.is_some_and(|image| {
        image.size == metadata.len() && image.modified_at == modified_at && image.active
    }) {
        return Ok(ImageRefresh::Unchanged);
    }
    let sha256 = sha256_file(path)?;
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_lowercase();
    let Some(previous) = previous else {
        return Ok(ImageRefresh::New(ImageRecord {
            id: Uuid::new_v4().to_string(),
            path: path.to_path_buf(),
            original_path: path.to_path_buf(),
            sha256,
            size: metadata.len(),
            modified_at,
            discovered_at: Utc::now(),
            extension,
            active: true,
            removed_at: None,
        }));
    };
    let content_changed = previous.sha256 != sha256;
    let record = ImageRecord {
        id: previous.id.clone(),
        path: path.to_path_buf(),
        original_path: previous.original_path.clone(),
        sha256,
        size: metadata.len(),
        modified_at,
        discovered_at: previous.discovered_at,
        extension,
        active: true,
        removed_at: None,
    };
    if content_changed {
        Ok(ImageRefresh::ContentChanged(record))
    } else {
        Ok(ImageRefresh::MetadataOnly(record))
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    sha256_reader(BufReader::new(file))
        .with_context(|| format!("failed to read {}", path.display()))
}

fn sha256_reader(mut reader: impl Read) -> Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};

    struct BoundedRead<R> {
        inner: R,
        max_buffer: usize,
    }

    impl<R: Read> Read for BoundedRead<R> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(
                buffer.len() <= self.max_buffer,
                "hashing requested a {} byte buffer",
                buffer.len()
            );
            self.inner.read(buffer)
        }
    }

    #[test]
    fn sha256_reader_hashes_input_in_bounded_chunks() {
        // Given: input larger than the maximum read buffer accepted by the source.
        let bytes = vec![0x5a; 32 * 1024];
        let reader = BoundedRead {
            inner: Cursor::new(&bytes),
            max_buffer: 8 * 1024,
        };

        // When: the streaming hash helper consumes it.
        let digest = sha256_reader(reader).expect("streaming hash");

        // Then: the digest is complete without requesting a whole-file buffer.
        assert_eq!(digest, format!("{:x}", Sha256::digest(&bytes)));
    }
}
