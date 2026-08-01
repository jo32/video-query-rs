//! Persistent SQLite-backed semantic frame index.

use crate::embedding::{EMBEDDING_DIMENSION, MODEL_VERSION};
use crate::keyframe::FrameQuality;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: &str = "1";
const FINGERPRINT_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub struct IndexedFrame {
    pub timestamp_seconds: f64,
    pub quality: FrameQuality,
    pub image_path: PathBuf,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct IndexedVideo {
    pub path: PathBuf,
    pub fingerprint: String,
    pub duration_seconds: f64,
    pub width: usize,
    pub height: usize,
    pub frames: Vec<IndexedFrame>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchMatch {
    pub video_path: PathBuf,
    pub timestamp_seconds: f64,
    pub similarity: f32,
    pub quality_score: f64,
    pub frame_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct IndexStats {
    pub videos: usize,
    pub frames: usize,
}

pub struct VideoIndex {
    root: PathBuf,
    connection: Connection,
}

impl VideoIndex {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("frames"))
            .with_context(|| format!("failed to create index directory {}", root.display()))?;
        let connection = Connection::open(root.join("index.sqlite3"))?;
        connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS metadata (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS videos (
                 id INTEGER PRIMARY KEY,
                 path TEXT NOT NULL UNIQUE,
                 fingerprint TEXT NOT NULL,
                 duration_seconds REAL NOT NULL,
                 width INTEGER NOT NULL,
                 height INTEGER NOT NULL,
                 indexed_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS frames (
                 id INTEGER PRIMARY KEY,
                 video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                 timestamp_seconds REAL NOT NULL,
                 quality_score REAL NOT NULL,
                 laplacian_variance REAL NOT NULL,
                 gradient_rms REAL NOT NULL,
                 mean_luma REAL NOT NULL,
                 contrast REAL NOT NULL,
                 clipped_fraction REAL NOT NULL,
                 image_path TEXT NOT NULL,
                 embedding BLOB NOT NULL
             );
             CREATE INDEX IF NOT EXISTS frames_video_time
                 ON frames(video_id, timestamp_seconds);",
        )?;

        let index = Self { root, connection };
        index.check_metadata()?;
        Ok(index)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn frames_directory(&self, fingerprint: &str) -> PathBuf {
        self.root.join("frames").join(fingerprint)
    }

    pub fn contains_current(&self, path: &Path, fingerprint: &str) -> Result<bool> {
        let path = path.to_string_lossy();
        let stored: Option<String> = self
            .connection
            .query_row(
                "SELECT fingerprint FROM videos WHERE path = ?1",
                [path.as_ref()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(stored.as_deref() == Some(fingerprint))
    }

    pub fn replace_video(&mut self, video: &IndexedVideo) -> Result<()> {
        for frame in &video.frames {
            if frame.embedding.len() != EMBEDDING_DIMENSION {
                bail!(
                    "frame at {:.3}s has {} dimensions; expected {EMBEDDING_DIMENSION}",
                    frame.timestamp_seconds,
                    frame.embedding.len()
                )
            }
        }

        let transaction = self.connection.transaction()?;
        let path = video.path.to_string_lossy();
        transaction.execute("DELETE FROM videos WHERE path = ?1", [path.as_ref()])?;
        transaction.execute(
            "INSERT INTO videos
             (path, fingerprint, duration_seconds, width, height, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                path.as_ref(),
                video.fingerprint,
                video.duration_seconds,
                video.width as i64,
                video.height as i64,
                unix_timestamp(),
            ],
        )?;
        let video_id = transaction.last_insert_rowid();

        {
            let mut statement = transaction.prepare_cached(
                "INSERT INTO frames
                 (video_id, timestamp_seconds, quality_score, laplacian_variance,
                  gradient_rms, mean_luma, contrast, clipped_fraction,
                  image_path, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for frame in &video.frames {
                statement.execute(params![
                    video_id,
                    frame.timestamp_seconds,
                    frame.quality.score,
                    frame.quality.laplacian_variance,
                    frame.quality.gradient_rms,
                    frame.quality.mean_luma,
                    frame.quality.contrast,
                    frame.quality.clipped_fraction,
                    frame.image_path.to_string_lossy().as_ref(),
                    embedding_to_blob(&frame.embedding),
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchMatch>> {
        if query.len() != EMBEDDING_DIMENSION {
            bail!(
                "query has {} dimensions; expected {EMBEDDING_DIMENSION}",
                query.len()
            )
        }
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut statement = self.connection.prepare(
            "SELECT videos.path, frames.timestamp_seconds, frames.quality_score,
                    frames.image_path, frames.embedding
             FROM frames JOIN videos ON videos.id = frames.video_id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                PathBuf::from(row.get::<_, String>(0)?),
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
                PathBuf::from(row.get::<_, String>(3)?),
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;

        let mut stored = Vec::new();
        for row in rows {
            let (video_path, timestamp_seconds, quality_score, frame_path, blob) = row?;
            stored.push((
                video_path,
                timestamp_seconds,
                quality_score,
                frame_path,
                blob_to_embedding(&blob)?,
            ));
        }

        let mut matches: Vec<SearchMatch> = stored
            .into_par_iter()
            .map(
                |(video_path, timestamp_seconds, quality_score, frame_path, embedding)| {
                    let similarity = query
                        .iter()
                        .zip(&embedding)
                        .map(|(left, right)| left * right)
                        .sum();
                    SearchMatch {
                        video_path,
                        timestamp_seconds,
                        similarity,
                        quality_score,
                        frame_path,
                    }
                },
            )
            .collect();
        matches.sort_unstable_by(|left, right| right.similarity.total_cmp(&left.similarity));
        matches.truncate(limit);
        Ok(matches)
    }

    pub fn stats(&self) -> Result<IndexStats> {
        let videos = self
            .connection
            .query_row("SELECT COUNT(*) FROM videos", [], |row| {
                row.get::<_, i64>(0)
            })?;
        let frames = self
            .connection
            .query_row("SELECT COUNT(*) FROM frames", [], |row| {
                row.get::<_, i64>(0)
            })?;
        Ok(IndexStats {
            videos: videos as usize,
            frames: frames as usize,
        })
    }

    fn check_metadata(&self) -> Result<()> {
        for (key, expected) in [
            ("schema_version", SCHEMA_VERSION),
            ("embedding_model", MODEL_VERSION),
            ("embedding_dimension", "512"),
        ] {
            let stored: Option<String> = self
                .connection
                .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                    row.get(0)
                })
                .optional()?;
            match stored {
                Some(value) if value != expected => bail!(
                    "index metadata mismatch for {key}: found {value}, expected {expected}; use a new index directory"
                ),
                Some(_) => {}
                None => {
                    self.connection.execute(
                        "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                        params![key, expected],
                    )?;
                }
            }
        }
        Ok(())
    }
}

pub fn video_fingerprint(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect video {}", path.display()))?;
    let mut file =
        File::open(path).with_context(|| format!("failed to open video {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&metadata.len().to_le_bytes());
    if let Ok(modified) = metadata.modified()
        && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
    {
        hasher.update(&duration.as_nanos().to_le_bytes());
    }

    let prefix_size = metadata.len().min(FINGERPRINT_CHUNK_BYTES as u64) as usize;
    let mut buffer = vec![0_u8; prefix_size];
    file.read_exact(&mut buffer)?;
    hasher.update(&buffer);

    if metadata.len() > FINGERPRINT_CHUNK_BYTES as u64 {
        let suffix_size = metadata.len().min(FINGERPRINT_CHUNK_BYTES as u64) as usize;
        file.seek(SeekFrom::End(-(suffix_size as i64)))?;
        buffer.resize(suffix_size, 0);
        file.read_exact(&mut buffer)?;
        hasher.update(&buffer);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(embedding));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn blob_to_embedding(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() != EMBEDDING_DIMENSION * size_of::<f32>() {
        bail!(
            "stored embedding has {} bytes; expected {}",
            blob.len(),
            EMBEDDING_DIMENSION * size_of::<f32>()
        )
    }
    Ok(blob
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_blob_round_trip() {
        let input: Vec<f32> = (0..EMBEDDING_DIMENSION)
            .map(|index| index as f32 / EMBEDDING_DIMENSION as f32)
            .collect();
        assert_eq!(
            blob_to_embedding(&embedding_to_blob(&input)).unwrap(),
            input
        );
    }

    #[test]
    fn index_rejects_wrong_model_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let first = VideoIndex::open(directory.path()).unwrap();
        first
            .connection
            .execute(
                "UPDATE metadata SET value = 'wrong' WHERE key = 'embedding_model'",
                [],
            )
            .unwrap();
        drop(first);
        assert!(VideoIndex::open(directory.path()).is_err());
    }

    #[test]
    fn exact_search_ranks_largest_cosine_first() {
        let directory = tempfile::tempdir().unwrap();
        let mut index = VideoIndex::open(directory.path()).unwrap();
        let mut first = vec![0.0_f32; EMBEDDING_DIMENSION];
        first[0] = 1.0;
        let mut second = vec![0.0_f32; EMBEDDING_DIMENSION];
        second[1] = 1.0;
        index
            .replace_video(&IndexedVideo {
                path: PathBuf::from("video.mp4"),
                fingerprint: "abc".into(),
                duration_seconds: 2.0,
                width: 1920,
                height: 1080,
                frames: vec![
                    IndexedFrame {
                        timestamp_seconds: 0.0,
                        quality: FrameQuality::default(),
                        image_path: PathBuf::from("zero.jpg"),
                        embedding: first,
                    },
                    IndexedFrame {
                        timestamp_seconds: 1.0,
                        quality: FrameQuality::default(),
                        image_path: PathBuf::from("one.jpg"),
                        embedding: second,
                    },
                ],
            })
            .unwrap();

        let mut query = vec![0.0_f32; EMBEDDING_DIMENSION];
        query[1] = 1.0;
        let results = index.search(&query, 2).unwrap();
        assert_eq!(results[0].timestamp_seconds, 1.0);
        assert!(results[0].similarity > results[1].similarity);
    }
}
