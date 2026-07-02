use anyhow::Result;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

pub(super) fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "create table if not exists vdr_embeddings (
            id integer primary key,
            image_id text not null,
            path text not null,
            sha256 text not null,
            kind text not null,
            model text not null,
            dimensions integer not null,
            content_hash text not null default '',
            vector_json text not null,
            active integer not null,
            indexed_at text not null
        );
        create index if not exists vdr_embeddings_lookup
            on vdr_embeddings(image_id, sha256, kind, model, dimensions, active);
        create index if not exists vdr_embeddings_content_lookup
            on vdr_embeddings(image_id, content_hash, kind, model, dimensions, active);
        create index if not exists vdr_embeddings_active
            on vdr_embeddings(active);",
    )?;
    migrate_content_hash(conn)
}

pub(super) fn content_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn parse_stored_vectors(vector_json: &str) -> Result<Vec<Vec<f32>>> {
    let value: serde_json::Value = serde_json::from_str(vector_json)?;
    match value.as_array().and_then(|items| items.first()) {
        Some(serde_json::Value::Array(_)) => Ok(serde_json::from_value(value)?),
        Some(_) => Ok(vec![serde_json::from_value(value)?]),
        None => Ok(Vec::new()),
    }
}

fn migrate_content_hash(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("pragma table_info(vdr_embeddings)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !columns.iter().any(|column| column == "content_hash") {
        conn.execute(
            "alter table vdr_embeddings add column content_hash text not null default ''",
            [],
        )?;
    }
    Ok(())
}
