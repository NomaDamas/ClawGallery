use crate::{
    DEFAULT_DIMENSIONS, ImageDocument, SimilarImageDuplicate, SimilarImageGroup, client, index,
    search, store,
};
use anyhow::Result;
use std::{collections::HashMap, path::Path};

pub(super) fn similar_image_groups(
    db_path: &Path,
    images: &[ImageDocument],
    threshold: f64,
) -> Result<Vec<SimilarImageGroup>> {
    let conn = store::open_store(db_path)?;
    let index_config =
        index::latest_active_index_config(&conn)?.unwrap_or_else(|| index::ActiveIndexConfig {
            model: client::default_model().to_string(),
            dimensions: DEFAULT_DIMENSIONS,
        });
    let active_images: HashMap<String, ImageDocument> = images
        .iter()
        .map(|image| (image.id.clone(), image.clone()))
        .collect();
    let mut vectors: Vec<_> = store::active_vectors(
        &conn,
        &active_images,
        &index_config.model,
        index_config.dimensions,
    )?
    .into_iter()
    .filter(|stored| stored.kind.is_image())
    .collect();
    vectors.sort_by(|left, right| {
        active_images[&left.image_id]
            .path
            .cmp(&active_images[&right.image_id].path)
    });
    let mut parents: Vec<usize> = (0..vectors.len()).collect();
    let mut scores: HashMap<(usize, usize), f64> = HashMap::new();
    for left in 0..vectors.len() {
        for right in (left + 1)..vectors.len() {
            let forward =
                search::late_interaction_score(&vectors[left].vectors, &vectors[right].vectors)?;
            let backward =
                search::late_interaction_score(&vectors[right].vectors, &vectors[left].vectors)?;
            let score = (forward + backward) / 2.0;
            if score >= threshold {
                union(&mut parents, left, right);
                scores.insert((left, right), score);
            }
        }
    }
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..vectors.len() {
        components
            .entry(find(&mut parents, index))
            .or_default()
            .push(index);
    }
    let mut groups = Vec::new();
    for component in components.values_mut() {
        if component.len() < 2 {
            continue;
        }
        component.sort_by(|left, right| {
            active_images[&vectors[*left].image_id]
                .path
                .cmp(&active_images[&vectors[*right].image_id].path)
        });
        let representative = component[0];
        let duplicates = component[1..]
            .iter()
            .map(|index| SimilarImageDuplicate {
                image_id: vectors[*index].image_id.clone(),
                score: best_component_score(*index, component, &scores),
            })
            .collect();
        groups.push(SimilarImageGroup {
            representative_id: vectors[representative].image_id.clone(),
            duplicates,
        });
    }
    groups.sort_by(|left, right| {
        active_images[&left.representative_id]
            .path
            .cmp(&active_images[&right.representative_id].path)
    });
    Ok(groups)
}

fn best_component_score(
    target: usize,
    component: &[usize],
    scores: &HashMap<(usize, usize), f64>,
) -> f64 {
    component
        .iter()
        .filter(|index| **index != target)
        .filter_map(|index| {
            let key = if *index < target {
                (*index, target)
            } else {
                (target, *index)
            };
            scores.get(&key).copied()
        })
        .fold(0.0, f64::max)
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find(parents, left);
    let right_root = find(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn find(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = find(parents, parents[index]);
    }
    parents[index]
}
