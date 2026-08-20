use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(mut indices: Vec<u32>, mut values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            bail!(
                "sparse vector length mismatch: {} indices vs {} values",
                indices.len(),
                values.len()
            );
        }
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_by_key(|&i| indices[i]);
        let mut sorted_indices = Vec::with_capacity(indices.len());
        let mut sorted_values = Vec::with_capacity(values.len());
        let mut last: Option<u32> = None;
        for i in order {
            let index = indices[i];
            if last == Some(index) {
                let acc = sorted_values.last_mut().expect("duplicate has prior value");
                *acc += values[i];
                continue;
            }
            last = Some(index);
            sorted_indices.push(index);
            sorted_values.push(values[i]);
        }
        indices = sorted_indices;
        values = sorted_values;
        Ok(Self { indices, values })
    }

    pub fn dot(&self, other: &Self) -> f64 {
        let mut i = 0;
        let mut j = 0;
        let mut acc = 0.0_f64;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    acc += f64::from(self.values[i]) * f64::from(other.values[j]);
                    i += 1;
                    j += 1;
                }
            }
        }
        acc
    }
}

pub fn parse_sparse_vector(value: &serde_json::Value) -> Result<SparseVector> {
    let Some(object) = value.as_object() else {
        bail!("sparse embedding must be an object with indices and values");
    };
    let indices: Vec<u32> = match object.get("indices") {
        Some(indices) => serde_json::from_value(indices.clone())?,
        None => bail!("sparse embedding missing indices"),
    };
    let values: Vec<f32> = match object.get("values") {
        Some(values) => serde_json::from_value(values.clone())?,
        None => bail!("sparse embedding missing values"),
    };
    SparseVector::new(indices, values)
}

pub fn rrf_score(rank: usize) -> f64 {
    1.0 / (60.0 + rank as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_dot_scores_overlapping_terms() {
        let query = SparseVector::new(vec![1, 3], vec![1.0, 2.0]).unwrap();
        let document = SparseVector::new(vec![3, 8], vec![0.5, 9.0]).unwrap();
        assert_eq!(query.dot(&document), 1.0);
    }

    #[test]
    fn sparse_dot_is_zero_without_overlap() {
        let left = SparseVector::new(vec![1], vec![4.0]).unwrap();
        let right = SparseVector::new(vec![2], vec![4.0]).unwrap();
        assert_eq!(left.dot(&right), 0.0);
    }

    #[test]
    fn rrf_prefers_earlier_ranks() {
        assert!(rrf_score(0) > rrf_score(1));
        assert_eq!(rrf_score(0), 1.0 / 61.0);
    }
}
