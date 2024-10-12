use std::collections::HashMap;

use crate::core::{dataframe::DataFrame, errors::DatasetError};

pub fn ancova(
    _df: &DataFrame,
    _fixed_factors: Vec<&str>,
    _covariates: Vec<&str>,
    _dependent_var: &str,
) -> Result<HashMap<String, f64>, DatasetError> {
    // TODO: This
    Ok(HashMap::new())
}

// #[cfg(test)]
// mod tests {
//     use crate::core::reader;
//     use super::*;

//     #[test]
//     fn test_ancova() {
//         let df = reader::import_csv("samples/data3.csv").unwrap();

//         let f_stats = ancova(&df, vec!["condition"], vec!["age"], "score");
//         println!("F-statistics: {:?}", f_stats);

//         // Add appropriate assertions here
//         assert!(f_stats.contains_key("condition"));
//         assert!(f_stats.contains_key("age"));
//         assert!(!f_stats.values().any(|&v| v.is_nan()));
//     }
// }
