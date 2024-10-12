use std::collections::HashMap;

use crate::core::dataframe::DataFrame;

pub fn ancova(
    df: &DataFrame,
    fixed_factors: Vec<&str>,
    covariates: Vec<&str>,
    dependent_var: &str,
) -> HashMap<String, f64> {
    let dependent_column = match df.columns.iter().find(|&x| x.name() == dependent_var) {
        Some(column) => column,
        None => {
            eprintln!("[ANCOVA] Column {} not found", dependent_var);
            return HashMap::new();
        }
    };

    if !dependent_column.is_integer() && !dependent_column.is_decimal() {
        eprintln!("[ANCOVA] Column {} is not a numeric type", dependent_var);
        return HashMap::new();
    }

    let grand_mean = dependent_column.mean();
    let grand_n = dependent_column.n();

    // Collect means for covariates
    let mut covariate_means: HashMap<String, f64> = HashMap::new();

    for covariate in &covariates {
        let covariate_column = match df.columns.iter().find(|&x| x.name() == *covariate) {
            Some(column) => column,
            None => {
                eprintln!("[ANCOVA] Covariate column {} not found", covariate);
                return HashMap::new();
            }
        };

        if !covariate_column.is_integer() && !covariate_column.is_decimal() {
            eprintln!("[ANCOVA] Column {} is not a numeric type", covariate);
            return HashMap::new();
        }

        let covariate_mean = covariate_column.mean();
        covariate_means.insert(covariate.to_string(), covariate_mean);
    }

    // Group values by fixed factors
    let mut factor_levels: HashMap<String, Vec<f64>> = HashMap::new();
    for factor in &fixed_factors {
        let column = match df.columns.iter().find(|&x| x.name() == *factor) {
            Some(column) => column,
            None => {
                eprintln!("[ANCOVA] Fixed factor column {} not found", factor);
                return HashMap::new();
            }
        };

        if column.is_categorical() {
            let c_values: Vec<String> = column
                .get_values()
                .iter()
                .filter_map(|x| x.downcast_ref::<String>().map(|v| v.clone()))
                .collect();

            for i in 0..c_values.len() {
                let value = &c_values[i];
                let group_name = format!("{}-{}", factor, value);

                factor_levels
                    .entry(group_name.clone())
                    .or_insert_with(Vec::new);

                // Collect dependent variable values for the group
                if let Some(value) = dependent_column.get_value(i).ok() {
                    if let Some(f_value) = value.downcast_ref::<f64>() {
                        factor_levels.get_mut(&group_name).unwrap().push(*f_value);
                    }
                }
            }
        } else {
            eprintln!(
                "[ANCOVA] Column {} is not a categorical type",
                column.name()
            );
            return HashMap::new();
        }
    }

    // TODO: This

    HashMap::new()
}

#[cfg(test)]
mod tests {
    use crate::core::reader;

    use super::*;

    #[test]
    fn test_ancova() {
        let df = reader::import_csv("samples/data3.csv").unwrap();

        let f_stats = ancova(&df, vec!["condition"], vec!["age"], "score");
        println!("F-statistics: {:?}", f_stats);

        // Add appropriate assertions here
        assert!(f_stats.contains_key("condition"));
        assert!(f_stats.contains_key("age"));
        assert!(!f_stats.values().any(|&v| v.is_nan()));
    }
}
