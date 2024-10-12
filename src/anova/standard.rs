use std::collections::HashMap;

use crate::core::{dataframe::DataFrame, errors::DatasetError};

pub fn anova(
    df: &DataFrame,
    fixed_factors: Vec<&str>,
    dependent_var: &str,
) -> Result<f64, DatasetError> {
    let dependent_column = match df.columns.iter().find(|&x| x.name() == dependent_var) {
        Some(column) => column,
        None => {
            return Err(DatasetError::ColumnNotFound(dependent_var.to_string()));
        }
    };

    if !dependent_column.is_integer() && !dependent_column.is_decimal() {
        return Err(DatasetError::ColumnTypeMismatch(
            dependent_var.to_string(),
            "numeric".to_owned(),
        ));
    }

    let grand_mean = dependent_column.mean();
    let grand_n = dependent_column.n();

    // Now we need to split the data into groups based on the fixed factors
    // we can do this by creating a hashmap with the group name as the key
    // and the values as the column values

    let mut factor_levels: HashMap<String, Vec<f64>> = HashMap::new();
    for factor in &fixed_factors {
        let column = match df.columns.iter().find(|&x| x.name() == *factor) {
            Some(column) => column,
            None => {
                return Err(DatasetError::ColumnNotFound(factor.to_string()));
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
            return Err(DatasetError::ColumnTypeMismatch(
                column.name().to_owned(),
                "categorical".to_owned(),
            ));
        }
    }

    // Now we can calculate the sum of squares for the model
    let mut model_ss = 0.0;
    let mut error_ss = 0.0;
    for (_, group_values) in factor_levels.iter() {
        let group_mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
        let group_n = group_values.len() as f64;

        let ssb_g = (group_mean - grand_mean).powi(2) * group_n;
        model_ss += ssb_g;

        let sse_g = group_values
            .iter()
            .map(|x| (x - group_mean).powi(2))
            .sum::<f64>();

        error_ss += sse_g;
    }

    // Mean sum of squares for model
    let df_model = factor_levels.len() - 1;
    let ms_model = model_ss / df_model as f64;

    // Mean sum of squares for error
    let df_error = grand_n - factor_levels.len();
    let ms_error = error_ss / df_error as f64;

    // F-statistic
    let f_stat = ms_model / ms_error;
    Ok(f_stat)
}

pub fn anova_expl(df: &DataFrame, dependent_vars: Vec<&str>) -> Result<f64, DatasetError> {
    let (grand_mean, grand_n) = df.grand_descriptives(&dependent_vars)?;

    let mut model_ss = 0.0; // SSM (Sum of Squares for Model)
    let mut error_ss = 0.0; // SSE (Sum of Squares for Error)
    for (_, column) in df.columns.iter().enumerate() {
        if dependent_vars.contains(&column.name()) {
            if column.is_integer() || column.is_decimal() {
                // Loop through the column
                let group_mean = column.mean();
                let group_n = column.n();

                // Model sum of squares
                let ssb_g = (group_mean - grand_mean).powi(2) * group_n as f64; // TODO: Fix this
                model_ss += ssb_g;

                let group_values = column.get_values_as_f64();

                let sse_g = group_values
                    .iter()
                    .map(|x| (x - group_mean).powi(2))
                    .sum::<f64>();

                error_ss += sse_g;
            } else {
                eprintln!("[ANOVA] Column {} is not a numeric type", column.name());
            }
        }
    }

    // Mean sum of squares for model
    let df_model = dependent_vars.len() - 1;
    let ms_model = model_ss / df_model as f64;

    // Mean sum of squares for error
    let df_error = grand_n - dependent_vars.len();
    let ms_error = error_ss / df_error as f64;

    // F-statistic
    let f_stat = ms_model / ms_error;
    Ok(f_stat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{helper::roughly_equals, reader};

    #[test]
    fn test_anova() {
        let df = reader::import_csv("samples/data3.csv").unwrap();

        let f_stat = anova(&df, vec!["condition"], "score").unwrap();
        assert!(roughly_equals(f_stat, 1964.0831358347912, 1e-12));
    }

    #[test]
    fn test_anova_expl() {
        let df = reader::import_csv("samples/data2-raw.csv").unwrap();

        let f_stat = anova_expl(&df, vec!["much", "mid", "little"]).unwrap();
        assert!(roughly_equals(f_stat, 601.9580351962868, 1e-12));
    }
}
