use std::collections::HashMap;

use crate::core::dataframe::DataFrame;

pub fn anova(df: &DataFrame, fixed_factors: Vec<&str>, dependent_var: &str) -> f64 {
    let dependent_column = match df.columns.iter().find(|&x| x.name() == dependent_var) {
        Some(column) => column,
        None => {
            eprintln!("[ANOVA] Column {} not found", dependent_var);
            return 0.0;
        }
    };

    if !dependent_column.is_integer() && !dependent_column.is_decimal() {
        eprintln!("[ANOVA] Column {} is not a numeric type", dependent_var);
        return 0.0;
    }

    let grand_mean = dependent_column.mean();
    let grand_n = dependent_column.n();

    // Now we need to split the data into groups based on the fixed factors
    // we can do this by creating a hashmap with the group name as the key
    // and the values as the column values

    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
    for (_, column) in df.columns.iter().enumerate() {
        if fixed_factors.contains(&column.name()) {
            if column.is_categorical() {
                let c_name = column.name();

                // now loop through the values
                let c_values: Vec<String> = column
                    .get_values()
                    .iter()
                    .filter_map(|x| {
                        if let Some(value) = x.downcast_ref::<String>() {
                            Some(value.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                for i in 0..c_values.len() {
                    let value = &c_values[i];
                    let join_name = format!("{}-{}", c_name, value);

                    // If the value is not in the hashmap, add it
                    if !groups.contains_key(&join_name) {
                        groups.insert(join_name.clone(), vec![]);
                    }

                    // Now we need to collect the values for this group, which we can do
                    // by taking in current index and getting the values from the dependent column
                    let v = match dependent_column.get_value(i) {
                        Ok(v) => match v.downcast_ref::<f64>() {
                            Some(v) => *v,
                            None => {
                                eprintln!("[ANOVA] Could not downcast value to f64");
                                continue;
                            }
                        },
                        Err(e) => {
                            eprintln!("[ANOVA] Could not get value from column, invalid relative value: {}", e);
                            continue;
                        }
                    };

                    // Now we can push the value to the group
                    groups.get_mut(&join_name).unwrap().push(v);
                }
            } else {
                eprintln!("[ANOVA] Column {} is not a categorical type", column.name());
            }
        }
    }

    // Now we can calculate the sum of squares for the model
    let mut model_ss = 0.0;
    let mut error_ss = 0.0;
    for (_, group_values) in groups.iter() {
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
    let df_model = groups.len() - 1;
    let ms_model = model_ss / df_model as f64;

    // Mean sum of squares for error
    let df_error = grand_n - groups.len();
    let ms_error = error_ss / df_error as f64;

    // F-statistic
    let f_stat = ms_model / ms_error;
    f_stat
}

pub fn anova_expl(df: &DataFrame, dependent_vars: Vec<&str>) -> f64 {
    let (grand_mean, grand_n) = df.grand_descriptives(&dependent_vars);

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
    f_stat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::reader;

    #[test]
    fn test_anova() {
        let df = reader::import_csv("samples/data3.csv").unwrap();

        let f_stat = anova(&df, vec!["condition"], "score");
        assert!((f_stat - 1964.0831358347912).abs() < 1e-12);
    }

    #[test]
    fn test_anova_expl() {
        let df = reader::import_csv("samples/data2-raw.csv").unwrap();

        let f_stat = anova_expl(&df, vec!["much", "mid", "little"]);
        assert!((f_stat - 601.9580351962868).abs() < 1e-12);
    }
}
