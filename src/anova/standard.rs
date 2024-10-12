use crate::core::{
    dataframe::{ColumnType, DataFrame},
    errors::DatasetError,
};

pub fn anova(
    df: &DataFrame,
    fixed_factors: Vec<&str>,
    dependent_var: &str,
) -> Result<f64, DatasetError> {
    let dv_scores = df.cat_iv_levels(&fixed_factors, dependent_var)?;
    let (grand_mean, grand_n) = df.grand_descriptives(&[dependent_var])?;

    // Now we can calculate the sum of squares for the model
    let mut model_ss = 0.0;
    let mut error_ss = 0.0;
    for (_, group_values) in dv_scores.iter() {
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
    let df_model = dv_scores.len() - 1;
    let ms_model = model_ss / df_model as f64;

    // Mean sum of squares for error
    println!("Grand N: {}", grand_n);
    println!("DV Scores: {:?}", dv_scores.len());
    let df_error = grand_n - dv_scores.len();
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
            if column.column_type() == ColumnType::Numerical {
                // Loop through the column
                let group_mean = column.mean();
                let group_n = column.n();

                // Model sum of squares
                let ssb_g = (group_mean - grand_mean).powi(2) * group_n as f64; // TODO: Fix this
                model_ss += ssb_g;

                let group_values = column.get_values_as_f64()?;

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
        let values = df.columns[2].get_values_as_f64();
        println!("Values: {:?}", values);
        assert_eq!(df.columns[2].n(), 100);

        let f_stat = anova(&df, vec!["condition"], "score");
        println!("F-statistics: {:?}", f_stat);

        assert!(roughly_equals(f_stat.unwrap(), 1964.0831358347912, 1e-12));
    }

    // #[test]
    // fn test_anova_expl() {
    //     let df = reader::import_csv("samples/data2-raw.csv").unwrap();

    //     let f_stat = anova_expl(&df, vec!["much", "mid", "little"]).unwrap();
    //     assert!(roughly_equals(f_stat, 601.9580351962868, 1e-12));
    // }
}
