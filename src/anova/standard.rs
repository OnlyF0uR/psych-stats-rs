use crate::core::dataframe::DataFrame;

pub fn anova(df: &DataFrame, column_wl: Vec<&str>) -> f64 {
    let mut means: Vec<f64> = Vec::new();

    let mut grand_n = 0;
    for column in df.columns.iter() {
        if column_wl.contains(&column.name()) {
            if column.is_integer() || column.is_decimal() {
                means.push(column.mean());
                grand_n += column.n();
            } else {
                eprintln!("[ANOVA] Column {} is not a numeric type", column.name());
            }
        }
    }

    let mut grand_mean = means.iter().sum::<f64>() / means.len() as f64;

    let mut model_ss = 0.0; // SSM (Sum of Squares for Model)
    let mut error_ss = 0.0; // SSE (Sum of Squares for Error)
    for (i, column) in df.columns.iter().enumerate() {
        if column_wl.contains(&column.name()) {
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
    let df_model = column_wl.len() - 1;
    let ms_model = model_ss / df_model as f64;

    // Mean sum of squares for error
    let df_error = grand_n - column_wl.len();
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
        let df = reader::import_csv("samples/anova-prep.csv").unwrap();

        // compare 'much', 'mid', and 'little'
        let f_stat = anova(&df, vec!["much", "mid", "little"]);
        assert!(f_stat == 601.9580351962868);
    }
}
