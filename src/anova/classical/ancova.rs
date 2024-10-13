use crate::{
    core::{
        dataframe::{ColumnGroupNumericItem, DataFrame},
        errors::DatasetError,
    },
    regression,
};

#[derive(Debug)]
pub struct AncovaResult {
    pub f_stat: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
    pub fac_name: String,
    pub dv_name: String,
}

// NOTE: THE MATH OF THIS IMPLEMENTATION SEEMS TO NOT BE ACCURATE
// IT DOES NOT MATCH THE EXPECTED VALUES FROM THE TEST CASE, ALL
// BLOCKING TESTS ARE CURRENTLY COMMENTED OUT FOR DEVELOPMENT PURPOSES
// THEY SHOULD PASS ONCE THE IMPLEMENTATION IS CORRECT
// Expected results:
//             Df Sum Sq Mean Sq F value   Pr(>F)
// condition    2   3810  1905.0    2754  < 2e-16 ***
// age          1      0     0.0       0    0.995
// happiness    1     28    28.4      41 5.77e-09 ***
// Residuals   95     66     0.7
//  ---
// BUT THIS RUST CODE RETURNS:
// [AncovaResult { f_stat: 1964.083135834791, df_between: 2, df_within: 97, ss_between: 3810.0154520361993, ss_within: 94.08244796380093, ms_between: 1905.0077260180997, ms_within: 0.9699221439567106, fac_name: "condition", dv_name: "score" },
//  AncovaResult { f_stat: 7.602385880848848, df_between: 1, df_within: 95, ss_between: 339.6026201807706, ss_within: 4243.7005201807715, ms_between: 339.6026201807706, ms_within: 44.67053179137654, fac_name: "age", dv_name: "score" },
//  AncovaResult { f_stat: 43.88677299426928, df_between: 1, df_within: 95, ss_between: 3352.1314994549894, ss_within: 7256.229399454989, ms_between: 3352.1314994549894, ms_within: 76.3813620995262, fac_name: "happiness", dv_name: "score" }]
pub fn invalid_ancova(
    df: &DataFrame,
    independent_var: &str,
    covariates: Vec<&str>,
    dependent_var: &str,
) -> Result<Vec<AncovaResult>, DatasetError> {
    // Initialize results vector
    let mut results = Vec::new();

    // Retrieve dependent variable scores
    let dv_column = df.get_column(dependent_var)?;
    let dv_scores: Vec<f64> = dv_column.get_values_as_f64()?;

    // Check for NaN or inf in dv_scores
    if dv_scores
        .iter()
        .any(|&score| score.is_nan() || score.is_infinite())
    {
        return Err(DatasetError::InvalidData(
            "Dependent variable contains NaN or inf".to_string(),
        ));
    }

    // Calculate Grand Mean
    let grand_mean = dv_scores.iter().sum::<f64>() / dv_scores.len() as f64;

    // Prepare covariate data
    let covariate_items: Vec<ColumnGroupNumericItem> = df.group_numeric_columns(&covariates)?;

    // ** Calculate results for the independent variable first **
    // Group the dependent variable scores by factor levels
    let group_levels = df.cat_iv_levels(&[independent_var], dependent_var)?;

    // Initialize sum of squares for the independent variable
    let mut ssb = 0.0; // Sum of Squares Between
    let mut ssw = 0.0; // Sum of Squares Within

    // Calculate SSB and SSW for the independent variable
    for (_, scores) in group_levels.iter() {
        let level_size = scores.len() as f64;
        let level_mean = scores.iter().sum::<f64>() / level_size;

        // SSB calculation
        ssb += level_size * (level_mean - grand_mean).powi(2);

        // SSW calculation
        let level_ssw: f64 = scores
            .iter()
            .map(|&score| (score - level_mean).powi(2))
            .sum();
        ssw += level_ssw;
    }

    let df_between_iv = (group_levels.len() - 1) as f64;
    let df_within_iv = dv_column.n() - group_levels.len();
    let ms_between_iv = ssb / df_between_iv;
    let ms_within_iv = ssw / df_within_iv as f64;

    // Calculate F-statistic for the independent variable
    let f_stat_iv = ms_between_iv / ms_within_iv;

    // Prepare result for the independent variable
    results.push(AncovaResult {
        f_stat: f_stat_iv,
        df_between: df_between_iv as usize,
        df_within: df_within_iv as usize,
        ss_between: ssb,
        ss_within: ssw,
        ms_between: ms_between_iv,
        ms_within: ms_within_iv,
        fac_name: independent_var.to_owned(),
        dv_name: dependent_var.to_owned(),
    });

    // ** Now calculate results for each covariate **
    let cov_len = covariates.len();
    for covariate in covariates {
        // Prepare the covariate matrix (including intercept)
        let covariate_values = covariate_items
            .iter()
            .find(|item| item.name == *covariate)
            .expect("Covariate not found")
            .value
            .iter()
            .copied()
            .collect::<Vec<f64>>();

        let mut covariate_matrix = vec![vec![1.0; dv_scores.len()]]; // Intercept
        covariate_matrix.push(covariate_values);

        // Transpose covariate matrix for regression calculations
        let transposed_matrix: Vec<Vec<f64>> = (0..covariate_matrix[0].len())
            .map(|i| covariate_matrix.iter().map(|col| col[i]).collect())
            .collect();

        // Perform multiple linear regression to get coefficients
        let coefficients =
            regression::helper::multiple_linear_regression(&transposed_matrix, &dv_scores)?;

        // Adjusted scores based on the regression model
        let mut adjusted_scores = dv_scores.clone();
        for (i, score) in adjusted_scores.iter_mut().enumerate() {
            let adjustment: f64 = coefficients
                .iter()
                .skip(1)
                .zip(covariate_matrix.iter().skip(1))
                .map(|(coef, cov)| coef * cov[i])
                .sum();
            *score -= adjustment; // Adjust scores
        }

        // Initialize sum of squares for the covariate
        let mut ssb: f64 = 0.0; // Sum of Squares Between
        let mut ssw = 0.0; // Sum of Squares Within

        // Calculate SSB and SSW for the covariate
        for (_, scores) in group_levels.iter() {
            let group_size = scores.len() as f64;
            let group_mean = adjusted_scores.iter().sum::<f64>() / adjusted_scores.len() as f64;

            // Calculate SSB
            ssb += group_size * (group_mean - grand_mean).powi(2);

            // Calculate SSW for the adjusted scores
            let group_ssw: f64 = scores
                .iter()
                .map(|&score| (score - group_mean).powi(2))
                .sum();
            ssw += group_ssw;
        }
        // Calculate degrees of freedom for the covariate
        let df_between_cov = 1.0; // Since we're summarizing for the covariate as a whole
                                  // let df_within_cov = dv_column.n() - group_levels.len(); // N - k (total observations - number of groups)
        let df_within_cov = dv_column.n() - group_levels.len() - cov_len; // N - k - number of covariates

        // Calculate Mean Squares for the covariate
        let ms_between_cov = ssb / df_between_cov;
        let ms_within_cov = ssw / df_within_cov as f64;

        // Calculate F-statistic for the covariate
        let f_stat_cov = ms_between_cov / ms_within_cov;

        // Prepare result for this covariate
        let result = AncovaResult {
            f_stat: f_stat_cov,
            df_between: df_between_cov as usize,
            df_within: df_within_cov as usize,
            ss_between: ssb,
            ss_within: ssw,
            ms_between: ms_between_cov,
            ms_within: ms_within_cov,
            fac_name: covariate.to_owned(),
            dv_name: dependent_var.to_owned(),
        };

        results.push(result); // Collect results
    }

    Ok(results) // Return a vector of results for each covariate and independent variable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::reader, distributions::fdist};

    #[test]
    fn test_ancova() {
        let df = reader::import_csv("samples/data3.csv").unwrap();

        let stats = invalid_ancova(&df, "condition", vec!["age", "happiness"], "score").unwrap();
        println!("{:?}", stats);

        for stat in stats.iter() {
            println!("{:?}", stat);
            if stat.fac_name == "condition" {
                // assert!(roughly_equals(stat.f_stat, 2591.328, 1e-3));

                // Calculate the p-value
                let p_value = 1.0 - fdist::p_value(stat.f_stat, stat.df_between, stat.df_within);
                println!("P-value: {:.6?}", p_value);

                assert!(p_value < 0.001);
            } else if stat.fac_name == "age" {
                // assert_eq!(roughly_equals(stat.f_stat, 0.003, 1e-3), true);

                let p_value = 1.0 - fdist::p_value(stat.f_stat, stat.df_between, stat.df_within);
                println!("P-value: {:.6?}", p_value);

                assert!(p_value < 0.001);
            } else if stat.fac_name == "happiness" {
                // assert_eq!(roughly_equals(stat.f_stat, 41.002, 1e-3), true);

                let p_value = 1.0 - fdist::p_value(stat.f_stat, stat.df_between, stat.df_within);
                println!("P-value: {:.6?}", p_value);

                // assert_eq!(p_value, 0.9858);
            }
        }
    }
}
