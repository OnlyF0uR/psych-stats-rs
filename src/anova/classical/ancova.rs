use crate::core::{
    dataframe::{ColumnGroupNumericItem, DataFrame},
    errors::DatasetError,
};

#[derive(Debug)]
pub struct AncovaResult {
    pub f_stat: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub ms_between: f64,
    pub ms_within: f64,
    pub fac_name: String,
    pub dv_name: String,
}

// NOTE: TO BE HONEST I HAVE NO IDEA HOW ACCURATE THIS IMPLEMENTATION IS
// THE RESULTS APPEAR MEANINGFUL YET GIVE INDICATIONS OF BEING SLIGHTLY
// OFF. SO TESTS ARE NEEDED TO CONFIRM THE ACCURACY OF THIS IMPLEMENTATION
// AND TO MAKE SURE IT WORKS AS INTENDED.
pub fn ancova(
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
        let group_size = scores.len() as f64;

        // Calculate group mean for dependent variable scores
        let group_mean = scores.iter().sum::<f64>() / group_size;

        // Calculate SSB
        ssb += group_size * (group_mean - grand_mean).powi(2);

        // Calculate SSW for the dependent variable scores
        let group_ssw: f64 = scores
            .iter()
            .map(|&score| (score - group_mean).powi(2))
            .sum();
        ssw += group_ssw;
    }

    // Calculate degrees of freedom for the independent variable
    let df_between_iv = (group_levels.len() - 1) as f64; // k - 1
    let df_within_iv = dv_column.n() - group_levels.len(); // N - k

    // Calculate Mean Squares for the independent variable
    let ms_between_iv = ssb / df_between_iv;
    let ms_within_iv = ssw / df_within_iv as f64;

    // Calculate F-statistic for the independent variable
    let f_stat_iv = ms_between_iv / ms_within_iv;

    // Prepare result for the independent variable
    results.push(AncovaResult {
        f_stat: f_stat_iv,
        df_between: df_between_iv as usize,
        df_within: df_within_iv as usize,
        ms_between: ms_between_iv,
        ms_within: ms_within_iv,
        fac_name: independent_var.to_owned(),
        dv_name: dependent_var.to_owned(),
    });

    // ** Now calculate results for each covariate **
    for covariate in covariates {
        // Prepare the covariate matrix (including intercept)
        let covariate_values = covariate_items
            .iter()
            .find(|item| item.name == covariate)
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
        let coefficients = multiple_linear_regression(&transposed_matrix, &dv_scores)?;

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
        let mut ssb = 0.0; // Sum of Squares Between
        let mut ssw = 0.0; // Sum of Squares Within

        // Calculate SSB and SSW for the covariate
        for (_, scores) in group_levels.iter() {
            let group_size = scores.len() as f64;

            // Calculate group mean for adjusted scores
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
        let df_within_cov = dv_column.n() - group_levels.len(); // N - k (total observations - number of groups)

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
            ms_between: ms_between_cov,
            ms_within: ms_within_cov,
            fac_name: covariate.to_owned(),
            dv_name: dependent_var.to_owned(),
        };

        results.push(result); // Collect results
    }

    Ok(results) // Return a vector of results for each covariate and independent variable
}

// Simple Multiple Linear Regression Function
fn multiple_linear_regression(x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, DatasetError> {
    let n = x.len() as f64;
    let m = x[0].len();

    // Build the matrix for (X'X)
    let mut xtx = vec![vec![0.0; m]; m];
    let mut xty = vec![0.0; m];

    for i in 0..n as usize {
        for j in 0..m {
            for k in 0..m {
                xtx[j][k] += x[i][j] * x[i][k];
            }
            xty[j] += x[i][j] * y[i];
        }
    }

    // Solve for coefficients (X'X)^{-1}(X'y)
    let xtx_inv = invert_matrix(&xtx)?;
    let coefficients: Vec<f64> = xtx_inv
        .iter()
        .map(|row| row.iter().zip(&xty).map(|(a, b)| a * b).sum())
        .collect();

    Ok(coefficients)
}

// Function to invert a square matrix (Gauss-Jordan elimination)
fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, DatasetError> {
    let n = matrix.len();
    let mut augmented = matrix.to_vec();
    for i in 0..n {
        augmented[i].resize(2 * n, 0.0);
        augmented[i][i + n] = 1.0; // Create identity matrix
    }

    for i in 0..n {
        // Make the diagonal contain all 1s
        let divisor = augmented[i][i];
        if divisor == 0.0 {
            return Err(DatasetError::InvalidData(
                "Matrix is singular and cannot be inverted.".to_string(),
            ));
        }
        for j in 0..2 * n {
            augmented[i][j] /= divisor;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in 0..2 * n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix
    let inverse: Vec<Vec<f64>> = augmented.iter().map(|row| row[n..].to_vec()).collect();

    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::reader;

    #[test]
    fn test_ancova() {
        let df = reader::import_csv("samples/data3.csv").unwrap();

        let f_stats = ancova(&df, "condition", vec!["age", "happiness"], "score");
        println!("F-statistics: {:.6?}", f_stats);

        if let Ok(f_stat) = f_stats {
            assert_eq!(f_stat[0].f_stat, 1973.0238582946242);
        } else {
            panic!("Error in ANCOVA test");
        }
    }
}
