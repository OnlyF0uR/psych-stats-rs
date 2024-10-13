use crate::core::errors::DatasetError;

pub fn multiple_linear_regression(x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, DatasetError> {
    let n = x.len(); // Number of samples
    let m = x[0].len(); // Number of features

    // Validate dimensions
    if n == 0 || m == 0 || y.len() != n {
        return Err(DatasetError::InvalidData(
            "The input dimensions are not valid.".to_string(),
        ));
    }

    // Build the matrix for (X'X)
    let mut xtx = vec![vec![0.0; m]; m];
    let mut xty = vec![0.0; m];

    for i in 0..n {
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

// Function to invert a square matrix (using Gauss-Jordan elimination)
fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, DatasetError> {
    let n = matrix.len();

    // Check if the matrix is square
    if n == 0 || matrix[0].len() != n {
        return Err(DatasetError::InvalidData(
            "Matrix must be square to invert.".to_string(),
        ));
    }

    let mut augmented = matrix.to_vec();

    // Create augmented matrix [A | I]
    for i in 0..n {
        augmented[i].resize(2 * n, 0.0);
        augmented[i][i + n] = 1.0; // Create identity matrix
    }

    for i in 0..n {
        // Ensure the pivot element is not zero
        let divisor = augmented[i][i];
        if divisor.abs() < f64::EPSILON {
            return Err(DatasetError::InvalidData(
                "Matrix is singular and cannot be inverted.".to_string(),
            ));
        }
        for j in 0..2 * n {
            augmented[i][j] /= divisor; // Normalize pivot row
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
