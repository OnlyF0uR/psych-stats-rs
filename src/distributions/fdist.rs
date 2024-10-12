pub fn p_value(f_value: f64, numerator_dof: usize, denominator_dof: usize) -> f64 {
    1.0 - f_cdf(f_value, numerator_dof, denominator_dof)
}

fn f_cdf(f_value: f64, numerator_dof: usize, denominator_dof: usize) -> f64 {
    let x = (numerator_dof as f64 * f_value)
        / (numerator_dof as f64 * f_value + denominator_dof as f64);
    incomplete_beta(x, numerator_dof as f64 / 2.0, denominator_dof as f64 / 2.0)
}

fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return x;
    }

    let max_iterations = 1000;
    let epsilon = 1e-8;

    let factor =
        (gamma_ln(a + b) - gamma_ln(a) - gamma_ln(b)).exp() * x.powf(a) * (1.0 - x).powf(b) / a;

    let mut result = 0.0;
    let mut term = 1.0;
    let mut n = 1;

    while n < max_iterations {
        term *= (n as f64 - 1.0 + a + b) * x / (n as f64 + a);
        result += term;
        if term < epsilon {
            break;
        }
        n += 1;
    }

    factor * result
}

fn gamma_ln(x: f64) -> f64 {
    let coefficient = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        1.208_650_973_866_179e-3,
        -0.5395239384953e-5,
    ];

    let y = x;
    let temp = x + 5.5;
    let temp = (x + 0.5) * temp.ln() - temp;
    let mut sum = 1.000000000190015;

    for (j, &coeff) in coefficient.iter().enumerate() {
        sum += coeff / (y + (j + 1) as f64);
    }

    temp + (2.5066282746310005 * sum / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_value() {
        let f_value = 3.0;
        let numerator_dof = 2;
        let denominator_dof = 3;
        let p_value_value = p_value(f_value, numerator_dof, denominator_dof);
        assert!(p_value_value == 0.38490018281897276);
    }
}
