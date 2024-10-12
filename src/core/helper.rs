pub fn roughly_equals(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}
