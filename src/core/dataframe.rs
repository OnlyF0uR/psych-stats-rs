use std::any::Any;

pub trait ColumnOps {
    fn name(&self) -> &str;
    fn get_value(&self, index: usize) -> Box<dyn Any>;
    fn get_values(&self) -> Box<Vec<&dyn Any>>;
    fn set_values(&mut self, values: Vec<&dyn Any>);
    fn get_values_as_f64(&self) -> Vec<f64>;
    fn add_entry(&mut self, value: &dyn Any);
    fn mean(&self) -> f64;
    fn n(&self) -> usize;
    fn variance(&self) -> f64;
    fn standard_deviation(&self) -> f64;
    fn median(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn freq(&self, value: &dyn Any) -> usize;
    fn is_categorical(&self) -> bool;
    fn is_integer(&self) -> bool;
    fn is_decimal(&self) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct IntegerEntry {
    pub index: usize,
    pub value: i64,
}

pub struct IntegerColumn {
    pub name: String,
    pub data: Vec<IntegerEntry>,
}

impl ColumnOps for IntegerColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_value(&self, index: usize) -> Box<dyn Any> {
        Box::new(self.data[index].value)
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(
            self.data
                .iter()
                .map(|entry| &entry.value as &dyn Any)
                .collect(),
        )
    }

    fn get_values_as_f64(&self) -> Vec<f64> {
        self.data.iter().map(|entry| entry.value as f64).collect()
    }

    fn set_values(&mut self, values: Vec<&dyn Any>) {
        self.data = values
            .iter()
            .enumerate()
            .map(|(index, value)| IntegerEntry {
                index,
                value: *value.downcast_ref::<i64>().unwrap(),
            })
            .collect();
    }

    fn add_entry(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<i64>().unwrap();
        let entry = IntegerEntry {
            index: self.data.len(),
            value: *value,
        };
        self.data.push(entry);
    }

    fn mean(&self) -> f64 {
        let sum: i64 = self.data.iter().map(|entry| entry.value).sum();
        sum as f64 / self.data.len() as f64
    }

    fn n(&self) -> usize {
        self.data.len()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();
        self.data
            .iter()
            .map(|entry| (entry.value as f64 - mean).powi(2))
            .sum::<f64>()
            / self.data.len() as f64
    }

    fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    fn median(&self) -> f64 {
        let mut values: Vec<i64> = self.data.iter().map(|entry| entry.value).collect();
        values.sort();
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) as f64 / 2.0
        } else {
            values[mid] as f64
        }
    }

    fn min(&self) -> f64 {
        self.data.iter().map(|entry| entry.value).min().unwrap() as f64
    }

    fn max(&self) -> f64 {
        self.data.iter().map(|entry| entry.value).max().unwrap() as f64
    }

    fn freq(&self, value: &dyn Any) -> usize {
        let value = value.downcast_ref::<i64>().unwrap();
        self.data
            .iter()
            .filter(|entry| entry.value == *value)
            .count()
    }

    fn is_integer(&self) -> bool {
        true
    }

    fn is_decimal(&self) -> bool {
        false
    }

    fn is_categorical(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecimalEntry {
    pub index: usize,
    pub value: f64,
}

pub struct DecimalColumn {
    pub name: String,
    pub data: Vec<DecimalEntry>,
}

impl ColumnOps for DecimalColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_value(&self, index: usize) -> Box<dyn Any> {
        Box::new(self.data[index].value)
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(
            self.data
                .iter()
                .map(|entry| &entry.value as &dyn Any)
                .collect(),
        )
    }

    fn get_values_as_f64(&self) -> Vec<f64> {
        self.data.iter().map(|entry| entry.value).collect()
    }

    fn set_values(&mut self, values: Vec<&dyn Any>) {
        self.data = values
            .iter()
            .enumerate()
            .map(|(index, value)| DecimalEntry {
                index,
                value: *value.downcast_ref::<f64>().unwrap(),
            })
            .collect();
    }

    fn add_entry(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<f64>().unwrap();
        let entry = DecimalEntry {
            index: self.data.len(),
            value: *value,
        };
        self.data.push(entry);
    }

    fn mean(&self) -> f64 {
        let sum: f64 = self.data.iter().map(|entry| entry.value).sum();
        sum / self.data.len() as f64
    }

    fn n(&self) -> usize {
        self.data.len()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();
        self.data
            .iter()
            .map(|entry| (entry.value - mean).powi(2))
            .sum::<f64>()
            / self.data.len() as f64
    }

    fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    fn median(&self) -> f64 {
        let mut values: Vec<f64> = self.data.iter().map(|entry| entry.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    fn min(&self) -> f64 {
        self.data
            .iter()
            .map(|entry| entry.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn max(&self) -> f64 {
        self.data
            .iter()
            .map(|entry| entry.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn freq(&self, value: &dyn Any) -> usize {
        let value = value.downcast_ref::<f64>().unwrap();
        self.data
            .iter()
            .filter(|entry| entry.value == *value)
            .count()
    }

    fn is_integer(&self) -> bool {
        false
    }

    fn is_decimal(&self) -> bool {
        true
    }

    fn is_categorical(&self) -> bool {
        false
    }
}

pub struct CategoricalEntry {
    pub index: usize,
    pub value: String,
}

pub struct CategoricalColumn {
    pub name: String,
    pub data: Vec<String>,
}

impl ColumnOps for CategoricalColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_value(&self, index: usize) -> Box<dyn Any> {
        Box::new(self.data[index].clone()) // Clone to return owned value
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(self.data.iter().map(|entry| entry as &dyn Any).collect())
    }

    fn get_values_as_f64(&self) -> Vec<f64> {
        vec![] // Categorical values cannot be converted to f64
    }

    fn set_values(&mut self, values: Vec<&dyn Any>) {
        self.data = values
            .iter()
            .map(|value| value.downcast_ref::<String>().unwrap().clone())
            .collect();
    }

    fn add_entry(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<String>().unwrap().clone();
        self.data.push(value);
    }

    fn mean(&self) -> f64 {
        0.0 // Mean is not applicable to categorical data
    }
    fn n(&self) -> usize {
        self.data.len()
    }
    fn variance(&self) -> f64 {
        0.0 // Variance is not applicable to categorical data
    }
    fn standard_deviation(&self) -> f64 {
        0.0 // Standard deviation is not applicable to categorical data
    }
    fn median(&self) -> f64 {
        0.0 // Median is not applicable to categorical data
    }
    fn min(&self) -> f64 {
        0.0 // Min is not applicable to categorical data
    }
    fn max(&self) -> f64 {
        0.0 // Max is not applicable to categorical data
    }
    fn freq(&self, value: &dyn Any) -> usize {
        let value = value.downcast_ref::<String>().unwrap();
        self.data.iter().filter(|entry| *entry == value).count()
    }

    fn is_integer(&self) -> bool {
        false
    }

    fn is_decimal(&self) -> bool {
        false
    }

    fn is_categorical(&self) -> bool {
        true
    }
}

#[derive(Default)]
pub struct DataFrame {
    pub columns: Vec<Box<dyn ColumnOps>>,
}

impl DataFrame {
    pub fn new() -> DataFrame {
        DataFrame::default()
    }

    pub fn add_integer_column(&mut self, name: &str, data: Vec<i64>) {
        let mut entries = Vec::new();
        for (index, value) in data.iter().enumerate() {
            entries.push(IntegerEntry {
                index,
                value: *value,
            });
        }
        let column = IntegerColumn {
            name: name.to_string(),
            data: entries,
        };
        self.columns.push(Box::new(column));
    }

    pub fn add_decimal_column(&mut self, name: &str, data: Vec<f64>) {
        let mut entries = Vec::new();
        for (index, value) in data.iter().enumerate() {
            entries.push(DecimalEntry {
                index,
                value: *value,
            });
        }
        let column = DecimalColumn {
            name: name.to_string(),
            data: entries,
        };
        self.columns.push(Box::new(column));
    }

    pub fn add_categorical_column(&mut self, name: &str, data: Vec<String>) {
        let column = CategoricalColumn {
            name: name.to_string(),
            data,
        };
        self.columns.push(Box::new(column));
    }

    pub fn add_value_to_column_str(&mut self, column_name: &str, value: &dyn Any) {
        let column = self
            .columns
            .iter_mut()
            .find(|column| column.name() == column_name)
            .unwrap();

        if column.is_integer() {
            if let Some(value) = value.downcast_ref::<i64>() {
                column.add_entry(value);
            }
        } else if column.is_decimal() {
            if let Some(value) = value.downcast_ref::<f64>() {
                column.add_entry(value);
            }
        } else if let Some(value) = value.downcast_ref::<String>() {
            column.add_entry(value);
        }
    }

    pub fn add_value_to_column(&mut self, column_index: usize, value: &dyn Any) {
        let column = &mut self.columns[column_index];

        if column.is_integer() {
            if let Some(value) = value.downcast_ref::<i64>() {
                column.add_entry(value);
            }
        } else if column.is_decimal() {
            if let Some(value) = value.downcast_ref::<f64>() {
                column.add_entry(value);
            }
        } else if let Some(value) = value.downcast_ref::<String>() {
            column.add_entry(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_integer_column() {
        let mut df = DataFrame::new();
        df.add_integer_column("ages", vec![23, 45, 67, 23, 45]);
        assert_eq!(df.columns.len(), 1);
        assert_eq!(df.columns[0].name(), "ages");
    }

    #[test]
    fn test_add_decimal_column() {
        let mut df = DataFrame::new();
        df.add_decimal_column("salaries", vec![3500.0, 4600.0, 4900.0]);
        assert_eq!(df.columns.len(), 1);
        assert_eq!(df.columns[0].name(), "salaries");
    }

    #[test]
    fn test_add_categorical_column() {
        let mut df = DataFrame::new();
        df.add_categorical_column("departments", vec!["HR".to_string(), "IT".to_string()]);
        assert_eq!(df.columns.len(), 1);
        assert_eq!(df.columns[0].name(), "departments");
    }

    #[test]
    fn test_mean_integer_column() {
        let mut df = DataFrame::new();
        df.add_integer_column("ages", vec![23, 45, 67, 23, 45]);
        let mean = df.columns[0].mean();
        assert_eq!(mean, 40.6);
    }

    #[test]
    fn test_mean_decimal_column() {
        let mut df = DataFrame::new();
        df.add_decimal_column("salaries", vec![3500.0, 4600.0, 4900.0]);
        let mean = df.columns[0].mean();
        assert_eq!(mean, 4333.333333333333);
    }

    #[test]
    fn test_mean_categorical_column() {
        let mut df = DataFrame::new();
        df.add_categorical_column("departments", vec!["HR".to_string(), "IT".to_string()]);
        let mean = df.columns[0].mean(); // Should return 0 instead of panic
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_frequency() {
        let mut df = DataFrame::new();

        // Add test data
        df.add_integer_column("ages", vec![23, 45, 67, 23, 45]);
        df.add_decimal_column("salaries", vec![3500.0, 4600.0, 4900.0, 5100.0]);
        df.add_categorical_column(
            "departments",
            vec![
                "HR".to_string(),
                "IT".to_string(),
                "Marketing".to_string(),
                "HR".to_string(),
                "IT".to_string(),
            ],
        );

        // Test frequency for integer column
        let age_column = &df.columns[0];
        assert_eq!(age_column.freq(&23i64), 2);
        assert_eq!(age_column.freq(&45i64), 2);
        assert_eq!(age_column.freq(&67i64), 1);
        assert_eq!(age_column.freq(&99i64), 0); // Non-existent value

        // Test frequency for decimal column
        let salary_column = &df.columns[1];
        assert_eq!(salary_column.freq(&3500.0), 1);
        assert_eq!(salary_column.freq(&4600.0), 1);
        assert_eq!(salary_column.freq(&4900.0), 1);
        assert_eq!(salary_column.freq(&6000.0), 0); // Non-existent value

        // Test frequency for categorical column
        let department_column = &df.columns[2];
        assert_eq!(department_column.freq(&"HR".to_string()), 2);
        assert_eq!(department_column.freq(&"IT".to_string()), 2);
        assert_eq!(department_column.freq(&"Marketing".to_string()), 1);
        assert_eq!(department_column.freq(&"Finance".to_string()), 0); // Non-existent value
    }
}
