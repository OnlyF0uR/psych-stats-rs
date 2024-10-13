use std::{any::Any, collections::HashMap};

use super::errors::DatasetError;

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Numerical,
    Categorical,
    Binary,
}

impl ColumnType {
    pub fn as_str(&self) -> &str {
        match self {
            ColumnType::Numerical => "Numerical",
            ColumnType::Categorical => "Categorical",
            ColumnType::Binary => "Binary",
        }
    }
}

pub trait ColumnOps {
    fn name(&self) -> &str;
    fn get_value(&self, index: usize) -> Result<Box<dyn Any>, DatasetError>;
    fn get_values(&self) -> Box<Vec<&dyn Any>>;
    fn set_values(&mut self, values: Vec<&dyn Any>);
    fn get_values_as_f64(&self) -> Result<Vec<f64>, DatasetError>;
    fn get_values_as_str(&self) -> Result<Vec<String>, DatasetError>;
    fn add_entry(&mut self, value: &dyn Any);
    fn mean(&self) -> f64;
    fn n(&self) -> usize;
    fn variance(&self) -> f64;
    fn standard_deviation(&self) -> f64;
    fn median(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn freq(&self, value: &dyn Any) -> usize;
    fn column_type(&self) -> ColumnType;
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryEntry {
    pub index: usize,
    pub value: bool,
}

pub struct BinaryColumn {
    pub name: String,
    pub data: Vec<BinaryEntry>,
}

impl ColumnOps for BinaryColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_value(&self, index: usize) -> Result<Box<dyn Any>, DatasetError> {
        if index >= self.data.len() {
            return Err(DatasetError::EmptyValue(index, self.data.len()));
        }
        Ok(Box::new(self.data[index].value))
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(
            self.data
                .iter()
                .map(|entry| &entry.value as &dyn Any)
                .collect(),
        )
    }

    fn get_values_as_f64(&self) -> Result<Vec<f64>, DatasetError> {
        return Err(DatasetError::ColumnTypeMismatch(
            self.name().to_owned(),
            ColumnType::Numerical,
        ));
    }

    fn get_values_as_str(&self) -> Result<Vec<String>, DatasetError> {
        return Err(DatasetError::ColumnTypeMismatch(
            self.name().to_owned(),
            ColumnType::Categorical,
        ));
    }

    fn set_values(&mut self, values: Vec<&dyn Any>) {
        self.data = values
            .iter()
            .enumerate()
            .map(|(index, value)| BinaryEntry {
                index,
                value: *value.downcast_ref::<bool>().unwrap(),
            })
            .collect();
    }

    fn add_entry(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<bool>().unwrap();
        let entry = BinaryEntry {
            index: self.data.len(),
            value: *value,
        };
        self.data.push(entry);
    }

    fn mean(&self) -> f64 {
        0.0
    }

    fn n(&self) -> usize {
        self.data.len()
    }

    fn variance(&self) -> f64 {
        0.0
    }

    fn standard_deviation(&self) -> f64 {
        0.0
    }

    fn median(&self) -> f64 {
        0.0
    }

    fn min(&self) -> f64 {
        self.data.iter().map(|entry| entry.value).min().unwrap() as i8 as f64
    }

    fn max(&self) -> f64 {
        self.data.iter().map(|entry| entry.value).max().unwrap() as i8 as f64
    }

    fn freq(&self, value: &dyn Any) -> usize {
        let value = value.downcast_ref::<bool>().unwrap();
        self.data
            .iter()
            .filter(|entry| entry.value == *value)
            .count()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Binary
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NumericalEntry {
    pub index: usize,
    pub value: f64,
}

pub struct NumericalColumn {
    pub name: String,
    pub data: Vec<NumericalEntry>,
}

impl ColumnOps for NumericalColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_value(&self, index: usize) -> Result<Box<dyn Any>, DatasetError> {
        if index >= self.data.len() {
            return Err(DatasetError::EmptyValue(index, self.data.len()));
        }
        Ok(Box::new(self.data[index].value))
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(
            self.data
                .iter()
                .map(|entry| &entry.value as &dyn Any)
                .collect(),
        )
    }

    fn get_values_as_f64(&self) -> Result<Vec<f64>, DatasetError> {
        Ok(self.data.iter().map(|entry| entry.value).collect())
    }

    fn get_values_as_str(&self) -> Result<Vec<String>, DatasetError> {
        return Err(DatasetError::ColumnTypeMismatch(
            self.name().to_owned(),
            ColumnType::Categorical,
        ));
    }

    fn set_values(&mut self, values: Vec<&dyn Any>) {
        self.data = values
            .iter()
            .enumerate()
            .map(|(index, value)| NumericalEntry {
                index,
                value: *value.downcast_ref::<f64>().unwrap(),
            })
            .collect();
    }

    fn add_entry(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<f64>().unwrap();
        let entry = NumericalEntry {
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

    fn column_type(&self) -> ColumnType {
        ColumnType::Numerical
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

    fn get_value(&self, index: usize) -> Result<Box<dyn Any>, DatasetError> {
        if index >= self.data.len() {
            return Err(DatasetError::EmptyValue(index, self.data.len()));
        }
        Ok(Box::new(self.data[index].clone()))
    }

    fn get_values(&self) -> Box<Vec<&dyn Any>> {
        Box::new(self.data.iter().map(|entry| entry as &dyn Any).collect())
    }

    fn get_values_as_f64(&self) -> Result<Vec<f64>, DatasetError> {
        return Err(DatasetError::ColumnTypeMismatch(
            self.name().to_owned(),
            ColumnType::Numerical,
        ));
    }

    fn get_values_as_str(&self) -> Result<Vec<String>, DatasetError> {
        Ok(self.data.clone())
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

    fn column_type(&self) -> ColumnType {
        ColumnType::Categorical
    }
}

#[derive(Default)]
pub struct DataFrame {
    pub columns: Vec<Box<dyn ColumnOps>>,
}

#[derive(Debug, Clone)]
pub struct ColumnGroupNumericItem {
    pub name: String,
    pub value: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ColumnGroupCategoricalItem {
    pub name: String,
    pub value: Vec<String>,
}

impl DataFrame {
    pub fn new() -> DataFrame {
        DataFrame::default()
    }

    pub fn add_binary_column(&mut self, name: &str, data: Vec<bool>) {
        let mut entries = Vec::new();
        for (index, value) in data.iter().enumerate() {
            entries.push(BinaryEntry {
                index,
                value: *value,
            });
        }
        let column = BinaryColumn {
            name: name.to_string(),
            data: entries,
        };
        self.columns.push(Box::new(column));
    }

    pub fn add_numerical_binary_column(
        &mut self,
        name: &str,
        data: Vec<u8>,
    ) -> Result<(), DatasetError> {
        let mut entries = Vec::new();
        for (index, value) in data.iter().enumerate() {
            let b = match *value {
                0 => false,
                1 => true,
                _ => {
                    return Err(DatasetError::ValueTypeMismatch(
                        value.to_string(),
                        name.to_owned(),
                        ColumnType::Binary,
                    ));
                }
            };

            entries.push(BinaryEntry { index, value: b });
        }
        let column = BinaryColumn {
            name: name.to_string(),
            data: entries,
        };
        self.columns.push(Box::new(column));

        Ok(())
    }

    pub fn add_numerical_column(&mut self, name: &str, data: Vec<f64>) {
        let mut entries = Vec::new();
        for (index, value) in data.iter().enumerate() {
            entries.push(NumericalEntry {
                index,
                value: *value,
            });
        }
        let column = NumericalColumn {
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

    pub fn add_value_to_column_str(
        &mut self,
        column_name: &str,
        value: &str,
    ) -> Result<(), DatasetError> {
        let column = self
            .columns
            .iter_mut()
            .find(|column| column.name() == column_name)
            .ok_or(DatasetError::ColumnNotFound(column_name.to_owned()))?;

        match column.column_type() {
            ColumnType::Binary => {
                if let Ok(value) = value.parse::<bool>() {
                    column.add_entry(&value);
                }
            }
            ColumnType::Numerical => {
                if let Ok(value) = value.parse::<f64>() {
                    column.add_entry(&value);
                }
            }
            ColumnType::Categorical => {
                column.add_entry(&value.to_owned());
            }
        };

        Ok(())
    }

    pub fn add_value_to_column(&mut self, column_index: usize, value: &dyn Any) {
        let column = &mut self.columns[column_index];

        match column.column_type() {
            ColumnType::Binary => {
                if let Some(value) = value.downcast_ref::<bool>() {
                    column.add_entry(value);
                }
            }
            ColumnType::Numerical => {
                if let Some(value) = value.downcast_ref::<f64>() {
                    column.add_entry(value);
                }
            }
            ColumnType::Categorical => {
                if let Some(value) = value.downcast_ref::<String>() {
                    column.add_entry(value);
                }
            }
        }
    }

    pub fn grand_descriptives(
        &self,
        observation_columns: &[&str],
    ) -> Result<(f64, usize), DatasetError> {
        let mut means: Vec<f64> = Vec::new();

        let mut grand_n = 0;
        for column in self.columns.iter() {
            if observation_columns.contains(&column.name()) {
                // log column type
                if column.column_type() == ColumnType::Numerical {
                    means.push(column.mean());
                    grand_n += column.n();
                } else {
                    return Err(DatasetError::ColumnTypeMismatch(
                        column.name().to_string(),
                        ColumnType::Numerical,
                    ));
                }
            }
        }

        let grand_mean = means.iter().sum::<f64>() / means.len() as f64;
        Ok((grand_mean, grand_n))
    }

    pub fn get_column(&self, column_name: &str) -> Result<&Box<dyn ColumnOps>, DatasetError> {
        let c = self.columns.iter().find(|&x| x.name() == column_name);
        match c {
            Some(column) => Ok(column),
            None => Err(DatasetError::ColumnNotFound(column_name.to_owned())),
        }
    }

    pub fn group_numeric_columns(
        &self,
        column_names: &[&str],
    ) -> Result<Vec<ColumnGroupNumericItem>, DatasetError> {
        let mut groups: Vec<ColumnGroupNumericItem> = Vec::new();
        for column in self.columns.iter() {
            if column_names.contains(&column.name()) {
                if column.column_type() == ColumnType::Numerical {
                    let values: Vec<f64> = column.get_values_as_f64()?;
                    let group = ColumnGroupNumericItem {
                        name: column.name().to_string(),
                        value: values,
                    };
                    groups.push(group);
                } else {
                    return Err(DatasetError::ColumnTypeMismatch(
                        column.name().to_string(),
                        ColumnType::Numerical,
                    ));
                }
            }
        }

        Ok(groups)
    }

    pub fn group_categorical_columns(
        &self,
        column_names: &[&str],
    ) -> Result<Vec<ColumnGroupCategoricalItem>, DatasetError> {
        let mut groups: Vec<ColumnGroupCategoricalItem> = Vec::new();
        for column in self.columns.iter() {
            if column_names.contains(&column.name()) {
                if column.column_type() == ColumnType::Categorical {
                    let values: Vec<String> = column
                        .get_values()
                        .iter()
                        .map(|x| x.downcast_ref::<String>().unwrap().to_string())
                        .collect();
                    let group = ColumnGroupCategoricalItem {
                        name: column.name().to_string(),
                        value: values,
                    };
                    groups.push(group);
                } else {
                    return Err(DatasetError::ColumnTypeMismatch(
                        column.name().to_string(),
                        ColumnType::Categorical,
                    ));
                }
            }
        }

        Ok(groups)
    }

    pub fn join_numeric_columns(&self, column_names: &[&str]) -> Result<Vec<f64>, DatasetError> {
        let mut values: Vec<f64> = Vec::new();
        for column in self.columns.iter() {
            if column_names.contains(&column.name()) {
                if column.column_type() == ColumnType::Numerical {
                    values.extend(column.get_values_as_f64()?);
                } else {
                    return Err(DatasetError::ColumnTypeMismatch(
                        column.name().to_string(),
                        ColumnType::Numerical,
                    ));
                }
            }
        }

        Ok(values)
    }

    pub fn cat_iv_levels(
        &self,
        iv_column_names: &[&str],
        dv_column_name: &str,
    ) -> Result<HashMap<String, Vec<f64>>, DatasetError> {
        let dv_column: &Box<dyn ColumnOps> =
            match self.columns.iter().find(|&x| x.name() == dv_column_name) {
                Some(column) => column,
                None => {
                    return Err(DatasetError::ColumnNotFound(dv_column_name.to_owned()));
                }
            };

        if dv_column.column_type() != ColumnType::Numerical {
            return Err(DatasetError::ColumnTypeMismatch(
                dv_column_name.to_owned(),
                ColumnType::Numerical,
            ));
        }

        let mut iv_levels: HashMap<String, Vec<f64>> = HashMap::new();
        for column in self.columns.iter() {
            let iv_name = column.name();
            // Check if the column is an independent variable
            if iv_column_names.contains(&iv_name) {
                // Check if the column is categorical (conditions e.g.)
                if column.column_type() == ColumnType::Categorical {
                    // Loop over the values
                    let c_values: Vec<String> = column
                        .get_values()
                        .iter()
                        .filter_map(|x| x.downcast_ref::<String>().cloned())
                        .collect();

                    for (i, ob) in c_values.iter().enumerate() {
                        let iv_name = format!("{}_{}", iv_name, ob);

                        iv_levels.entry(iv_name.clone()).or_default();

                        // Collect dependent variable for this group
                        if let Ok(value) = dv_column.get_value(i) {
                            if let Some(f_value) = value.downcast_ref::<f64>() {
                                iv_levels.get_mut(&iv_name).unwrap().push(*f_value);
                            }
                        }
                    }
                } else {
                    return Err(DatasetError::ColumnTypeMismatch(
                        iv_name.to_owned(),
                        ColumnType::Categorical,
                    ));
                }
            }
        }

        Ok(iv_levels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_binary_column() {
        let mut df = DataFrame::new();

        df.add_binary_column("present", vec![true, false, false, true]);
        assert_eq!(df.columns[0].name(), "present");

        assert!(df
            .add_numerical_binary_column("was_late", vec![1, 0, 0, 1])
            .is_ok());
        assert_eq!(df.columns[1].name(), "was_late");

        assert_eq!(df.columns.len(), 2);
    }

    #[test]
    fn test_add_numerical_column() {
        let mut df = DataFrame::new();
        df.add_numerical_column("salaries", vec![3500.0, 4600.0, 4900.0]);
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
    fn test_freq() {
        let mut df = DataFrame::new();
        df.add_binary_column("ages", vec![true, true, true, false, false]);
        let freq = df.columns[0].freq(&true);
        assert_eq!(freq, 3);
    }

    #[test]
    fn test_mean() {
        let mut df = DataFrame::new();
        df.add_numerical_column("salaries", vec![3500.0, 4600.0, 4900.0]);
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
        df.add_binary_column("is_senior", vec![false, true, false, false]);
        df.add_numerical_column("salaries", vec![3500.0, 4600.0, 4900.0, 4900.0]);
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
        let is_senior_column = &df.columns[0];
        assert_eq!(is_senior_column.freq(&true), 1);
        assert_eq!(is_senior_column.freq(&false), 3);

        // Test frequency for numerical column
        let salary_column = &df.columns[1];
        assert_eq!(salary_column.freq(&3500.0), 1);
        assert_eq!(salary_column.freq(&4600.0), 1);
        assert_eq!(salary_column.freq(&4900.0), 2);
        assert_eq!(salary_column.freq(&6000.0), 0); // Non-existent value

        // Test frequency for categorical column
        let department_column = &df.columns[2];
        assert_eq!(department_column.freq(&"HR".to_string()), 2);
        assert_eq!(department_column.freq(&"IT".to_string()), 2);
        assert_eq!(department_column.freq(&"Marketing".to_string()), 1);
        assert_eq!(department_column.freq(&"Finance".to_string()), 0); // Non-existent value
    }
}
