use std::{error::Error, fmt};

use super::dataframe::ColumnType;

#[derive(Debug)]
pub enum DatasetError {
    EmptyValue(usize, usize),
    ColumnTypeMismatch(String, ColumnType),
    ColumnNotFound(String),
    ValueTypeMismatch(String, String, ColumnType),
    InvalidData(String),
}

// Implement Display for custom error formatting
impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DatasetError::EmptyValue(ref index, ref length) => {
                write!(f, "Index out of range: {}/{}", index, length)
            }
            DatasetError::ColumnTypeMismatch(ref column_name, ref expected_type) => {
                write!(
                    f,
                    "Column {} is not of {} type",
                    column_name,
                    expected_type.as_str()
                )
            }
            DatasetError::ColumnNotFound(ref column_name) => {
                write!(f, "Column {} not found", column_name)
            }
            DatasetError::ValueTypeMismatch(ref value, ref column_name, ref expected_type) => {
                write!(
                    f,
                    "Value {} in column {} is not of {} type",
                    value,
                    column_name,
                    expected_type.as_str()
                )
            }
            DatasetError::InvalidData(ref message) => write!(f, "{}", message),
        }
    }
}

// Implement the Error trait for custom error handling
impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            DatasetError::EmptyValue(_, _) => None,
            DatasetError::ColumnTypeMismatch(_, _) => None,
            DatasetError::ColumnNotFound(_) => None,
            DatasetError::ValueTypeMismatch(_, _, _) => None,
            DatasetError::InvalidData(_) => None,
        }
    }
}
