use std::{error::Error, fmt};

#[derive(Debug)]
pub enum DatasetError {
    EmptyValueError(usize, usize),
    ColumnTypeMismatch(String, String),
    ColumnNotFound(String),
}

// Implement Display for custom error formatting
impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DatasetError::EmptyValueError(ref index, ref length) => {
                write!(f, "Index out of range: {}/{}", index, length)
            }
            DatasetError::ColumnTypeMismatch(ref column_name, ref expected_type) => {
                write!(f, "Column {} is not of {} type", column_name, expected_type)
            }
            DatasetError::ColumnNotFound(ref column_name) => {
                write!(f, "Column {} not found", column_name)
            }
        }
    }
}

// Implement the Error trait for custom error handling
impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            DatasetError::EmptyValueError(_, _) => None,
            DatasetError::ColumnTypeMismatch(_, _) => None,
            DatasetError::ColumnNotFound(_) => None,
        }
    }
}
