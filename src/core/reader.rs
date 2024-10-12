use std::{error::Error, fs::File, io::BufReader};

use csv::ReaderBuilder;

use super::dataframe::DataFrame;

pub fn import_csv(path: &str) -> Result<DataFrame, Box<dyn Error>> {
    let file = File::open(path)?;
    let buffered = BufReader::new(file);

    let mut rdr = ReaderBuilder::new()
        .has_headers(true) // Adjust depending on the file
        .from_reader(buffered);

    let mut df = DataFrame::new();

    // vec for headers until we have identified the type of the column
    let headers = rdr
        .headers()?
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();

    // loop over the records
    let mut is_first = true;
    for result in rdr.records() {
        let record = result?;
        if is_first {
            for (index, field) in record.iter().enumerate() {
                if field.contains(".") {
                    let parsed = field.parse::<f64>()?;
                    df.add_numerical_column(&headers[index], vec![parsed]);
                } else if is_binary(&headers[index], field) {
                    if field == "0" || field == "false" {
                        df.add_binary_column(&headers[index], vec![false]);
                    } else {
                        df.add_binary_column(&headers[index], vec![true]);
                    }
                } else if field.parse::<i64>().is_ok() {
                    let parsed = field.parse::<f64>()?;
                    df.add_numerical_column(&headers[index], vec![parsed]);
                } else {
                    df.add_categorical_column(&headers[index], vec![field.to_owned()]);
                }
            }
            is_first = false;
        } else {
            // IDEA: We could check if we have inferred the right type, given headers[index] and the value
            // we encountered in the record. If not, we could update the type in the header_map.

            for (index, field) in record.iter().enumerate() {
                df.add_value_to_column_str(&headers[index], field)?;
            }
        }
    }

    Ok(df)
}

fn is_binary(header: &str, value: &str) -> bool {
    if header.to_lowercase().contains("_id") {
        return false;
    }
    value == "0" || value == "1" || value == "true" || value == "false"
}

#[cfg(test)]
mod tests {
    use crate::core::dataframe::ColumnType;

    use super::*;

    #[test]
    fn test_import_csv() {
        let df = import_csv("samples/data1.csv").unwrap();
        assert_eq!(df.columns.len(), 3);

        // lets assert frequency to test
        assert_eq!(df.columns[0].name(), "Column_A");
        assert_eq!(df.columns[1].name(), "Column_B");
        assert_eq!(df.columns[2].name(), "Column_C");

        let dtype = df.columns[0].column_type();
        assert_eq!(dtype, ColumnType::Numerical);

        // assert the first value of every column
        let item = df.columns[0].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<f64>().unwrap(), &3.0);

        let item = df.columns[1].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<f64>().unwrap(), &6.5);

        let item = df.columns[2].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<String>().unwrap(), &"a".to_string());
    }
}
