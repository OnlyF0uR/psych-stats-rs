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
                // check if we can parse to float64
                if field.parse::<f64>().is_ok() && field.contains(".") {
                    // add decimal column

                    df.add_decimal_column(&headers[index], vec![field.parse::<f64>().unwrap()]);
                } else if field.parse::<i64>().is_ok() {
                    // add integer column
                    df.add_integer_column(&headers[index], vec![field.parse::<i64>().unwrap()]);
                } else {
                    // add categorical column
                    df.add_categorical_column(&headers[index], vec![field.to_string()]);
                }
            }
            is_first = false;
        } else {
            // IDEA: We could check if we have inferred the right type, given headers[index] and the value
            // we encountered in the record. If not, we could update the type in the header_map.

            for (index, field) in record.iter().enumerate() {
                // println!("Field: {}", field.parse::<i64>().unwrap());

                let decimal_value = field.parse::<f64>();

                if decimal_value.is_ok() && field.contains(".") {
                    let decimal_value = decimal_value.unwrap();
                    // add decimal column
                    df.add_value_to_column_str(&headers[index], &decimal_value);
                } else if let Ok(integer_value) = field.parse::<i64>() {
                    // add integer column
                    df.add_value_to_column_str(&headers[index], &integer_value);
                } else {
                    // add categorical column
                    df.add_value_to_column_str(&headers[index], &field.to_string());
                }
            }
        }
    }
    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_csv() {
        let df = import_csv("samples/data1.csv").unwrap();
        assert_eq!(df.columns.len(), 3);

        // lets assert frequency to test
        assert_eq!(df.columns[0].name(), "Column_A");
        assert_eq!(df.columns[1].name(), "Column_B");
        assert_eq!(df.columns[2].name(), "Column_C");

        // assert the first value of every column
        let item = df.columns[0].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<i64>().unwrap(), &1);

        let item = df.columns[1].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<f64>().unwrap(), &6.5);

        let item = df.columns[2].get_value(0).unwrap();
        assert_eq!(item.downcast_ref::<String>().unwrap(), &"a".to_string());
    }
}
