import pandas as pd


# Goal of this file is to encode dataset which i used to train and evaluate unprotected neural network.
# It takes original dataset for evaluation and encodes by the same encode table as the internal network configuration
# values
def transform_dataset(input_filename, output_filename):
    # Mapping based on my c code table
    mapping = {
        -7: "0b110100010000001",
        -6: "0b010100010000010",
        -5: "0b100000000000011",
        -4: "0b100100010000100",
        -3: "0b010000000000101",
        -2: "0b110000000000110",
        -1: "0b000100010000111",
         1: "0b000100010001000",
         2: "0b110000000001001",
         3: "0b010000000001010",
         4: "0b100100010001011",
         5: "0b100000000001100",
         6: "0b010100010001101",
         7: "0b110100010001110"
    }

    try:
        df = pd.read_csv(input_filename)
        
        input_cols = [col for col in df.columns if col != 'label']

        for col in input_cols:
            df[col] = df[col].map(mapping)

        df.to_csv(output_filename, index=False)
        print(f"Success! Transformed data saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    transform_dataset('test.csv', 'transformed_dataset.csv')