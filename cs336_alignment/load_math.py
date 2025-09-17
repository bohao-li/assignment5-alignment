import pandas as pd

parquet_file_path = 'data/MATH/math.parquet'

try:
    # Use the read_parquet() function to load the file into a DataFrame
    df = pd.read_parquet(parquet_file_path)
    
    # --- Basic Data Exploration ---
    
    # Get the number of rows and columns
    num_rows, num_cols = df.shape
    print("Successfully loaded the Parquet file.")
    print(f"\nDataFrame shape: {num_rows} rows, {num_cols} columns")
    
    # Print the column names and their data types
    print("\nData types and non-null values:")
    df.info()
    
    # Print basic descriptive statistics for numerical columns
    print("\nDescriptive statistics:")
    print(df.describe())
    
    # Print the first 5 rows to verify the data was loaded correctly
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    
except FileNotFoundError:
    print(f"Error: The file '{parquet_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")