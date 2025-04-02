import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
def load_data(file_path):
    try:
        # Assuming the dataset is a CSV file downloaded from Kaggle
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print("First few rows of the dataset:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# 2. Clean the dataset
def clean_data(df):
    if df is None:
        return None

    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Handle missing values
    # For numerical columns (e.g., rainfall), fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # For categorical columns (e.g., SUBDIVISION), fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df

# 3. Convert data types and create features
def process_data(df):
    if df is None:
        return None

    # Convert YEAR and MONTH to appropriate types if needed
    if 'YEAR' in df.columns and df['YEAR'].dtype == 'object':
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')

    if 'MONTH' in df.columns and df['MONTH'].dtype == 'object':
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')

    # Create a date column for time series analysis (if YEAR and MONTH exist)
    if 'YEAR' in df.columns and 'MONTH' in df.columns:
        df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str), format='%Y-%m', errors='coerce')

    # Sort by date if DATE column exists
    if 'DATE' in df.columns:
        df = df.sort_values(by='DATE')

    # Ensure rainfall data is numeric
    rainfall_cols = [col for col in df.columns if 'RAIN' in col.upper() or 'PRECIP' in col.upper()]
    for col in rainfall_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# 4. Normalize/Scale numerical data
def scale_data(df):
    if df is None:
        return None

    # Identify numerical columns to scale (exclude YEAR, MONTH, etc.)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['YEAR', 'MONTH']  # Add other columns to exclude if needed
    scale_cols = [col for col in numerical_cols if col not in exclude_cols]

    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        print("\nData scaled using StandardScaler.")
        return df, scaler
    else:
        print("\nNo numerical columns to scale.")
        return df, None

# 5. Save preprocessed data
def save_preprocessed_data(df, file_path='preprocessed_rainfall.csv'):
    if df is not None:
        df.to_csv(file_path, index=False)
        print(f"\nPreprocessed data saved to {file_path}")

def main():
    # Specify the path to your downloaded CSV file from Kaggle
    file_path = 'Datasets/rainfall_in_india.csv'  # Update this path

    # Load data
    df = load_data(file_path)

    # Clean data
    df = clean_data(df)

    if df is not None:
        # Process data (convert types, create features)
        df = process_data(df)

        # Scale numerical data
        df, scaler = scale_data(df)

        # Save preprocessed data
        save_preprocessed_data(df)

if __name__ == "__main__":
    main()