import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_rainfall_data(file_path):
    """
    Preprocess the rainfall data from the CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the rainfall_yearly.csv file
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned and preprocessed dataframe
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Melt the dataframe to convert months from columns to rows
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df_melted = pd.melt(
        df,
        id_vars=['subdivision', 'year'],
        value_vars=months,
        var_name='month',
        value_name='rainfall'
    )
    
    # Create a month number column for easier sorting
    month_to_num = {month: i+1 for i, month in enumerate(months)}
    df_melted['month_num'] = df_melted['month'].map(month_to_num)
    
    # Sort by subdivision, year, and month
    df_melted = df_melted.sort_values(['subdivision', 'year', 'month_num'])
    
    # Add a date column (first day of each month) for time series analysis
    df_melted['date'] = pd.to_datetime(df_melted['year'].astype(str) + '-' + 
                                        df_melted['month_num'].astype(str) + '-01')
    
    # Check for missing values
    missing_values = df_melted['rainfall'].isna().sum()
    print(f"Number of missing rainfall values: {missing_values}")
    
    # Replace any missing values with the mean for that month and subdivision
    if missing_values > 0:
        df_melted['rainfall'] = df_melted.groupby(['subdivision', 'month'])['rainfall'].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Add season column
    season_map = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
        12: 'Winter'
    }
    df_melted['season'] = df_melted['month_num'].map(season_map)
    
    # Create seasonal aggregates
    seasonal_df = df_melted.groupby(['subdivision', 'year', 'season'])['rainfall'].sum().reset_index()
    
    # Calculate year-on-year changes
    df_melted['prev_year_rainfall'] = df_melted.groupby(['subdivision', 'month'])['rainfall'].shift(1)
    df_melted['yoy_change'] = df_melted['rainfall'] - df_melted['prev_year_rainfall']
    df_melted['yoy_change_pct'] = (df_melted['yoy_change'] / df_melted['prev_year_rainfall']) * 100
    
    return {
        'monthly': df_melted,
        'seasonal': seasonal_df,
        'original': df
    }

def analyze_rainfall_data(data_dict):
    """
    Perform basic analysis on the preprocessed rainfall data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing preprocessed dataframes
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    monthly_df = data_dict['monthly']
    seasonal_df = data_dict['seasonal']
    
    # Calculate basic statistics for monthly rainfall
    monthly_stats = monthly_df.groupby('month')['rainfall'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
    
    # Calculate seasonal statistics
    seasonal_stats = seasonal_df.groupby('season')['rainfall'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
    
    # Calculate yearly statistics
    yearly_stats = monthly_df.groupby(['subdivision', 'year'])['rainfall'].sum().reset_index()
    yearly_stats = yearly_stats.groupby('year')['rainfall'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
    
    return {
        'monthly_stats': monthly_stats,
        'seasonal_stats': seasonal_stats,
        'yearly_stats': yearly_stats
    }

def visualize_rainfall_data(data_dict):
    """
    Create visualizations for the rainfall data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing preprocessed dataframes
    """
    monthly_df = data_dict['monthly']
    
    # Set the style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Monthly rainfall distribution (box plot)
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='month', y='rainfall', data=monthly_df)
    plt.title('Monthly Rainfall Distribution (1901-1908)')
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.savefig('monthly_rainfall_distribution.png')
    plt.close()
    
    # Time series of total yearly rainfall
    yearly_total = monthly_df.groupby(['year'])['rainfall'].sum().reset_index()
    plt.figure(figsize=(14, 7))
    plt.plot(yearly_total['year'], yearly_total['rainfall'], marker='o')
    plt.title('Total Yearly Rainfall (1901-1908)')
    plt.xlabel('Year')
    plt.ylabel('Total Rainfall (mm)')
    plt.grid(True)
    plt.savefig('yearly_rainfall_trend.png')
    plt.close()
    
    # Heatmap of monthly rainfall
    pivot_df = monthly_df.pivot_table(index='year', columns='month', values='rainfall')
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap='Blues', annot=True, fmt='.1f')
    plt.title('Monthly Rainfall Heatmap (1901-1908)')
    plt.savefig('monthly_rainfall_heatmap.png')
    plt.close()

if __name__ == "__main__":
    # Path to the CSV file
    file_path = "../Datasets/Rainfall/rainfall_yearly.csv"
    
    # Preprocess the data
    data_dict = preprocess_rainfall_data(file_path)
    
    # Analyze the data
    analysis_results = analyze_rainfall_data(data_dict)
    
    # Visualize the data
    visualize_rainfall_data(data_dict)
    
    # Display some basic statistics
    print("\nMonthly Rainfall Statistics:")
    print(analysis_results['monthly_stats'])
    
    print("\nSeasonal Rainfall Statistics:")
    print(analysis_results['seasonal_stats'])
    
    print("\nYearly Rainfall Statistics:")
    print(analysis_results['yearly_stats'])
    
    # Save the processed data
    data_dict['monthly'].to_csv('processed_monthly_rainfall.csv', index=False)
    data_dict['seasonal'].to_csv('processed_seasonal_rainfall.csv', index=False)
    
    print("\nPreprocessing completed. Output files saved.")