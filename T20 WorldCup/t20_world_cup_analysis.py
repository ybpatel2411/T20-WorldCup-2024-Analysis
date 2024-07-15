import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
matches_df = pd.read_csv('c:/Users/Yash/Desktop/T20 WorldCup/matches.csv')
deliveries_df = pd.read_csv('c:/Users/Yash/Desktop/T20 WorldCup/deliveries.csv')

# Display the first few rows of both datasets
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

print("Matches Data:")
print(matches_df)

print("\nDeliveries Data:")
print(deliveries_df)

# Data Cleaning
matches_df['date'] = pd.to_datetime(matches_df['date'], format='%Y/%m/%d')  # Corrected format here
matches_df.dropna(inplace=True)
deliveries_df.dropna(inplace=True)

# Ensure the necessary columns are present in matches_df
required_columns_matches = ['winner', 'win_by_runs', 'win_by_wickets']
missing_columns = [column for column in required_columns_matches if column not in matches_df.columns]

if missing_columns:
    print(f"Error: Missing required columns in matches dataset. Missing columns: {missing_columns}")
else:
    # Descriptive Statistics for matches_df
    print("\nDescriptive Statistics for Matches:")
    print(matches_df.describe())

    # Descriptive Statistics for deliveries_df
    print("\nDescriptive Statistics for Deliveries:")
    print(deliveries_df.describe())

    # Exploratory Data Analysis with seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=matches_df[['win_by_runs', 'win_by_wickets']])
    plt.title('Distribution of Win by Runs and Wickets')
    plt.ylabel('Count')
    plt.xlabel('Metrics')
    plt.show()

    # Interactive Visualization with Plotly
    fig = px.scatter(matches_df, x='winner', y='win_by_runs', color='venue', 
                     title='Winning Runs by Venue',
                     labels={'winner': 'Winning Team', 'win_by_runs': 'Winning Runs'})
    fig.show()

    # Prepare data for predictive model
    matches_df.dropna(subset=['win_by_runs', 'win_by_wickets'], inplace=True)
    X = matches_df[['win_by_runs']]
    y = matches_df['win_by_wickets']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse:.2f}')

    # Additional Visualizations
    # Example: Average winning runs per team
    avg_winning_runs = matches_df.groupby('winner')['win_by_runs'].mean().reset_index()
    fig = px.bar(avg_winning_runs, x='winner', y='win_by_runs', title='Average Winning Runs per Team')
    fig.show()

    # Example: Total runs per batsman
    total_runs_batsman = deliveries_df.groupby('batsman')['batsman_runs'].sum().reset_index()
    fig = px.bar(total_runs_batsman, x='batsman', y='batsman_runs', title='Total Runs per Batsman')
    fig.show()
