import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def report_missing(df):
    """
    Calculate the number of null and blank value.
    
    Parameter:
    df (DataFrame)
        The dataframe used to check for missing values
    
    Return:
    completed_report (DataFrame)
        The report table including the number of null, blank values and their percentage in total
    """
    # Total observation count
    total_obs = df.shape[0]
    # Create a dataframe
    missing = pd.DataFrame()
    # Total nulls
    missing['null_count'] = df.isnull().sum()
    # Total blank value
    missing['blank_count'] = [df[df[c].astype(str) == ""][c].count() for c in df.columns]
    # Total missing value
    missing['total_missing'] = missing.sum(axis = 1)
    # Report missing percentage
    missing['null_percent'] = round(100* (missing['null_count']/ total_obs), 2)
    missing['blank_percent'] = round(100* (missing['blank_count']/ total_obs), 2)
    missing['total_missing_percent'] = round(100* (missing['total_missing']/ total_obs), 2)
    
    completed_report = missing.sort_values(
        by = 'total_missing_percent',
        ascending = False
    )
    return completed_report

def report_duplicates(df):
    """
    Prints duplicate rows in the DataFrame based on all columns.
    """
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        print(duplicates)
    else:
        print("No duplicate rows found.")

def drop_duplicate(df, idx='all'):
    """
    Drops rows with the given indexes, or drops all duplicate rows if idx='all'.

    Parameters:
    idx (list, 'all'): List of indexes to drop, or 'all' (default) to drop all duplicates.

    Returns:
    pd.DataFrame: DataFrame after dropping.
    """
    if idx == 'all':
        # Drop all duplicate rows (keep first occurrence)
        df = df.drop_duplicates(keep='first')
        print("All duplicate rows removed.")
    elif isinstance(idx, (list, pd.Index)):
        # Drop specified indexes
        df = df.drop(idx)
        print(f"Rows at indexes {idx} removed.")
    else:
        print("Invalid input for idx. Must be a list of indexes or 'all'.")
    return df

def train_test_spliter(df, target_var):
    # Split the data into features (X) and target (y), and then into training and testing sets
    X = df.drop(columns=[target_var])
    y = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
    
    # Scale the training and test data using standardscaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test