import sqlite3 as sq3
import pandas as pd

def fetch_data_from_db(query: str) -> pd.DataFrame:
    """
    Connects to a SQLite database, executes the given query, and returns the result as a Pandas DataFrame.

    Parameters:
        query (str): The SQL query to execute.
        db_path (str): Path to the SQLite database file. Default is "data/product_database.db".

    Returns:
        pd.DataFrame: The result of the query as a Pandas DataFrame.
    """
    db_path = "data/local_database.db"
    try:
        # Connect to the database
        with sq3.connect(db_path) as con:
            # Execute the query and fetch data into a DataFrame
            df = pd.read_sql_query(query, con)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
