import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def main():
    # Load environment variables
    load_dotenv()
    db_url = os.getenv("API_URL")
    url = urlparse(db_url)

    # Connect to PostgreSQL
    try:
        connection = psycopg2.connect(
            dbname=url.path[1:],  # Remove leading '/'
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        print("Connected to the PostgreSQL database successfully.")

        # Fetch all table names
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        tables = cursor.fetchall()

        # Create SQLite connection using SQLAlchemy
        sqlite_engine = create_engine("sqlite:///local_database.db", echo=False)
        print("SQLite database created.")

        # Load each table and save it to SQLite
        for (table_name,) in tables:
            print(f"Loading table {table_name}...")
            df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
            df.to_sql(table_name, sqlite_engine, index=False, if_exists="replace")
            print(f"Table {table_name} saved to SQLite.")

        cursor.close()
        connection.close()
        print("All tables have been successfully exported to SQLite.")

    except Exception as error:
        print("Error:", error)

if __name__ == "__main__":
    main()
