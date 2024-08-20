import os
import time
from typing import Dict, Optional, List, Any
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


class GhvSnowflake:
    def __init__(self):
        self.dict_db_env = None
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        self.get_db_env()
        self.backoff_delays = [1, 2, 4, 8, 16]  # Delays in seconds

    def __del__(self):
        """Destructor to ensure the Snowflake connection is closed."""
        try:
            self.close_connection()
        except Exception as e:
            print(f"Error closing Snowflake connection: {e}")

    def close_connection(self):
        """Closes the Snowflake connection if it's open."""
        if self.conn is not None and not self.conn.is_closed():
            self.conn.close()
            self.conn = None

    def get_db_env(self) -> Dict[str, str]:
        """Fetches database environment variables."""
        if self.dict_db_env is None:
            self.dict_db_env = {
                "snowflake_user": os.getenv("SNOWFLAKE_USER"),
                "snowflake_password": os.getenv("SNOWFLAKE_PASSWORD"),
                "snowflake_account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "snowflake_warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
                "snowflake_db": os.getenv("SNOWFLAKE_DB"),
                "snowflake_schema": os.getenv("SNOWFLAKE_SCHEMA"),
            }
        return self.dict_db_env

    def get_snowflake_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Establishes a connection to Snowflake with hardcoded backoff delay."""
        if self.conn is None or self.conn.is_closed():
            dict_db_env = self.get_db_env()
            for attempt, delay in enumerate(self.backoff_delays, 1):
                try:
                    self.conn = snowflake.connector.connect(
                        user=dict_db_env["snowflake_user"],
                        password=dict_db_env["snowflake_password"],
                        account=dict_db_env["snowflake_account"],
                        warehouse=dict_db_env["snowflake_warehouse"],
                        database=dict_db_env["snowflake_db"],
                        schema=dict_db_env["snowflake_schema"],
                        timeout=30  # Set a timeout for connection
                    )
                    break
                except snowflake.connector.errors.OperationalError as e:
                    print(f"Connection attempt {attempt} failed: {
                          e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            if self.conn is None or self.conn.is_closed():
                raise Exception(
                    "Could not connect to Snowflake after multiple attempts.")
        return self.conn

    def store_embedding_data(self, data: List[Dict[str, Any]]):
        """
        Stores the embedding data in the Snowflake table 'github_to_vector_code_chunks'.

        Args:
            data (List[Dict[str, Any]]): A list of dictionaries containing the embedding data.
        """
        conn = self.get_snowflake_connection()
        df = pd.DataFrame(data)

        try:
            success, nchunks, nrows, _ = write_pandas(
                conn, df, '"github_to_vector_code_chunks"')
            if success:
                print(f"Successfully inserted {nrows} rows into Snowflake.")
            else:
                print("Failed to insert rows into Snowflake.")
        except Exception as e:
            print(f"Error inserting data into Snowflake: {e}")

    def read_embedding_data(self, repo_name: str, file_name: str) -> pd.DataFrame:
        """
        Reads the embedding data from the Snowflake table 'github_to_vector_code_chunks' for a specific file.

        Args:
            repo_name (str): The name of the repository.
            file_name (str): The name of the file.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved embedding data.
        """
        conn = self.get_snowflake_connection()
        query = f"""
        SELECT *
        FROM "github_to_vector_code_chunks"
        WHERE "repo_name" = '{repo_name}' AND "file_name" = '{file_name}'
        """

        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error reading data from Snowflake: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def test_connection(self) -> bool:
        """Tests opening a connection to Snowflake and performing a basic SELECT query."""
        try:
            conn = self.get_snowflake_connection()
            query = 'SELECT * FROM "github_to_vector_code_chunks" LIMIT 10'
            df = pd.read_sql(query, conn)

            print("Snowflake connection test successful. Retrieved rows:")
            print(df)
            return True
        except Exception as e:
            print(f"Snowflake connection test failed with error: {e}")
            return False


if __name__ == "__main__":
    # Initialize the Snowflake client
    snowflake_client = GhvSnowflake()

    # Test the connection to Snowflake and perform a basic SELECT query
    connection_test_result = snowflake_client.test_connection()

    # Output the result of the connection test
    print(f"Connection Test Passed: {connection_test_result}")
