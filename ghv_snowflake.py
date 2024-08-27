import os
import time
from typing import Dict, Optional, List, Any
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from datetime import datetime, timezone


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
            try:
                self.conn.close()
                print("Snowflake connection closed.")
            except snowflake.connector.Error as e:
                print(f"Error closing Snowflake connection: {e}")
                raise
            finally:
                self.conn = None  # Reset the cached connection to None

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

    def read_embedding_data(self, repo_name: str, file_name: str) -> pd.DataFrame:
        """
        Reads the embedding data from the Snowflake table 'github_to_vector_text' for a specific file.

        Args:
            repo_name (str): The name of the repository.
            file_name (str): The name of the file.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved embedding data.
        """
        conn = self.get_snowflake_connection()
        query = f"""
        SELECT *
        FROM "github_to_vector_text"
        WHERE "repo_name" = '{repo_name}' AND "file_name" = '{file_name}'
        """

        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error reading data from Snowflake: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def store_single_embedding(self, data: Dict[str, Any]):
        """
        Stores a single record in the Snowflake table 'github_to_vector_text'.

        Args:
            data (Dict[str, Any]): A dictionary containing the data for a single embedding.
            commit (bool): Whether to commit the transaction after the insertion.
        """
        required_columns: List[str] = [
            "org_name", "repo_name", "file_folder", "file_name",
            "text", "index_name", "embedding_id"
        ]

        # Ensure all required fields are present and not None
        for column in required_columns:
            if column not in data or data[column] is None:
                raise ValueError(
                    f"Missing or null value for required column: {column}")

        conn = self.get_snowflake_connection()
        cursor = conn.cursor()

        # Override storage_datetime with current timestamp
        data["storage_datetime"] = datetime.now(
            timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        try:
            # Construct the SQL INSERT statement
            insert_sql = """
            INSERT INTO "github_to_vector_text" 
            ("org_name", "repo_name", "file_folder", "file_name", "text", "index_name", "embedding_id", "storage_datetime")
            VALUES (%(org_name)s, %(repo_name)s, %(file_folder)s, %(file_name)s, %(text)s, %(index_name)s, %(embedding_id)s, %(storage_datetime)s)
            """
            # Execute the SQL command with the provided data
            cursor.execute(insert_sql, data)
            conn.commit()

            # print(f"Successfully inserted the record into Snowflake.")

        except snowflake.connector.Error as e:
            print(f"Error during the insertion to Snowflake: {e}")
            raise

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

        finally:
            cursor.close()

    def read_embedding_by_id(self, embedding_id: str) -> pd.DataFrame:
        """
        Reads a specific record from the Snowflake table 'github_to_vector_text' using the embedding_id.

        Args:
            embedding_id (str): The ID of the embedding to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved record.
        """
        conn = self.get_snowflake_connection()
        query = f"""
        SELECT *
        FROM "github_to_vector_text"
        WHERE "embedding_id" = '{embedding_id}'
        """

        try:
            df = pd.read_sql(query, conn)
            return df
        except snowflake.connector.Error as e:
            print(f"Error reading embedding from Snowflake: {e}")
            # Handle or log the error, and raise it if necessary
            raise

        except Exception as e:
            print(f"Unexpected error: {e}")
            # Handle or log the error, and raise it if necessary
            raise

    def test_connection(self) -> bool:
        """Tests opening a connection to Snowflake and performing a basic SELECT query."""
        try:
            conn = self.get_snowflake_connection()
            query = 'SELECT * FROM "github_to_vector_text" LIMIT 10'
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
