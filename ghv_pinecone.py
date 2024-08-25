import os
from pinecone import Pinecone, ServerlessSpec, UpsertResponse, UnauthorizedException
from dotenv import load_dotenv
from typing import List, Dict, Any
from ghv_openai import GhvOpenAI
from ghv_snowflake import GhvSnowflake
from ghv_github import GhvGithub
import pandas as pd
from datetime import datetime
import re


class GhvPinecone:
    """
    Class for interacting with Pinecone to store and query embeddings.
    Sequence:   1 - Constructor: Initialize Pinecone client
                2 - Create or reuse existing index
                3 - Upsert vectors (from model embeddings)
                4 - Query vector (with a search embedding)

    Test mode:  If PINECONE_TEST_MODE is set to True, all indexes in the project are deleted.
    This is transparent to the client. We delete everything during GhvPinecone initialization.
    """
    # Class constants for Pinecone pricing
    STORAGE_COST_PER_GB_PER_HOUR = 0.00045
    WRITE_COST_PER_MILLION_UNITS = 2.00
    READ_COST_PER_MILLION_UNITS = 8.25
    BYTES_PER_DIMENSION = 4

    def __init__(self, embedding_model_name="", base_index_name: str = "", metric: str = "", dimension: int = 0):
        load_dotenv()  # Load environment variables from .env file
        self.api_key: str = os.getenv("PINECONE_API_KEY")
        self.project_name: str = os.getenv(
            "PINECONE_PROJECT_NAME", "ghvector")
        # These call all be overriden with create_index
        if metric == "":
            self.metric: str = os.getenv("PINECONE_METRIC", "cosine")
        else:
            self.metric = metric
        if dimension == 0:
            self.dimension = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
        else:
            self.dimension = dimension
        if base_index_name == "":
            self.base_index_name: str = os.getenv(
                "PINECONE_BASE_INDEX_NAME", "ghv")
        else:
            self.base_index_name = base_index_name
        if embedding_model_name == "":
            self.embedding_model_name: str = GhvOpenAI.embedding_model_default
        else:
            self.embedding_model_name = embedding_model_name

        self.pc = Pinecone(api_key=self.api_key)

        # Until these vars are set, we don't have a valid index and can't upsert, etc.
        # So we'll set them in the get_and_prep_index function
        self.index_name = ""
        self.index_description = None
        self.index_host = ""
        self.index = None

    def delete_all_indexes(self):
        """
        Deletes all indexes in the Pinecone project. DANGER!!
        """
        # Test mode: delete all indexes in the project
        if os.getenv("PINECONE_TEST_MODE", "false").lower() == "true":
            self._delete_all_indexes()

    def get_and_prep_index(self) -> str:
        """
        Creates or reuses a Pinecone index based on the specified embedding model and dimensions.
        """
        self.index_name = self._create_or_reuse_index(
            self.embedding_model_name, self.dimension)
        self.index_description: str = self.pc.describe_index(self.index_name)
        self.index_host = self.index_description.host
        self.index = self.pc.Index(self.index_name, host=self.index_host)
        return self.index_name

    def _check_index_exists(self, index_name: str) -> bool:
        """
        Checks if a Pinecone index with the given name already exists.

        Args:
            index_name (str): The name of the index to check.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return index_name in self.pc.list_indexes().names()

    def _create_or_reuse_index(self, embedding_model_name: str, dimension: int) -> str:
        """
        Creates a new Pinecone index or reuses an existing one if it already exists.

        Args:
            model_name (str): The name of the embedding model being used, included in the index name.
            dimensions (int): The number of dimensions for the embedding vectors.
        """
        # Construct and clean the initial index name
        index_name = f"{self.base_index_name}_{
            embedding_model_name[:20]}_{dimension}"
        cleaned_index_name = re.sub(r'[^a-z0-9\-]', '-', index_name.lower())
        self.index_name = cleaned_index_name[:45]

        # Check if the index already exists using the check_index_exists function
        if self._check_index_exists(self.index_name):
            print(
                f"\tIndex '{self.index_name}' already exists. Reusing the existing index.")
        else:
            # Create the new index if it doesn't exist
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            print(f"\tCreated Pinecone index '{self.index_name}' with {
                  dimension} dimensions and '{self.metric}' metric.")
        return self.index_name

    def _connect_to_index(self):
        """
        Connects to the Pinecone index or creates it if it doesn't exist.
        """
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """
        Upserts a batch of vectors into the Pinecone index.

        Args:
            vectors (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
                - 'id': The unique ID for the vector.
                - 'values': The vector embedding.
                - 'metadata': Optional metadata associated with the vector.
        """
        if not self.index:
            print("\tNo index connected. Please connect to an index first.")
            return 0
        # vector_length: int = len(vectors[0].get("values", []))
        num_vectors: int = len(vectors)
        upsert_response: UpsertResponse = self.index.upsert(vectors)
        upserted_count: int = upsert_response.get("upserted_count", 0)
        if upserted_count != num_vectors:
            print(f"\tMismatch between upserted vectors and total vectors. Upserted {
                  upserted_count} != {num_vectors}")
        # else:
        #     print(f"\tUpserted {upserted_count} vector of length {vector_length} to Pinecone index: {
        #       self.index_name}")
        return upserted_count

    def query_vector(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index with a vector and returns the top_k most similar vectors.

        Args:
            vector (List[float]): The vector to query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of the top_k most similar vectors.
        """
        if not self.index:
            print("\tNo index connected. Please connect to an index first.")
            return []
        try:
            # Perform the query
            result = self.index.query(vector=vector, top_k=top_k)
            print(f"Query result: {result['matches']}")
            return result['matches']  # Return the list of matches

        except UnauthorizedException as e:
            # Handle unauthorized access specifically
            print(f"UnauthorizedException: {e}")
            print(
                "Please check your API key and ensure you have the necessary permissions.")
            raise  # Re-raise the exception to signal the failure

        except Exception as e:
            # General exception handling
            print(f"An error occurred during the query: {e}")
            raise  # Re-raise the exception to ensure it's not silently ignored

    def delete_vector(self, vector_id: str):
        """
        Deletes a vector from the Pinecone index by its ID.

        Args:
            vector_id (str): The ID of the vector to delete.
        """
        if not self.index:
            print("\tNo index connected. Please connect to an index first.")
            return
        self.index.delete(ids=[vector_id])
        print(f"Deleted vector with ID: {
              vector_id} from Pinecone index: {self.index_name}")

    def fetch_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Fetches a vector from the Pinecone index by its ID.

        Args:
            vector_id (str): The ID of the vector to fetch.

        Returns:
            Dict[str, Any]: The fetched vector data.
        """
        if not self.index:
            print("\tNo index connected. Please connect to an index first.")
            return {}
        result = self.index.fetch(ids=[vector_id])  # Fetch the vector by ID
        return result  # Return the fetched vector data

    def describe_index(self):
        """
        Describe the index to get stats and other metadata.

        Returns:
            Dict[str, Any]: The stats of the Pinecone index.
        """
        if not self.index:
            print("\tNo index connected. Please connect to an index first.")
            return {}
        return self.index.describe_index_stats()  # Get and return the index stats

    def _delete_all_indexes(self):
        """
        Deletes all indexes in the Pinecone project. DANGER!!
        """
        # List all indexes in the project
        indexes = self.pc.list_indexes().names()

        if not indexes:
            print("No indexes found in the project.")
            return

        # Iterate through the list of indexes and delete each one
        count: int = 0
        for index_name in indexes:
            print(f"\r\tDeleting index: {index_name}", end="")
            self.pc.delete_index(name=index_name)
            count += 1

        print(f"\n\t{count} indexes have been deleted.")

    def calculate_storage_cost(self, dimensions: int, num_vectors: int = 1, hours: int = 1) -> float:
        """
        Calculates the cost of storing vectors in Pinecone.

        Args:
            num_vectors (int): The number of vectors being stored.
            hours (int): The number of hours the vectors are stored (default is 1 hour).

        Returns:
            float: The estimated cost of storing the vectors for the specified time.
        """
        # Convert dimensions to bytes
        vector_size_bytes = dimensions * self.BYTES_PER_DIMENSION
        bytes_per_gb = 1024 * 1024 * 1024

        # Calculate total storage in GB
        total_storage_gb = (num_vectors * vector_size_bytes) / bytes_per_gb

        # Total storage cost
        return total_storage_gb * self.STORAGE_COST_PER_GB_PER_HOUR * hours

    def calculate_write_cost(self, num_vectors: int = 1) -> float:
        """
        Calculates the cost of writing vectors to Pinecone.

        Args:
            num_vectors (int): The number of vectors being written.

        Returns:
            float: The estimated cost of writing the vectors.
        """
        # Pinecone charges $2.00 per 1M Write Units
        write_units = num_vectors / 1_000_000
        return write_units * self.WRITE_COST_PER_MILLION_UNITS

    def calculate_read_cost(self, num_queries: int = 1) -> float:
        """
        Calculates the cost of querying vectors in Pinecone.

        Args:
            num_queries (int): The number of queries made.

        Returns:
            float: The estimated cost of querying the vectors.
        """
        # Pinecone charges $8.25 per 1M Read Units
        read_units = num_queries / 1_000_000
        return read_units * self.READ_COST_PER_MILLION_UNITS


def gen_embedding_and_upsert(ghv_pc: GhvPinecone, dict_test: Dict, ghv_openai: GhvOpenAI) -> int:
    """
    Stores a test vector from the test dictionary.

    Args:
        ghv_openai (GhvOpenAI): The GhvOpenAI instance used to generate embeddings.
    """

    # Generate embedding for the dummy function using GhvOpenAI
    function_response = ghv_openai.generate_embeddings(dict_test["text"])

    # Access the embedding vector directly from the response
    function_embedding = function_response['embedding']

    # Upsert the function embedding to Pinecone
    upserted_count: int = ghv_pc.upsert_vectors([{
        "id": dict_test["id"],
        "values": function_embedding,
        "metadata": dict_test["metadata"]
    }])
    return upserted_count


def test_embedding_search(ghv_pc: GhvPinecone, dict_test: Dict, ghv_openai: GhvOpenAI) -> pd.DataFrame:
    """
    Tests the vector query by generating an embedding for a dummy function and querying with a related prompt.

    Args:
        ghv_openai (GhvOpenAI): The GhvOpenAI instance used to generate embeddings.
    """

    query_response = ghv_openai.generate_embeddings(dict_test["prompt"])

    # Access the embedding vector directly from the response
    query_embedding = query_response['embedding']

    # Query Pinecone using the query embedding
    print(f"\tQuerying Pinecone {ghv_pc.index_name} with the prompt: '{
        dict_test['prompt']}'")

    results = ghv_pc.query_vector(vector=query_embedding, top_k=5)

    pricing_per_token: float = ghv_openai.embedding_models[ghv_openai.embedding_model].get(
        "pricing_per_token", 0.0)
    num_tokens: int = ghv_openai.count_tokens(
        dict_test["text"]) + ghv_openai.count_tokens(dict_test["prompt"])
    # Structure results for DataFrame compatibility
    structured_results = []
    for result in results:
        row = {
            "embedding_model": ghv_openai.embedding_model,
            "dimensions": ghv_openai.dimensions,
            "index_name": ghv_pc.index_name,
            "prompt": dict_test["prompt"],
            "text": dict_test["text"],
            "result_id": result.get("id", ""),
            "score": float(result.get("score", 0.0)),
            "num_tokens": num_tokens,
            "cost": float(pricing_per_token) * num_tokens,
            "vectordb_storage_cost": ghv_pc.calculate_write_cost() + ghv_pc.calculate_storage_cost(ghv_openai.dimensions),
            "vectordb_read_cost": ghv_pc.calculate_read_cost(),
        }
        structured_results.append(row)

    # Convert structured results into a DataFrame
    results_df = pd.DataFrame(structured_results)

    return results_df


if __name__ == "__main__":
    # Initialize an empty DataFrame to hold all results
    all_results_df = pd.DataFrame()

    # Static list of embeddings with text, prompt, ID, and metadata
    embedding_tests: List[Dict] = [
        {
            "id": "embedding-test-1",
            "text": """
            Vector databases are designed to store and retrieve high-dimensional vectors, which are generated by models like those used for embeddings.
            These databases use metrics such as cosine similarity or dot product to determine the closeness of vectors. Properly tuning these metrics is crucial for optimal performance.
            Pinecone is a popular vector database service that supports scalable and real-time vector search.
            """,
            "prompt": "What are the key considerations when choosing a metric for a vector database?",
            "metadata": {"description": "Test for vector database metric choice"}
        },
        {
            "id": "embedding-test-2",
            "text": """
            When integrating embeddings with a vector database, it is essential to ensure that the dimensions of the embeddings align with the database configuration.
            Mismatched dimensions can lead to errors and suboptimal performance. Additionally, the choice of embedding model can significantly impact both the accuracy and efficiency of your search queries.
            """,
            "prompt": "How do you ensure that the dimensions of embeddings align with the database configuration?",
            "metadata": {"description": "Test for embedding dimension alignment"}
        },
        {
            "id": "embedding-test-3",
            "text": """
            Embedding models

            Which embedding model you use has a lot to do with the type of content you are vectorizing. Price should be considered also.
            The dimensions value must match your Pinecone settings also.

            Learn about embeddings

            Model                  | Dimensions | Pricing            | Pricing with Batch API
            ---------------------- | ---------- | ------------------ | ----------------------
            text-embedding-3-small | 1,536      | $0.020 / 1M tokens | $0.010 / 1M tokens     
            text-embedding-3-large | 3,072      | $0.130 / 1M tokens | $0.065 / 1M tokens     
            ada v2                 | 1,536      | $0.100 / 1M tokens | $0.050 / 1M tokens     
            """,
            "prompt": "How do I choose the right embedding model?",
            "metadata": {"description": "Test for embedding model selection"}
        },
        {
            "id": "embedding-test-4",
            "text": """def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr""",
            "prompt": "How can I sort an array using the bubble sort algorithm?",
            "metadata": {"description": "Test for bubble sort algorithm"}
        },
        {
            "id": "embedding-test-5",
            "text": """def binary_search(arr, x):
            low = 0
            high = len(arr) - 1
            while low <= high:
                mid = (high + low) // 2
                if arr[mid] < x:
                    low = mid + 1
                elif arr[mid] > x:
                    high = mid - 1
                else:
                    return mid
            return -1""",
            "prompt": "How do I implement a binary search algorithm?",
            "metadata": {"description": "Test for binary search algorithm"}
        },
        {
            "id": "embedding-test-6",
            "text": """def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)""",
            "prompt": "What is the factorial of a number, and how do I compute it recursively?",
            "metadata": {"description": "Test for factorial calculation"}
        },
        {
            "id": "embedding-test-7",
            "text": """def fibonacci(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a""",
            "prompt": "Can you generate the nth Fibonacci number using a loop?",
            "metadata": {"description": "Test for Fibonacci number generation"}
        },
        {
            "id": "embedding-test-8",
            "text": """def gcd(a, b):
            while b:
                a, b = b, a % b
            return a""",
            "prompt": "How do I find the greatest common divisor (GCD) of two numbers?",
            "metadata": {"description": "Test for GCD calculation"}
        },
        {
            "id": "embedding-test-9",
            "text": """def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True""",
            "prompt": "Is there a way to determine if a number is prime?",
            "metadata": {"description": "Test for prime number detection"}
        },
        {
            "id": "embedding-test-10",
            "text": """def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + middle + quicksort(right)""",
            "prompt": "What is the best way to implement the quicksort algorithm?",
            "metadata": {"description": "Test for quicksort algorithm"}
        },
        {
            "id": "embedding-test-11",
            "text": """def palindrome_check(s):
            return s == s[::-1]""",
            "prompt": "How do I check if a string is a palindrome?",
            "metadata": {"description": "Test for palindrome check"}
        },
        {
            "id": "embedding-test-12",
            "text": """def sum_of_array(arr):
            total = 0
            for num in arr:
                total += num
            return total""",
            "prompt": "How can I calculate the sum of all elements in an array?",
            "metadata": {"description": "Test for sum of array"}
        },
        {
            "id": "embedding-test-13",
            "text": """def try_catch_example():
            try:
                result = 10 / 0
            except ZeroDivisionError:
                print("Cannot divide by zero")
            finally:
                print("Execution completed")""",
            "prompt": "How can I handle exceptions in Python, like dividing by zero?",
            "metadata": {"description": "Test for exception handling in Python"}
        }
    ]

    # Delete all indexes in the project if test mode is enabled
    delete_indexes: GhvPinecone = GhvPinecone()
    delete_indexes.delete_all_indexes()

    # Cache the GhvOpenAI and GhvPinecone instances for each model since we reuse them
    dicts: Dict = {}
    ghv_openai_client: GhvOpenAI = None
    pinecone_client: GhvPinecone = None
    snowflake_client: GhvSnowflake = GhvSnowflake()
    github_client = GhvGithub()

    # Initialize the Snowflake row dictionary which we'll update in the loop below then store
    snowflake_row: Dict[str, Any] = {
        "org_name": github_client.repo_owner,          # GitHub organization name
        "repo_name": "test",                # Repository name within the organization
        "file_name": "test.py",             # File name where the code chunk resides
        "line_start": 0,                    # Starting line number of the code chunk
        "line_end": 0,                      # Ending line number of the code chunk
        # Text that is vectorized and stored as embedding in Pinecone
        "text": "",
        # String-based identifier for the embedding in Pinecone
        "embedding_id": "",
        "index_name": "",    # Name of the Pinecone index where the embedding is stored
        # Timestamp for when the record was stored (None means use the default current timestamp)
        "storage_datetime": None
    }

    # Iterate over each embedding test and generate embeddings for each model
    # Then upsert the embeddings to Pinecone and store the embeddings reference info in Snowflake
    for model, model_settings in GhvOpenAI.embedding_models.items():
        dicts[model] = {"openai": None, "pinecone": None}
        # Initialize OpenAI client with the specific model and dimensions
        ghv_openai_client = GhvOpenAI(
            model=model, dimensions=model_settings["dimensions"])
        dicts[model]["openai"] = ghv_openai_client
        pinecone_client = GhvPinecone(
            embedding_model_name=model, dimension=model_settings["dimensions"])
        index_name = pinecone_client.get_and_prep_index()
        dicts[model]["pinecone"] = pinecone_client
        print(f"\tInserting embeddings with model: {model}")

        count: int = 0
        for embedding_test in embedding_tests:
            count += gen_embedding_and_upsert(pinecone_client,
                                              embedding_test, ghv_openai_client)
            snowflake_row["text"] = embedding_test["text"]
            snowflake_row["embedding_id"] = embedding_test["id"]
            snowflake_row["index_name"] = index_name
            snowflake_client.store_single_embedding(
                snowflake_row)

            print(f"\r\tInsert count: {count}", end="")
        print("\n")

    # Iterate over each embedding test and query the embeddings for each model
    for model, model_settings in GhvOpenAI.embedding_models.items():
        print(f"\nTesting prompt queries with model: {model}")

        ghv_openai_client = dicts.get(model, None).get("openai", None)
        pinecone_client = dicts.get(model, None).get("pinecone", None)

        count: int = 0
        for embedding_test in embedding_tests:
            results_df = test_embedding_search(pinecone_client,
                                               embedding_test, ghv_openai_client)

            # Merge the results into the overall DataFrame
            all_results_df = pd.concat(
                [all_results_df, results_df], ignore_index=True)

    # For each row in the dataframe, query the snowflake table for the embedding data
    # This is the ultimate test to tie back the scored results to the original text and evaluate the quality of the match
    for index, row in all_results_df.iterrows():
        # Query the Snowflake table for the embedding data
        embedding_data = snowflake_client.read_embedding_by_id(
            row["result_id"])

        if not embedding_data.empty:
            # Extract the "text" column and store it in the "original_text" column
            all_results_df.at[index,
                              "original_text"] = embedding_data["text"].iloc[0]
        else:
            all_results_df.at[index, "original_text"] = None

    # Sort the final DataFrame by score in descending order, since that's what we're interested in
    all_results_df = all_results_df.sort_values(by="score", ascending=False)
    # Print and save the final DataFrame after all iterations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pinecone_client.index_name}_all_models_{timestamp}.csv"
    all_results_df.to_csv(filename, index=False)
    print(f"Saved all results to {filename}")

    # Print the final merged results
    print("Final Query Results:")
    print(all_results_df)

    # Describe index stats to check the current state of the index
    index_stats = pinecone_client.describe_index()
    print("Index Stats:")
    print(index_stats)
