import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict, Any
from ghv_openai import GhvOpenAI
import pandas as pd
from datetime import datetime


class GhvPinecone:
    # Static list of embedding models with their settings
    embedding_models = [
        {"model_name": "text-embedding-ada-002", "pricing_per_token": 0.0004},
        {"model_name": "text-embedding-3-small", "pricing_per_token": 0.00025},
        {"model_name": "text-embedding-3-large", "pricing_per_token": 0.0005}
    ]

    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv(
            "PINECONE_INDEX_NAME", "ghvector-file-chunks")
        self.metric = os.getenv("PINECONE_METRIC", "cosine")

        self.pc = Pinecone(api_key=self.api_key)
        self._connect_to_index()

    def create_index(self, model_name: str, dimensions: int):
        """
        Creates a new Pinecone index
        If an index with the specified name already exists, appends a numeric suffix to create a unique name.

        Args:
            model_name (str): The name of the embedding model being used, included in the index name.
            dimensions (int): The number of dimensions for the embedding vectors.
        """

        # Generate a timestamp to ensure the index name is unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.index_name = f"{self.index_name}_{
            model_name}_{dimensions}_{timestamp}"

        # Create the new index with the (potentially modified) name
        self.pc.create_index(
            name=self.index_name,
            dimension=dimensions,  # Use the specific dimensions for this model
            metric=self.metric,
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )
        print(f"Created Pinecone index '{self.index_name}' with {
              dimensions} dimensions and '{self.metric}' metric.")

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

    def test_query_vector_with_openai(self, model_name: str, dimensions: int, text: str, prompt: str, ghv_openai: GhvOpenAI) -> pd.DataFrame:
        """
        Tests the vector query by generating an embedding for a dummy function and querying with a related prompt.
        Outputs the results to a DataFrame.

        Args:
            model_name (str): The name of the embedding model being tested.
            dimensions (int): The number of dimensions for the embedding vectors.
            text (str): The text to generate the embedding from.
            prompt (str): The prompt to query against the generated embedding.
            ghv_openai (GhvOpenAI): The GhvOpenAI instance used to generate embeddings.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the query.
        """
        # Set the specific model and dimensions for this test
        ghv_openai.embedding_model = model_name
        ghv_openai.dimensions = dimensions

        # Generate embedding for the dummy function using GhvOpenAI
        function_response = ghv_openai.generate_embeddings(text)
        function_embedding = function_response['embedding']

        # Upsert the function embedding to Pinecone
        self.upsert_vectors([{
            "id": f"{model_name}-dummy-function-id",
            "values": function_embedding,
            "metadata": {"description": "Dummy function for testing"}
        }])

        # Generate embedding for the prompt
        query_response = ghv_openai.generate_embeddings(prompt)
        query_embedding = query_response['embedding']

        # Query Pinecone using the query embedding
        print(f"Querying Pinecone index with the prompt: '{prompt}'")
        results = self.query_vector(vector=query_embedding, top_k=5)

        # Collect results into a DataFrame for comparative analysis
        results_df = pd.DataFrame(results)
        results_df['embedding_model'] = model_name
        results_df['dimensions'] = dimensions
        results_df['prompt'] = prompt
        results_df['text'] = text
        return results_df

    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Upserts a batch of vectors into the Pinecone index.

        Args:
            vectors (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
                - 'id': The unique ID for the vector.
                - 'values': The vector embedding.
                - 'metadata': Optional metadata associated with the vector.
        """
        self.index.upsert(vectors)
        print(f"Upserted {len(vectors)} vectors to Pinecone index: {
              self.index_name}")

    def query_vector(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index with a vector and returns the top_k most similar vectors.

        Args:
            vector (List[float]): The vector to query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of the top_k most similar vectors.
        """
        result = self.index.query(
            vector=vector, top_k=top_k)  # Perform the query
        return result['matches']  # Return the list of matches

    def delete_vector(self, vector_id: str):
        """
        Deletes a vector from the Pinecone index by its ID.

        Args:
            vector_id (str): The ID of the vector to delete.
        """
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
        result = self.index.fetch(ids=[vector_id])  # Fetch the vector by ID
        return result  # Return the fetched vector data

    def describe_index(self):
        """
        Describe the index to get stats and other metadata.

        Returns:
            Dict[str, Any]: The stats of the Pinecone index.
        """
        return self.index.describe_index_stats()  # Get and return the index stats

    def test_query_vector_with_openai(self, text: str, prompt: str, ghv_openai: GhvOpenAI) -> pd.DataFrame:
        """
        Tests the vector query by generating an embedding for a dummy function and querying with a related prompt.
        Outputs the results to a DataFrame and saves it as a CSV file.

        Args:
            ghv_openai (GhvOpenAI): The GhvOpenAI instance used to generate embeddings.
        """

        # Generate embedding for the dummy function using GhvOpenAI
        function_response = ghv_openai.generate_embeddings(text)

        # Access the embedding vector directly from the response
        function_embedding = function_response['embedding']

        # Upsert the function embedding to Pinecone
        self.upsert_vectors([{
            "id": "dummy-function-id",
            "values": function_embedding,
            "metadata": {"description": "Dummy function with try-catch block"}
        }])

        query_response = ghv_openai.generate_embeddings(prompt)

        # Access the embedding vector directly from the response
        query_embedding = query_response['embedding']

        # Query Pinecone using the query embedding
        print(f"Querying Pinecone index with the prompt: '{prompt}'")
        results = self.query_vector(vector=query_embedding, top_k=5)

        # Collect results into a DataFrame for comparative analysis
        results_df = pd.DataFrame(results)
        results_df['embedding_model'] = ghv_openai.embedding_model
        results_df['dimensions'] = ghv_openai.dimensions
        results_df['prompt'] = prompt
        results_df['text'] = text
        return results_df


if __name__ == "__main__":
    pinecone_client = GhvPinecone()
    ghv_openai_client = GhvOpenAI()

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

    for model in GhvOpenAI.embedding_models:
        print(f"Testing with model: {model['model_name']}")
        pinecone_client.create_index(
            model_name=model["model_name"], dimensions=model["dimensions"])

        # Initialize OpenAI client with the specific model and dimensions
        ghv_openai_client = GhvOpenAI(
            model=model["model_name"], dimensions=model["dimensions"])

        total_tokens = 0

        for embedding_test in embedding_tests:
            results_df = pinecone_client.test_query_vector_with_openai(
                embedding_test, ghv_openai_client)
            total_tokens += len(embedding_test["text"].split()) + \
                len(embedding_test["prompt"].split())

            # Merge the results into the overall DataFrame
            all_results_df = pd.concat(
                [all_results_df, results_df], ignore_index=True)

        # Estimate and print the cost for this model
        cost = total_tokens * model["pricing_per_token"]
        print(f"Estimated cost for model {model['model_name']}: ${cost:.6f}")

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
