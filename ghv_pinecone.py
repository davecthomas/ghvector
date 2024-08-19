import os
import random
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict, Any


class GhvPinecone:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv(
            "PINECONE_INDEX_NAME", "github-code-chunks")
        # Default dimension
        self.dimension = int(os.getenv("PINECONE_DIMENSION", 512))
        self.metric = os.getenv("PINECONE_METRIC", "cosine")  # Default metric

        # Initialize Pinecone client using the Pinecone class
        self.pc = Pinecone(api_key=self.api_key)

        # Create or connect to the index
        self._connect_to_index()

    def _connect_to_index(self):
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
            vector=vector, top_k=top_k)  # Use keyword arguments here
        return result['matches']

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
        result = self.index.fetch(ids=[vector_id])
        return result

    def describe_index(self):
        """
        Describe the index to get stats and other metadata.

        Returns:
            Dict[str, Any]: The stats of the Pinecone index.
        """
        return self.index.describe_index_stats()


if __name__ == "__main__":
    # Example usage
    pinecone_client = GhvPinecone()

    # Generate a random 512-dimensional vector
    vector_data = [
        {
            "id": "example-vector-id",
            # Random values between -1 and 1
            "values": [random.uniform(-1, 1) for _ in range(512)],
            "metadata": {"file": "example.java", "line_range": "10-20"}
        }
    ]

    # Upsert the vector to Pinecone
    pinecone_client.upsert_vectors(vector_data)

    # Fetching the vector to verify it was stored correctly
    fetched_vector = pinecone_client.fetch_vector(vector_data[0]['id'])
    print(f"Fetched vector details after upsert: {fetched_vector}")

    # Querying the vector (replace with an actual query vector)
    # Another random 512-dimensional vector
    query_vector = [random.uniform(-1, 1) for _ in range(512)]
    query_result = pinecone_client.query_vector(vector=query_vector, top_k=10)
    print("Query result:")
    for match in query_result:
        print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {
              match.get('metadata', 'No metadata')}")

    # Describe index stats to check the current state of the index
    index_stats = pinecone_client.describe_index()
    print("Index Stats:")
    print(index_stats)
