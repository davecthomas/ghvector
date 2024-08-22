# ghv_openai.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse


class GhvOpenAI:
    # Static list of embedding models with their settings
    embedding_models = [
        {"model_name": "text-embedding-ada-002",
            "dimensions": 1536, "pricing_per_token": 0.0004},
        {"model_name": "text-embedding-3-small",
            "dimensions": 512, "pricing_per_token": 0.00025},
        {"model_name": "text-embedding-3-large",
            "dimensions": 1536, "pricing_per_token": 0.0005}
    ]

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512):
        """
        Initializes the GhvOpenAI class, setting the model and dimensions.

        Args:
            model (str): The embedding model to use.
            dimensions (int): The number of dimensions for the embeddings.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = model  # Use the model passed to the constructor
        self.dimensions = dimensions  # Use the dimensions passed to the constructor
        self.user = os.getenv("OPENAI_USER", "default_user")
        self.client = OpenAI(api_key=self.api_key)

    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generates embeddings for a given text using OpenAI's embeddings API.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings and metadata.
        """
        # Construct the parameters using EmbeddingCreateParams
        params: EmbeddingCreateParams = {
            "input": [text],  # Input text as a list of strings
            "model": self.embedding_model,  # Model to use
            "dimensions": self.dimensions,  # Number of dimensions
        }

        # Call the API to create the embedding
        response: CreateEmbeddingResponse = self.client.embeddings.create(
            **params)

        # Extract the embedding from the response
        embedding = response.data[0].embedding
        return {
            "embedding": embedding,
            "text": text,
            "dimensions": self.dimensions,  # Dimensions of the embedding
            "user": self.user  # User identifier
        }

    def process_file_chunks(self, file_chunks: List[str], file_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Processes a list of file chunks by generating embeddings for each chunk.

        Args:
            file_chunks (List[str]): A list of text chunks from a file.
            file_info (Dict[str, str]): A dictionary containing information about the file (repo, path, etc.).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing embeddings and related metadata.
        """
        embeddings_list = []
        for chunk in file_chunks:
            embedding_data = self.generate_embeddings(chunk)
            embedding_data.update({
                "repo": file_info["repo"],
                "folder": file_info["folder"],
                "path": file_info["path"],
                "file_name": file_info["file_name"],
                "chunk": chunk
            })
            embeddings_list.append(embedding_data)

        return embeddings_list

    def test_github_openai_integration(self, github_client, file_info: Dict[str, str]):
        """
        Tests the integration between the GitHub and OpenAI classes by processing
        file chunks and generating embeddings for them.

        Args:
            github_client (GhsGithub): An instance of the GhsGithub class.
            file_info (Dict[str, str]): A dictionary containing information about the file (repo, path, etc.).
        """
        # Fetch chunks from the GitHub file
        file_chunks = github_client.get_file_chunks(file_info)

        # Generate embeddings for each chunk
        embeddings_data = self.process_file_chunks(file_chunks, file_info)

        # Print the results for testing purposes
        # Limit output to the first 3 chunks
        for i, embedding_data in enumerate(embeddings_data[:3]):
            print(f"\nEmbedding {
                  i+1} for file {file_info['file_name']} in {file_info['repo']}:")
            print(f"Chunk: {embedding_data['chunk']}")
            # Truncate embedding for display
            print(f"Embedding: {
                  embedding_data['embedding'][:5]}... [truncated]")
            print("--- End of Embedding ---")

    def test_openai_connectivity(self, list_text: List[str] = ["Hello, World!"]) -> bool:
        """
        Tests connectivity to the OpenAI API by generating embeddings for the provided text.

        Args:
            list_text (List[str]): A list of text strings to embed. Defaults to ["Hello, World!"].

        Returns:
            bool: True if the API request is successful, False otherwise.
        """
        try:
            for text in list_text:
                embedding_data = self.generate_embeddings(text)
                print(f"Successfully generated embedding for: {text}")
                print(f"Embedding: {
                      embedding_data['embedding'][:5]}... [truncated]")

            return True

        except Exception as e:
            print(f"OpenAI API connectivity test failed with error: {e}")
            return False


if __name__ == "__main__":
    # Initialize the OpenAI client
    openai_client = GhvOpenAI()

    # Test connectivity to the OpenAI API
    connectivity_test_result = openai_client.test_openai_connectivity()

    # Output the result of the connectivity test
    print(f"Connectivity Test Passed: {connectivity_test_result}")

    # Add more tests or calls here if needed
