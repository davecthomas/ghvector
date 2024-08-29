# ghv_openai.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse
import tiktoken


class GhvOpenAI:
    # Static list of embedding models with their settings
    embedding_models: Dict = {
        "text-embedding-ada-002": {"dimensions": 1536, "pricing_per_token": 0.0004},
        "text-embedding-3-small": {"dimensions": 1536, "pricing_per_token": 0.00025},
        "text-embedding-3-large": {"dimensions": 3072, "pricing_per_token": 0.0005}}

    embedding_model_default = "text-embedding-3-small"

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
        self.completions_model = os.getenv(
            "OPENAI_COMPLETIONS_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text based on a generic tokenizer cl100k_base.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        # Get the appropriate tokenizer for the model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and count the tokens
        tokens = tokenizer.encode(text)
        return len(tokens)

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
            "model": self.embedding_model,  # Model to use for embedding
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

    def sendPrompt(self, prompt: str) -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.

        Returns:
            str: The completion result as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.completions_model,
                messages=[
                    {"role": "system",
                        "content": "You are a helpful software coding assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response from the completion
            completion = response.choices[0].message.content

            # If the content seems truncated, send a follow-up request or handle continuation
            while response.choices[0].finish_reason == 'length':
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {"role": "system", "content": "Continue."},
                    ]
                )
                completion += response.choices[0].message.content
            return completion

        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise

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
