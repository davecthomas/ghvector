# ghvector_main.py

from ghv_github import GhvGithub
from ghv_openai import GhvOpenAI
from ghv_snowflake import GhvSnowflake
from dotenv import load_dotenv
import os
from typing import List, Dict


class GhvMain:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.github_client = GhvGithub()
        self.openai_client = GhvOpenAI()
        self.snowflake_client = GhvSnowflake()

    def get_test_repo(self) -> str:
        """
        Retrieves the first repository name from the REPO_NAMES list in the .env file.

        Returns:
            str: The name of the first repository in the list.
        """
        repo_names = os.getenv("REPO_NAMES", "").split(",")
        if not repo_names:
            raise ValueError("No repositories found in REPO_NAMES.")
        return repo_names[0].strip()

    def process_repository(self, repo_name: str):
        """
        Orchestrates the process of fetching files, generating embeddings, and storing them.

        Args:
            repo_name (str): The name of the repository to process.
        """
        # Fetch file chunks from the specified repository
        print(f"Fetching file chunks from repository: {repo_name}")
        file_chunks: List[str] = self.github_client.get_file_chunks(
            file_info, test_mode=True)

        # Generate embeddings for each chunk and store in Snowflake
        for file_info, chunks in file_chunks.items():
            print(f"Processing file: {file_info}")
            embeddings = self.openai_client.process_file_chunks(
                chunks, file_info)

            # Store the embeddings in Snowflake
            print(f"Storing embeddings for file: {file_info}")
            self.snowflake_client.store_embedding_data(embeddings)

        print(f"Processing of repository {repo_name} completed.")

    def test_workflow(self):
        """
        Test method to verify the entire workflow using the first repository in REPO_NAMES.
        """
        repo_name = self.get_test_repo()
        self.process_repository(repo_name)


if __name__ == "__main__":
    # Initialize the main class
    main_processor = GhvMain()

    # Test the entire workflow
    main_processor.test_workflow()
