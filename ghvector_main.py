# ghvector_main.py

from ghv_github import GhvGithub
from ghv_openai import GhvOpenAI
from ghv_snowflake import GhvSnowflake
from ghv_pinecone import GhvPinecone
from ghv_chunker import GhvChunker
from dotenv import load_dotenv
import os
from typing import Any, List, Dict


class GhvMain:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.github_client = GhvGithub()
        self.openai_client = GhvOpenAI()
        self.snowflake_client = GhvSnowflake()
        self.pinecone_client = GhvPinecone()
        # Conditionally delete all indexes (test mode)
        self.pinecone_client.delete_all_indexes()
        self.repo_names = os.getenv("REPO_NAMES", "").split(",")
        if not self.repo_names:
            self.repo_names = []
            raise ValueError("No repositories found in REPO_NAMES.")

    def get_test_repo(self) -> str:
        """
        Retrieves the first repository name from the REPO_NAMES list in the .env file.

        Returns:
            str: The name of the first repository in the list.
        """
        return self.repo_names[0].strip()

    def process_repository(self, repo_name: str):
        """
        Orchestrates the process of fetching files, generating embeddings, and storing them.

        Args:
            repo_name (str): The name of the repository to process.
        """
        # Step 1: List all files in the repository
        print(f"Listing files in repository: {repo_name}")
        files_list_dict = self.github_client.list_files_in_repo(repo_name)
        if not files_list_dict:
            print(f"No files found in repository: {repo_name}")
            return
        # Remove files that are not supported by the chunker
        files_list_dict = [file_dict for file_dict in files_list_dict if os.path.splitext(
            file_dict.get("path", ".xyz"))[1] in GhvChunker.get_supported_file_types()]

        # Step 2: Process each file
        for file_info in files_list_dict:
            print(f"\tFetching chunks from file: {file_info['path']}")
            file_chunks: List[str] = None
            file_chunks = self.github_client.get_file_chunks(file_info)

            # Generate embeddings for each chunk and store in Snowflake
            if file_chunks:
                embeddings = self.openai_client.process_file_chunks(
                    file_chunks, file_info)

                list_dict_upsert: List[Dict[str, Any]] = []
                list_dict_snowflake: List[Dict[str, Any]] = []
                for i, embedding in enumerate(embeddings):
                    embedding_id = self.github_client.repo_owner + "/" + embedding["repo"] + "/" + (
                        f"_{embedding['folder']}/" if embedding["folder"] else "") + embedding["path"] + ":" + str(i)
                    list_dict_upsert.append({
                        "id": embedding_id,
                        "values": embedding["embedding"],
                        "metadata": {"chunk": embedding["chunk"]}
                    })
                    list_dict_snowflake.append({
                        "org_name": self.github_client.repo_owner,
                        "repo_name": embedding["repo"],
                        "file_folder": embedding["folder"],
                        "file_name": embedding["path"],
                        "text": embedding["chunk"],
                        "index_name": self.pinecone_client.index_name,
                        "embedding_id": embedding_id
                    })

                # Store the embeddings in Pinecone
                print(f"\n\tStoring embeddings for file: {file_info['path']}")
                self.pinecone_client.get_and_prep_index()

                # Upsert the function embedding to Pinecone
                upserted_count: int = self.pinecone_client.upsert_vectors(
                    list_dict_upsert)

                # Store the embeddings in Snowflake
                print(f"\tStoring embeddings in Snowflake for file: {
                      file_info['path']}")
                for dict_snowflake in list_dict_snowflake:
                    snowflake_embedding: Dict[str, Any] = {
                        "org_name": self.github_client.repo_owner,
                        "repo_name": embedding["repo"],
                        "file_folder": embedding["folder"],
                        "file_name": embedding["path"],
                        "text": dict_snowflake["text"],
                        "index_name": self.pinecone_client.index_name,
                        "embedding_id": dict_snowflake["embedding_id"]
                    }
                    self.snowflake_client.store_single_embedding(
                        snowflake_embedding)

        print(f"Processing of repository {repo_name} completed.")

    def test_workflow(self):
        """
        Test method to verify the entire workflow using the first repository in REPO_NAMES.
        """
        repo_name = self.get_test_repo()
        self.process_repository(repo_name)


if __name__ == "__main__":
    main_processor = GhvMain()

    # For each repo, chunk every file and generate embeddings, then store them
    for repo_name in main_processor.repo_names:
        main_processor.process_repository(repo_name)

    # Test the entire workflow
    # main_processor.test_workflow()
