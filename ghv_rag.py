from typing import List
from ghv_openai import GhvOpenAI
from ghv_pinecone import GhvPinecone
from ghv_snowflake import GhvSnowflake


class GhvRAG:
    """
    Retrieval Augmented Generation (RAG) class that uses OpenAI, Pinecone, and Snowflake to augment prompts.
    The augmented prompt is intended to be coupled with the original prompt and passed to OpenAI's completions API.
    """

    def __init__(self, openai_client: GhvOpenAI, pinecone_client: GhvPinecone, snowflake_client: GhvSnowflake):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.snowflake_client = snowflake_client

    def augment_prompt(self, prompt: str) -> List[str]:
        """
        Augments the given prompt by generating an embedding, querying Pinecone for similar embeddings,
        and retrieving the associated texts from Snowflake.

        Args:
            prompt (str): The user's prompt.

        Returns:
            List[str]: A list of texts retrieved from Snowflake, corresponding to the top 3 similar embeddings.
        """
        # Step 1: Generate an embedding from the prompt using GhvOpenAI
        embedding_response = self.openai_client.generate_embeddings(prompt)
        embedding = embedding_response['embedding']

        # Step 2: Query Pinecone with the generated embedding
        query_results = self.pinecone_client.query_vector(embedding, top_k=3)

        # Step 3: Extract the top 3 scored IDs from the query results
        top_ids = [result['id'] for result in query_results[:3]]

        # Step 4: Query Snowflake to retrieve the "text" column for the top 3 embedding IDs
        retrieved_texts = []
        for embedding_id in top_ids:
            df = self.snowflake_client.read_embedding_by_id(embedding_id)
            if not df.empty:
                retrieved_texts.append(df["text"].iloc[0])

        return retrieved_texts


def testRAG():
    # Instantiate the necessary clients
    openai_client = GhvOpenAI()
    pinecone_client = GhvPinecone()
    snowflake_client = GhvSnowflake()

    # Create the GhvRAG instance
    rag = GhvRAG(openai_client=openai_client,
                 pinecone_client=pinecone_client, snowflake_client=snowflake_client)

    # Prompt the user for input. E.g. write me a python file chunker that opens a python file and breaks up .py code files into logical chunks and returns a list of chunks
    user_prompt = input("Enter your prompt: ")

    # Augment the prompt
    augmented_texts = rag.augment_prompt(user_prompt)

    # Output the augmented texts
    augmented_prompt: str = (
        f"{user_prompt}\n Use this code for context:\n"
        f" 1. {augmented_texts[0]}\n"
        f" 2. {augmented_texts[1]}\n"
        f" 3. {augmented_texts[2]}"
    )
    print("\nAugmented prompt:")
    print(augmented_prompt)
    openai_response: str = openai_client.sendPrompt(augmented_prompt)
    print("\nOpenAI response:")
    print(openai_response)


if __name__ == "__main__":
    testRAG()
