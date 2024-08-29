from typing import List
from ghv_openai import GhvOpenAI
from ghv_pinecone import GhvPinecone
from ghv_prompt_history import GhvPromptHistory
from ghv_snowflake import GhvSnowflake
from fastapi import Body, FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List
import uvicorn
import threading

# Assuming GhvRAG and other related classes are already defined/imported


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

# Define the testWeb function


def testWeb():
    app = FastAPI()

    # Set up Jinja2 for HTML templating
    templates = Jinja2Templates(directory="templates")

    # Initialize the GhvPromptHistory class
    prompt_history = GhvPromptHistory()

    # Initialize necessary clients and GhvRAG instance
    openai_client = GhvOpenAI()
    pinecone_client = GhvPinecone()
    snowflake_client = GhvSnowflake()
    rag = GhvRAG(openai_client=openai_client,
                 pinecone_client=pinecone_client, snowflake_client=snowflake_client)

    @app.get("/", response_class=HTMLResponse)
    async def read_form(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "history": prompt_history.get_history(), "current_result": None})

    @app.post("/", response_class=HTMLResponse)
    async def handle_prompt(request: Request, user_prompt: str = Body(...)):
        # Augment the prompt using GhvRAG
        print(f"prompt: {user_prompt}")
        try:
            # Extract the prompt
            print(f"Received prompt: {user_prompt}")

            # Proceed with augmenting the prompt
            augmented_texts = rag.augment_prompt(user_prompt)
            augmented_prompt = (
                f"{user_prompt}\n Use this code for context:\n 1. {
                    augmented_texts[0]}\n 2. {augmented_texts[1]}\n 3. {augmented_texts[2]}"
            )
            openai_response = openai_client.sendPrompt(augmented_prompt)
            prompt_history.add_entry(user_prompt, openai_response)
            print(f"Response: {openai_response}")

            # Return JSON response instead of HTML for better debugging
            return JSONResponse(content={"current_result": openai_response})

        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="An error occurred during processing")
        # augmented_texts = rag.augment_prompt(user_prompt)
        # augmented_prompt = f"{user_prompt}\n Use this code for context:\n 1. {
        #     augmented_texts[0]}\n 2. {augmented_texts[1]}\n 3. {augmented_texts[2]}"
        # openai_response = openai_client.sendPrompt(augmented_prompt)
        # # Add the entry to the prompt history
        # prompt_history.add_entry(user_prompt, openai_response)
        # print(f"response: {openai_response}")

        # # Return the updated HTML with the new conversation added
        # return templates.TemplateResponse("index.html", {
        #     "request": request,
        #     "history": prompt_history.get_history(),
        #     "current_result": openai_response
        # })

    @app.delete("/delete-history/{index}")
    async def delete_history(index: int):
        try:
            prompt_history.delete_entry(index)
            return {"message": "History item deleted successfully."}
        except IndexError:
            raise HTTPException(
                status_code=404, detail="History item not found")

    @app.post("/rename-history/{index}")
    async def rename_history_entry(index: int, request: Request):
        data = await request.json()
        new_name = data.get("new_name", "").strip()
        if new_name:
            prompt_history.rename_entry(index, new_name)
            return {"status": "success"}
        return {"status": "error", "message": "Invalid name"}

    # Run the app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


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


# Main function to toggle between modes
if __name__ == "__main__":
    # mode = input(
    #     "Enter 'web' for web mode or 'cli' for command-line mode: ").strip().lower()

    # if mode == 'web':
    testWeb()
    # else:
    #     testRAG()
