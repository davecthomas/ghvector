# ghvector - Vectorize chunks from Github repos

Chunks up code, creates embeddings, stores them in Pinecone, and stores the metadata and original chunks in Snowflake.

# Generating Embeddings from Github

```mermaid
graph TD
    A[GitHub Repository] -->|1. Fetch Files| B[GhsGithub Class]
    B -->|2. List Files| C{Filter Files}
    C -->|3. Apply GIT_INCLUDE<br>and GITHUB_EXCLUDE_SUBDIRS| D[Filtered Files]
    D -->|4. Chunk Files| E[File Chunks]
    E -->|5. Process Chunks<br>GhvOpenAI Class| F[Generate Embeddings]
    F -->|6. Embedding Vectors| G{Store Data}
    G -->|7. Pinecone| H[Pinecone Vector DB]
    G -->|8. Snowflake| I[Snowflake DB]

    %% Subflow explanations
    B --> J{Test Mode?}
    J -->|Yes| K[Limit Chunks to Test Limit]
    J -->|No| L[Process All Chunks]
    K --> F
    L --> F

    G --> M[Add Metadata]

```

## Retrieving Embeddings and Augmenting a Generated Prompt

```mermaid
graph TD
    A[User Prompt] -->|1. Query Pinecone| B[Pinecone Similarity Search]
    B -->|2. Found Vectors?| C{Vectors Found?}
    C -->|Yes| D[Query Snowflake for File Chunks]
    C -->|No| E[Handle No Results]
    D -->|3. Retrieve File Chunks| F[Augment Prompt with Chunks]
    F -->|4. Send to OpenAI| G[OpenAI Completion API]
```

## Embedding models

Which embedding model you use has a lot to do with the type of content you are vectorizing. Price should be considered also.
The dimensions value must match your Pinecone settings also.

[Learn about embeddings](#)

| Model                  | Dimensions | Pricing            | Pricing with Batch API |
| ---------------------- | ---------- | ------------------ | ---------------------- |
| text-embedding-3-small | 1,536      | $0.020 / 1M tokens | $0.010 / 1M tokens     |
| text-embedding-3-large | 3,072      | $0.130 / 1M tokens | $0.065 / 1M tokens     |
| ada v2                 | 1,536      | $0.100 / 1M tokens | $0.050 / 1M tokens     |

# Install

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 ghdeps.py
```

# Settings - the .env file should have

```
# GitHub Settings
GITHUB_API_TOKEN=
REPO_OWNER=
REPO_NAMES=
GIT_INCLUDE=*.py,*.java,*.js,*.ts,*.html,*.css,*.json,*.yml,*.yaml,*.md,*.txt,*.csv,*.tsv,*.xml,*.sql,*.sh,*.bat
GITHUB_EXCLUDE_SUBDIRS=

# Snowflake Settings
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DB=
SNOWFLAKE_SCHEMA=

# Pinecone and OpenAI Shared Settings (this needs to match your selected model)
EMBEDDING_DIMENSIONS=1536

# Pinecone Settings
PINECONE_API_KEY=
PINECONE_DIMENSION=512
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_PROJECT_NAME=
# The full index name is built from the base, the embedding model, and dimensions
PINECONE_BASE_INDEX_NAME=
# true in test mode will cause all indexes to be deleted at GhvPicecone init
PINECONE_TEST_MODE=

OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=
OPENAI_EMBEDDING_DIMENSIONS=512

```
