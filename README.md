# ghvector - Vectorize chunks from Github repos

Chunks up code, creates embeddings, stores them in Pinecone, and stores the metadata and original chunks in Snowflake.

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

# Pinecone Settings
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_DIMENSION=512
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=
OPENAI_EMBEDDING_DIMENSIONS=512

```
