-- Description: SQL schema for the GitHub to Vector Code Chunks table
-- Assumes Snowflake as the target database
-- Note: Snowflake does not support AUTOINCREMENT, so we use the IDENTITY() function instead
CREATE OR REPLACE TABLE "github_to_vector_code_chunks" (
    "id" INT IDENTITY(1,1) PRIMARY KEY,             -- Auto-incrementing primary key using IDENTITY
    "org_name" VARCHAR(255) NOT NULL,               -- Name of the GitHub organization
    "repo_name" VARCHAR(255) NOT NULL,              -- Name of the repository within the organization
    "file_name" VARCHAR(255) NOT NULL,              -- File name where the code chunk resides
    "line_start" INT NOT NULL,                      -- Starting line number of the code chunk
    "line_end" INT NOT NULL,                        -- Ending line number of the code chunk
    "code_chunk" STRING NOT NULL,                   -- The actual source code chunk
    "embedding_id" STRING NOT NULL UNIQUE,          -- String-based identifier for the embedding in Pinecone
    "storage_datetime" TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() -- Timestamp for when the record was stored
);

