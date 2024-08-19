# ghvector_main.py

from ghv_github import GhvGithub

if __name__ == "__main__":
    github_client = GhvGithub()
    code_chunks = github_client.fetch_code_chunks()
    print(code_chunks)
