from __future__ import annotations
from typing import Dict, Any, List
import os
import json
import time
import requests
import fnmatch
import base64
from dotenv import load_dotenv
from requests.models import Response
from requests.exceptions import Timeout, RequestException, HTTPError, ConnectionError
from urllib3.exceptions import ProtocolError

# Globals
GITHUB_API_BASE_URL = "https://api.github.com"
API_TOKEN = os.getenv("GITHUB_API_TOKEN")
MAX_ITEMS_PER_PAGE = 100  # Limited by Github API


class GhvGithub:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.repo_owner = os.getenv("REPO_OWNER", None)
        repo_names_env = os.getenv("REPO_NAMES")
        self.repo_names = repo_names_env.split(",") if repo_names_env else []
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {API_TOKEN}"
        }
        self.git_include = os.getenv("GIT_INCLUDE", "").split(",")

    def check_API_rate_limit(self, response: Response) -> bool:
        """
        Check if we overran our rate limit. Take a short nap if so.
        Return True if we overran.
        """
        if response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers:
            if int(response.headers['X-Ratelimit-Remaining']) == 0:
                print(f"\t403 forbidden response header shows X-Ratelimit-Remaining at {
                      response.headers['X-Ratelimit-Remaining']} requests.")
                self.sleep_until_ratelimit_reset_time(
                    int(response.headers['X-RateLimit-Reset']))
        return (response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers)

    def sleep_until_ratelimit_reset_time(self, reset_timestamp: int) -> None:
        """
        Sleep until the GitHub API rate limit reset time.
        """
        current_time = int(time.time())
        sleep_time = max(reset_timestamp - current_time, 0)
        print(f"Sleeping for {sleep_time} seconds due to rate limiting...")
        time.sleep(sleep_time)

    def github_request_exponential_backoff(self, url: str, params: Dict[str, Any] = {}) -> List[Dict]:
        """
        Returns a list of pages (or just one page) where each page is the full json response
        object. The caller must know to process these pages as the outer list of the result. 
        Retry backoff in 422, 202, or 403 (rate limit exceeded) responses
        """
        exponential_backoff_retry_delays_list: list[int] = [
            1, 2, 4, 8, 16, 32, 64]
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {API_TOKEN}"
        }

        retry: bool = False
        retry_count: int = 0
        response: Response = Response()
        retry_url: str = None
        pages_list: List[Dict] = []
        page = 1
        if "per_page" not in params:
            params["per_page"] = MAX_ITEMS_PER_PAGE

        while True:
            params["page"] = page

            try:
                response = requests.get(url, headers=headers, params=params)
            except Timeout:
                print(
                    f"Request to {url} with params {params} timed out on attempt {retry_count + 1}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds.")
                retry = True
                retry_count += 1
                continue
            except ProtocolError as e:
                print(
                    f"Protocol error on attempt {retry_count + 1}: {e}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds.")
                retry = True
                retry_count += 1
                continue
            except ConnectionError as ce:
                print(
                    f"Connection error on attempt {retry_count + 1}: {ce}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds.")
                retry = True
                retry_count += 1
                continue
            except HTTPError as he:
                print(
                    f"HTTP error on attempt {retry_count + 1}: {he}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds.")
                retry = True
                retry_count += 1
                continue
            except RequestException as e:
                print(
                    f"Request exception on attempt {retry_count + 1}: {e}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds.")
                retry = True
                retry_count += 1
                continue

            if retry or (response is not None and response.status_code != 200):
                if response.status_code == 422 and response.reason == "Unprocessable Entity":
                    dict_error: Dict[str, any] = json.loads(response.text)
                    print(
                        f"Skipping: {response.status_code} {response.reason} for url {url}\n\t{dict_error['message']}\n\t{dict_error['errors'][0]['message']}")

                elif retry or response.status_code == 202 or response.status_code == 403:  # Try again
                    for retry_attempt_delay in exponential_backoff_retry_delays_list:
                        if 'Location' in response.headers:
                            retry_url = response.headers.get('Location')
                        # The only time we override the exponential backoff if we are asked by Github to wait
                        if 'Retry-After' in response.headers:
                            retry_attempt_delay = response.headers.get(
                                'Retry-After')
                        # Wait for n seconds before checking the status
                        time.sleep(retry_attempt_delay)
                        retry_response_url: str = retry_url if retry_url else url
                        print(
                            f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
                        # A 403 may require us to take a nap
                        self.check_API_rate_limit(response)

                        try:
                            response = requests.get(
                                retry_response_url, headers=headers)
                        except Timeout:
                            print(
                                f"Request to {url} with params {params} timed out on attempt {retry_count + 1}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue
                        except ProtocolError as e:
                            print(
                                f"Protocol error on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue
                        except ConnectionError as ce:
                            print(
                                f"Connection error on attempt {retry_count + 1}: {ce}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue
                        except HTTPError as he:
                            print(
                                f"HTTP error on attempt {retry_count + 1}: {he}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue
                        except RequestException as e:
                            print(
                                f"Request exception on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue
                        except Exception as e:
                            print(
                                f"Unexpected exception on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds.")
                            retry = True
                            retry_count += 1
                            continue

                        # Check if the retry response is 200
                        if response.status_code == 200:
                            break  # Exit the loop on successful response
                        else:
                            print(
                                f"\tRetried request and still got bad response status code: {response.status_code}")

            if response.status_code == 200:
                page_json = response.json()
                if not page_json or (isinstance(page_json, list) and not page_json):
                    break  # Exit if the page is empty
                pages_list.append(response.json())
            else:
                self.check_API_rate_limit(response)
                print(
                    f"Retries exhausted. Giving up. Status code: {response.status_code}")
                break

            if 'next' not in response.links:
                break  # Check for a 'next' link to determine if we should continue
            else:
                url = response.links.get("next", "").get("url", "")

            page += 1

        return pages_list

    def list_files_in_repo(self, repo_name: str) -> List[Dict[str, str]]:
        """
        Recursively lists files in the specified repository, filtered by the GIT_INCLUDE patterns
        and excluding directories specified in GITHUB_EXCLUDE_SUBDIRS.
        Returns a list of dictionaries containing the repo, folder (fully qualified path), path, and file name.
        """
        # Load exclude subdirectories from environment variable
        exclude_subdirs = os.getenv("GITHUB_EXCLUDE_SUBDIRS", "").split(",")

        def should_exclude(path: str) -> bool:
            """Helper function to check if the current path should be excluded."""
            for exclude in exclude_subdirs:
                if path.strip('/').split('/')[0] == exclude.strip('/'):
                    return True
            return False

        def get_files_in_directory(url: str, folder: str = "") -> List[Dict[str, str]]:
            file_details_list = []
            response_pages = self.github_request_exponential_backoff(url)

            if response_pages:
                for page in response_pages:
                    for file in page:
                        file_path = file.get('path', '')
                        file_type = file.get('type', '')

                        # Skip excluded directories
                        if should_exclude(file_path):
                            continue

                        if file_type == 'dir':  # If the item is a directory, recurse into it
                            subdir_url = f"{
                                GITHUB_API_BASE_URL}/repos/{self.repo_owner}/{repo_name}/contents/{file_path}"
                            subfolder = os.path.join(folder, os.path.basename(
                                file_path))  # Build the full folder path
                            file_details_list.extend(
                                get_files_in_directory(subdir_url, folder=subfolder))
                        elif file_type == 'file' and any(fnmatch.fnmatch(file_path, pattern) for pattern in self.git_include):
                            file_details_list.append({
                                "repo": repo_name,
                                "folder": folder,  # Fully qualified path under the repo
                                "path": file_path,
                                "file_name": os.path.basename(file_path)
                            })

            return file_details_list

        # Start from the root directory
        root_url = f"{
            GITHUB_API_BASE_URL}/repos/{self.repo_owner}/{repo_name}/contents"
        file_list = get_files_in_directory(root_url)

        if file_list:
            return file_list

        print(f"Failed to list files in repository {repo_name}")
        return file_list  # Return empty list if no files are found

    def get_file_chunks(self, repo_name: str, file_path: str, chunk_size: int = 512) -> List[str]:
        """
        Retrieves and chunks the content of a file from the GitHub repository.
        """
        url = f"{
            GITHUB_API_BASE_URL}/repos/{self.repo_owner}/{repo_name}/contents/{file_path}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            file_content = response.json().get("content", "")
            # Decode the content if it’s base64 encoded (as GitHub API does)
            content_decoded = base64.b64decode(file_content).decode('utf-8')
            # Chunk the content
            chunks = [content_decoded[i:i + chunk_size]
                      for i in range(0, len(content_decoded), chunk_size)]
            return chunks
        else:
            print(f"Failed to fetch content for {file_path} in {
                  repo_name}. Status code: {response.status_code}")
            return []

    def test_list_and_chunk_files(self):
        """
        Lists the files in the first repository and outputs the first 10 chunks of the first 3 files.
        """
        if not self.repo_names:
            print("No repositories specified.")
            return

        first_repo = self.repo_names[0]
        print(f"Listing files in the repository: {first_repo}")

        files = self.list_files_in_repo(first_repo)
        print(f"Found {len(files)} files matching the patterns in {
              first_repo}:")
        print(files[:10])  # List the first 10 files for demonstration

        # Output the first 10 chunks of the first 3 files
        for i, file_path in enumerate(files[:3]):
            print(f"\nFetching chunks from file {i+1}: {file_path}")
            chunks = self.get_file_chunks(first_repo, file_path)
            for chunk in chunks[:10]:  # Limit to the first 10 chunks
                print(chunk)
                print("--- End of chunk ---")


if __name__ == "__main__":
    github_client = GhvGithub()
    github_client.test_list_and_chunk_files()
