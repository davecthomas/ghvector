import os
import re
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime


class GhvChunker(ABC):
    MAX_CHUNK_SIZE = 300        # Maximum number of tokens per chunk
    MAX_CHUNKS_PER_FILE = 10
    AVERAGE_WORD_LENGTH = 10  # Assumed average word length in characters

    def __init__(self, file_name: str = None):
        self.file_name = file_name
        self.file_type = self.get_file_type() if file_name else None
        self.content = ""
        self.chunks = []

    @staticmethod
    def get_supported_file_types() -> List[str]:
        return ['.py', '.java', '.sql']

    @abstractmethod
    def chunk_content(self):
        pass

    def get_file_type(self):
        if self.file_name:
            _, file_extension = os.path.splitext(self.file_name)
            return file_extension
        return None

    def set_content(self, content: str = None, file_name: str = None):
        self.content = content
        if file_name:
            self.file_name = file_name
            # Set the file type based on the new file name
            self.file_type = self.get_file_type()

    def _add_chunk(self, chunk: str) -> int:
        if len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self.chunks.append(chunk)
            print(f"\r\tAdded chunk {len(self.chunks)}", end="")
        else:
            print(f"\n\tMax chunks of {
                  self.MAX_CHUNKS_PER_FILE} per file limit reached.")
        return len(self.chunks)

    def get_chunks(self):
        if not self.chunks:
            self.chunk_content()
        return self.chunks

    def estimate_file_tokens(self):
        """Estimates the total number of tokens in the file content."""
        return sum(self._estimate_token_count(line) for line in self.content.splitlines())

    def _estimate_token_count(self, line: str) -> int:
        """Estimates the number of tokens in a line of text."""
        return len(re.findall(r'\w+', line))

    def read_limited_file(self, file_name):
        """
        Test only use. 
        Reads a file up to the maximum estimated character limit.
        This is for local file reads only and is not for Github API content retrieval.
        """
        max_characters_to_read = self.AVERAGE_WORD_LENGTH * \
            self.MAX_CHUNKS_PER_FILE * self.MAX_CHUNK_SIZE
        with open(file_name, 'r') as file:
            content = file.read(max_characters_to_read)
        return content

    @staticmethod
    def create_chunker(file_name: str) -> 'GhvChunker':
        """
        Factory method to create a chunker based on the file extension.
        """
        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension == '.py':
            return PythonChunker(file_name)
        elif file_extension == '.java':
            return JavaChunker(file_name)
        elif file_extension == '.sql':
            return SQLChunker(file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


class PythonChunker(GhvChunker):
    def chunk_content(self) -> List[str]:
        lines = self.content.splitlines()
        current_chunk = []
        current_size = 0
        in_multiline_string = False
        multiline_string_delimiter = None
        last_line_was_class = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            # Handle multiline strings
            if in_multiline_string:
                current_chunk.append(line)
                current_size += self._estimate_token_count(line)
                if stripped_line.endswith(multiline_string_delimiter):
                    in_multiline_string = False
                i += 1
                continue

            if stripped_line.startswith(('"""', "'''")):
                if stripped_line.endswith(('"""', "'''")) and len(stripped_line) > 3:
                    token_count = self._estimate_token_count(line)
                else:
                    in_multiline_string = True
                    multiline_string_delimiter = stripped_line[:3]
                    token_count = self._estimate_token_count(line)
            else:
                token_count = self._estimate_token_count(line)

            # Handle decorator lines (e.g., @abstractmethod)
            if stripped_line.startswith('@'):
                if current_chunk:
                    # Finalize the current chunk before starting a new chunk for the decorator
                    chunk_number = self._add_chunk("\n".join(current_chunk))
                    if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                        break  # Exit if the maximum number of chunks is reached
                    current_chunk = []
                    current_size = 0

                # Start a new chunk with the decorator
                current_chunk.append(line)
                current_size += token_count

                # Ensure that the next line (function/method declaration) is included in the same chunk
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('def '):
                        current_chunk.append(lines[i + 1])
                        current_size += self._estimate_token_count(next_line)
                        i += 1
                        # Look ahead for the rest of the function/method block
                        while i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line.startswith(('def ', 'class ', '@')):
                                break  # Stop if the next line starts a new decorator, function, or class
                            current_chunk.append(lines[i + 1])
                            current_size += self._estimate_token_count(
                                next_line)
                            i += 1

                continue  # Move to the next line after handling the decorator

            # Detect block starts (e.g., def, class, if, etc.)
            block_start = False
            if stripped_line.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try ', 'except ', 'with ', 'elif ', 'else:')):
                block_start = True
                if stripped_line.startswith('class '):
                    last_line_was_class = True
                elif stripped_line.startswith('def ') and last_line_was_class:
                    last_line_was_class = False  # Don't start a new chunk if 'def' follows 'class'
                else:
                    last_line_was_class = False  # Reset if this is a regular block start
                if stripped_line.startswith(('if ', 'else', 'for ', 'while ')) and current_chunk:
                    current_chunk.append(line)
                    current_size += token_count
                    i += 1
                    continue  # Skip adding the line again below

                # Finalize the chunk before starting a new function or class
                if current_chunk:
                    chunk_number = self._add_chunk("\n".join(current_chunk))
                    if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                        break  # Exit if the maximum number of chunks is reached
                    current_chunk = []
                    current_size = 0

            # Add the current line to the chunk
            current_chunk.append(line)
            current_size += token_count

            # Ensure sequential lines are grouped together
            if not block_start and current_size < self.MAX_CHUNK_SIZE:
                lookahead = i + 1
                while lookahead < len(lines):
                    lookahead_line = lines[lookahead].strip()

                    # Ensure the loop breaks if a decorator or block start is found
                    if lookahead_line.startswith(('@', 'def ', 'class ')):
                        break  # Stop if the next line starts a new decorator, function, or class

                    current_chunk.append(lines[lookahead])
                    current_size += self._estimate_token_count(lookahead_line)
                    lookahead += 1

                i = lookahead - 1  # Adjust i to the last line processed

            # Finalize chunk if it exceeds the maximum size
            if current_size + token_count > self.MAX_CHUNK_SIZE:
                chunk_number = self._add_chunk("\n".join(current_chunk))
                if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                    break  # Exit if the maximum number of chunks is reached
                current_chunk = []
                current_size = 0

            i += 1

        if current_chunk and len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self._add_chunk("\n".join(current_chunk))

        return self.chunks

    def _add_chunk(self, chunk: str) -> int:
        if len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self.chunks.append(chunk)
            print(f"\r\tAdded {len(self.chunks)} chunks.", end="")
        else:
            print(f"\n\tMax chunks of {
                  self.MAX_CHUNKS_PER_FILE} per file limit reached.")
        return len(self.chunks)


class JavaChunker(GhvChunker):
    """
    """

    def chunk_content(self) -> List[str]:
        lines = self.content.splitlines()
        current_chunk = []
        current_size = 0
        indent_stack = []
        current_indent = 0
        in_multiline_comment = False
        multiline_comment_delimiter = None

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            # Handle multiline comments
            if in_multiline_comment:
                current_chunk.append(line)
                current_size += self._estimate_token_count(line)
                if stripped_line.endswith(multiline_comment_delimiter):
                    in_multiline_comment = False
                i += 1
                continue

            if stripped_line.startswith(('/*')):
                if stripped_line.endswith('*/') and len(stripped_line) > 2:
                    token_count = self._estimate_token_count(line)
                else:
                    in_multiline_comment = True
                    multiline_comment_delimiter = '*/'
                    token_count = self._estimate_token_count(line)
            else:
                token_count = self._estimate_token_count(line)

            # Update indent stack
            if stripped_line:
                indent_level = len(line) - len(stripped_line)
                if indent_level > current_indent:
                    indent_stack.append(current_indent)
                elif indent_level < current_indent:
                    while indent_stack and indent_stack[-1] >= indent_level:
                        indent_stack.pop()
                current_indent = indent_level

            # Handle annotations (e.g., @Override)
            if stripped_line.startswith('@') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(('public ', 'private ', 'protected ', 'void ', 'class ')):
                    current_chunk.append(line)
                    current_size += token_count
                    i += 1
                    # Continue processing the method or class definition line
                    line = lines[i]
                    stripped_line = line.strip()
                    token_count = self._estimate_token_count(line)

            # Detect block starts (e.g., methods, classes, if, etc.)
            block_start = False
            if stripped_line.startswith(('public ', 'private ', 'protected ', 'void ', 'class ', 'if ', 'for ', 'while ', 'try ', 'catch ', 'else ')):
                block_start = True
                if stripped_line.startswith(('if ', 'else', 'for ', 'while ')) and current_chunk:
                    current_chunk.append(line)
                    current_size += token_count
                    i += 1
                    continue  # Skip adding the line again below

                # Finalize the chunk before starting a new method or class
                if current_chunk:
                    chunk_number = self._add_chunk("\n".join(current_chunk))
                    if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                        break  # Exit if the maximum number of chunks is reached
                    current_chunk = []
                    current_size = 0

            # Add the current line to the chunk
            if not block_start or stripped_line.startswith(('public ', 'private ', 'protected ', 'void ', 'class ')):
                current_chunk.append(line)
                current_size += token_count

            # Lookahead to detect the start of the next method or class
            if stripped_line.startswith(('public ', 'private ', 'protected ', 'void ', 'class ')):
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith(('public ', 'private ', 'protected ', 'void ', 'class ')):
                        chunk_number = self._add_chunk(
                            "\n".join(current_chunk))
                        if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                            break  # Exit if the maximum number of chunks is reached
                        current_chunk = []
                        current_size = 0

            # Finalize chunk if it exceeds the maximum size
            if current_size + token_count > self.MAX_CHUNK_SIZE and not indent_stack:
                chunk_number = self._add_chunk("\n".join(current_chunk))
                if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                    break  # Exit if the maximum number of chunks is reached
                current_chunk = []
                current_size = 0

            i += 1

        if current_chunk and len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self._add_chunk("\n".join(current_chunk))

        return self.chunks

    def _add_chunk(self, chunk: str) -> int:
        if len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self.chunks.append(chunk)
            print(f"\r\tAdded chunk {len(self.chunks)}", end="")
        else:
            print(f"\n\tMax chunks of {
                  self.MAX_CHUNKS_PER_FILE} per file limit reached.")
        return len(self.chunks)


class SQLChunker(GhvChunker):
    def chunk_content(self) -> List[str]:
        lines = self.content.splitlines()
        current_chunk = []
        current_size = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            token_count = self._estimate_token_count(line)

            # Detect block starts for SQL keywords like TABLE, PROCEDURE, TRIGGER, VIEW, etc.
            if re.match(r'^\b(CREATE|ALTER|DROP)\b\s+\b(TABLE|PROCEDURE|TRIGGER|VIEW|FUNCTION)\b', stripped_line, re.IGNORECASE):
                if current_chunk:
                    # Finalize the current chunk before starting a new SQL block
                    chunk_number = self._add_chunk("\n".join(current_chunk))
                    if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                        break  # Exit if the maximum number of chunks is reached
                    current_chunk = []
                    current_size = 0

            # Add the current line to the chunk
            current_chunk.append(line)
            current_size += token_count

            # Finalize chunk if it exceeds the maximum size
            if current_size > self.MAX_CHUNK_SIZE:
                chunk_number = self._add_chunk("\n".join(current_chunk))
                if chunk_number >= self.MAX_CHUNKS_PER_FILE:
                    break  # Exit if the maximum number of chunks is reached
                current_chunk = []
                current_size = 0

            i += 1

        if current_chunk and len(self.chunks) < self.MAX_CHUNKS_PER_FILE:
            self._add_chunk("\n".join(current_chunk))

        return self.chunks


if __name__ == "__main__":
    # Local test mode: Collect chunks for all Python files in the current directory
    data = []
    for file_name in os.listdir("."):
        if file_name.endswith(".py"):
            print(f"\nProcessing file: {file_name}")
            chunker = GhvChunker.create_chunker(file_name)
            total_tokens = chunker.estimate_file_tokens()
            print(f"\n\tEstimated total tokens: {total_tokens}")
            chunker.chunk_content()
            chunks = chunker.get_chunks()

            for idx, chunk in enumerate(chunks):
                data.append({
                    "filename": file_name,
                    "file type": chunker.get_file_type(),
                    "chunk number": idx + 1,
                    "chunk text": chunk,
                    "word_count": len(chunk.split())
                })

    # Convert the data to a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"chunked_python_files_{timestamp}.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n\nChunks detail saved to {output_filename}")
