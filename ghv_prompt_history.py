import os
import json


class GhvPromptHistory:
    def __init__(self, file_path="history.json"):
        self.file_path = file_path
        self.history = self._load_history()

    def _load_history(self):
        """Load history from the JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return json.load(file)
        return []

    def save_history(self):
        """Save history to the JSON file."""
        with open(self.file_path, "w") as file:
            json.dump(self.history, file, indent=4)

    def add_entry(self, prompt: str, result: str):
        """Add a new entry to the history and save it."""
        self.history.append({"prompt": prompt, "result": result})
        self.save_history()

    def get_history(self):
        """Return the current history."""
        return self.history

    def delete_entry(self, index):
        if 0 <= index < len(self.history):
            del self.history[index]
            self.save_history()
