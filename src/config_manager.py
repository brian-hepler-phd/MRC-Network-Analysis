import yaml
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.yaml'

class ConfigManager:
    """
    A simple manager to read from and write to the project's config.yaml file.
    """
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        self.config = self._load_config()

    def _load_config(self):
        """Loads the YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_config(self):
        """Saves the current configuration back to the file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False, indent=2)

    def get_path(self, key: str, section: str = 'pipeline_outputs') -> str:
        """
        Retrieves a path from the configuration file.
        
        Args:
            key: The key for the path (e.g., 'disambiguated_authors_path').
            section: The top-level section ('static_inputs' or 'pipeline_outputs').
            
        Returns:
            The file path as a string.
        """
        path = self.config.get(section, {}).get(key)
        if path is None:
            raise KeyError(f"Path for key '{key}' in section '{section}' not found in config.")
        return path

    def update_path(self, key: str, new_path: str, section: str = 'pipeline_outputs'):
        """
        Updates a path in the configuration file and saves it.
        
        Args:
            key: The key for the path to update.
            new_path: The new file path string to save.
            section: The top-level section to update.
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = new_path
        self._save_config()
        print(f"Updated config: '{key}' set to '{new_path}'")