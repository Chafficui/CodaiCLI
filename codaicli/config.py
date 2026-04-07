"""Configuration management for CodaiCLI."""

import json
from pathlib import Path


class Config:
    """Manages configuration and API keys."""
    
    def __init__(self):
        """Initialize configuration."""
        self.config_dir = Path.home() / ".codaicli"
        self.config_file = self.config_dir / "config.json"
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
        
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def save(self):
        """Save configuration to file."""
        # Ensure the directory exists
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
            
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value

    def set_api_key(self, provider: str, key: str):
        """Set API key for a provider (legacy compat)."""
        self.set(f"{provider}_api_key", key)