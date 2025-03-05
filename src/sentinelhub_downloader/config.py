"""Configuration for the Sentinel Hub Downloader."""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import click


class Config:
    """Configuration manager for Sentinel Hub credentials and settings."""

    def __init__(self):
        """Initialize the config manager."""
        self.config_dir = Path.home() / ".sentinelhub-downloader"
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            "client_id": "",
            "client_secret": "",
            "instance_id": "",
            "output_dir": str(Path.cwd() / "downloads"),
        }
        self._config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from the config file."""
        if not self.config_file.exists():
            self.config_dir.mkdir(exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.default_config, f, indent=2)
            return self.default_config.copy()

        with open(self.config_file, "r") as f:
            return json.load(f)

    def save_config(self) -> None:
        """Save configuration to the config file."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set a configuration value."""
        self._config[key] = value
        self.save_config()

    def is_configured(self) -> bool:
        """Check if the client is configured with valid credentials."""
        return all(
            self._config.get(key)
            for key in ["client_id", "client_secret", "instance_id"]
        )

    def configure_wizard(self) -> None:
        """Interactive configuration wizard."""
        click.echo("Configuring Sentinel Hub Downloader credentials")
        click.echo("Get your credentials from https://apps.sentinel-hub.com/dashboard/")

        self._config["client_id"] = click.prompt(
            "Client ID", default=self.get("client_id", "")
        )
        self._config["client_secret"] = click.prompt(
            "Client Secret", default=self.get("client_secret", ""), hide_input=True
        )
        self._config["instance_id"] = click.prompt(
            "Instance ID", default=self.get("instance_id", "")
        )
        self._config["output_dir"] = click.prompt(
            "Default output directory",
            default=self.get("output_dir", str(Path.cwd() / "downloads")),
        )

        self.save_config()
        click.echo("Configuration saved!") 