"""A command-line tool to download GeoTIFF data from Sentinel Hub."""

__version__ = "0.1.0"

from sentinelhub_downloader.api import SentinelHubAPI
from sentinelhub_downloader.config import Config

__all__ = ["SentinelHubAPI", "Config"] 