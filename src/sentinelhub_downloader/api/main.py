"""Main Sentinel Hub API client class that combines all functionality."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

from sentinelhub_downloader.config import Config
from sentinelhub_downloader.api.client import SentinelHubClient
from sentinelhub_downloader.api.catalog import CatalogAPI
from sentinelhub_downloader.api.downloader import DownloaderAPI
from sentinelhub_downloader.api.process import ProcessAPI
from sentinelhub_downloader.api.metadata import MetadataAPI
from sentinelhub_downloader.api.byoc import BYOCAPI

logger = logging.getLogger("sentinelhub_downloader")

class SentinelHubAPI:
    """Main API client that provides access to all Sentinel Hub functionalities."""
    
    def __init__(self, config: Optional[Config] = None, debug: bool = False):
        """Initialize the Sentinel Hub API.
        
        Args:
            config: Configuration object containing credentials
            debug: Enable debug logging
        """
        # Use provided config or create a new one
        self.config = config or Config()
        self.debug = debug
        
        # Initialize client
        self.client = SentinelHubClient(self.config, debug=debug)
        
        # Initialize API components
        self.process_api = ProcessAPI(self.client)
        self.catalog_api = CatalogAPI(self.client)
        self.metadata_api = MetadataAPI(self.client)
        
        # Initialize higher-level APIs that depend on the components
        self.downloader_api = DownloaderAPI(self.client, self.process_api)
        self.byoc_api = BYOCAPI(self.client, self.process_api, self.catalog_api, self.metadata_api)
    
    # Proxy methods to underlying APIs for backward compatibility
    
    def search_images(self, *args, **kwargs):
        """Search for available images."""
        return self.catalog_api.search_images(*args, **kwargs)
    
    def get_available_dates(self, *args, **kwargs):
        """Get available dates for a collection."""
        return self.catalog_api.get_available_dates(*args, **kwargs)
    
    def download_image(self, *args, **kwargs):
        """Download a single image."""
        return self.downloader_api.download_image(*args, **kwargs)
    
    def download_timeseries(self, *args, **kwargs):
        """Download a time series of images."""
        return self.catalog_api.download_timeseries(*args, **kwargs)
    
    def download_byoc_timeseries(self, *args, **kwargs):
        """Download a time series of images from a BYOC collection."""
        return self.byoc_api.download_byoc_timeseries(*args, **kwargs)
    
    def get_collection_info(self, *args, **kwargs):
        """Get information about a collection."""
        return self.metadata_api.get_collection_info(*args, **kwargs)
    
    def get_stac_info(self, *args, **kwargs):
        """Get STAC collection information."""
        return self.metadata_api.get_stac_info(*args, **kwargs)
    
    def get_byoc_info(self, *args, **kwargs):
        """Get BYOC collection information."""
        return self.metadata_api.get_byoc_info(*args, **kwargs)
    
    def extract_band_info(self, *args, **kwargs):
        """Extract band information from collection metadata."""
        return self.metadata_api.extract_band_info(*args, **kwargs)
    
    def get_collection_data_type(self, *args, **kwargs):
        """Extract default data type from collection metadata."""
        return self.metadata_api.get_collection_data_type(*args, **kwargs)
    
    def get_collection_band_names(self, *args, **kwargs):
        """Extract band names from collection metadata."""
        return self.metadata_api.get_collection_band_names(*args, **kwargs)
    
    def get_collection_nodata_value(self, *args, **kwargs):
        """Extract common nodata value from collection metadata."""
        return self.metadata_api.get_collection_nodata_value(*args, **kwargs)
    
    def create_dynamic_evalscript(self, *args, **kwargs):
        """Create a dynamic evalscript to extract specified bands.
        
        Forwards to ProcessAPI.create_dynamic_evalscript
        """
        return self.process_api.create_dynamic_evalscript(*args, **kwargs) 