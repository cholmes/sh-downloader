"""Sentinel Hub API client package."""

from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from sentinelhub_downloader.api.client import SentinelHubClient
from sentinelhub_downloader.api.catalog import CatalogAPI
from sentinelhub_downloader.api.downloader import DownloaderAPI
from sentinelhub_downloader.api.process import ProcessAPI
from sentinelhub_downloader.api.metadata import MetadataAPI
from sentinelhub_downloader.api.byoc import BYOCAPI

# Main class that combines all functionality
from sentinelhub_downloader.api.main import SentinelHubAPI

__all__ = [
    "SentinelHubAPI",
    "SentinelHubClient",
    "CatalogAPI",
    "DownloaderAPI",
    "ProcessAPI",
    "MetadataAPI",
    "BYOCAPI",
]

class SentinelHubAPI:
    """Main API class for Sentinel Hub Downloader."""
    
    def __init__(self, config, debug=False):
        """Initialize the API client."""
        self.config = config
        self.debug = debug
        self.client = SentinelHubClient(config)
        
        # Initialize APIs in the correct order to handle dependencies
        self.catalog = CatalogAPI(self.client)
        self.process = ProcessAPI(self.client)
        self.metadata = MetadataAPI(self.client)
        
        # BYOC API needs references to other APIs
        self.byoc = BYOCAPI(
            self.client,
            process_api=self.process,
            catalog_api=self.catalog,
            metadata_api=self.metadata
        )
    
    # ... other methods ...

    def get_stac_feature(self, collection_id: str, feature_id: str) -> Dict[str, Any]:
        """Get STAC feature information.
        
        Args:
            collection_id: Collection ID (e.g., sentinel-2-l2a or byoc-uuid)
            feature_id: Feature ID to retrieve
            
        Returns:
            Dictionary containing the feature information
        """
        return self.catalog.get_stac_feature(collection_id, feature_id)

    def get_stac_info(self, collection_id: str) -> Dict[str, Any]:
        """Get STAC collection information.
        
        Args:
            collection_id: Collection ID (e.g., sentinel-2-l2a or byoc-uuid)
            
        Returns:
            Dictionary containing the collection information
        """
        return self.metadata.get_stac_info(collection_id)

    def download_byoc_timeseries(
        self,
        byoc_id: str,
        bbox: Tuple[float, float, float, float],
        time_interval: Tuple[datetime, datetime],
        output_dir: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        time_difference_days: Optional[int] = 1,
        filename_template: Optional[str] = None,
        evalscript: Optional[str] = None,
        auto_discover_bands: bool = True,
        specified_bands: Optional[List[str]] = None,
        nodata_value: Optional[float] = None,
        scale_metadata: Optional[float] = None,
        data_type: str = "AUTO",
    ) -> List[str]:
        """Download a time series of images from a BYOC collection.
        
        Delegates to BYOCAPI.download_byoc_timeseries.
        """
        return self.byoc.download_byoc_timeseries(
            byoc_id=byoc_id,
            bbox=bbox,
            time_interval=time_interval,
            output_dir=output_dir,
            size=size,
            time_difference_days=time_difference_days,
            filename_template=filename_template,
            evalscript=evalscript,
            auto_discover_bands=auto_discover_bands,
            specified_bands=specified_bands,
            nodata_value=nodata_value,
            scale_metadata=scale_metadata,
            data_type=data_type
        )

    def create_dynamic_evalscript(self, bands: List[str], data_type: str = "AUTO") -> str:
        """Create a dynamic evalscript for the specified bands.
        
        Delegates to ProcessAPI.create_dynamic_evalscript.
        """
        return self.process.create_dynamic_evalscript(bands, data_type)

    def extract_band_info(self, stac_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract band information from STAC collection info.
        
        Delegates to MetadataAPI.extract_band_info.
        """
        return self.metadata.extract_band_info(stac_info)

    def get_default_evalscript(self) -> str:
        """Get the default evalscript.
        
        Delegates to ProcessAPI._get_default_evalscript.
        """
        return self.process._get_default_evalscript()

    def get_collection_data_type(self, collection_id: str) -> str:
        """Get the data type for a collection.
        
        Args:
            collection_id: Collection ID (e.g., sentinel-2-l2a or byoc-uuid)
            
        Returns:
            Data type string (e.g., 'uint16', 'float32')
        """
        return self.metadata.get_collection_data_type(collection_id)

    def get_collection_nodata_value(self, collection_id: str) -> Optional[float]:
        """Get the nodata value for a collection.
        
        Args:
            collection_id: Collection ID (e.g., sentinel-2-l2a or byoc-uuid)
            
        Returns:
            Nodata value if available, None otherwise
        """
        return self.metadata.get_collection_nodata_value(collection_id)

    def search_images(
        self,
        collection: str,
        time_interval: Tuple[datetime, datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        byoc_id: Optional[str] = None,
        limit: int = 100,
        max_cloud_cover: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for images in a collection.
        
        Args:
            collection: Collection name (e.g., 'sentinel-2-l2a' or 'byoc')
            time_interval: Tuple of (start_time, end_time) as datetime objects
            bbox: Optional bounding box as (min_lon, min_lat, max_lon, max_lat)
            byoc_id: BYOC collection ID (required when collection is 'byoc')
            limit: Maximum number of results to return
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            List of dictionaries containing image metadata
        """
        return self.catalog.search_images(
            collection=collection,
            time_interval=time_interval,
            bbox=bbox,
            byoc_id=byoc_id,
            limit=limit,
            max_cloud_cover=max_cloud_cover
        ) 