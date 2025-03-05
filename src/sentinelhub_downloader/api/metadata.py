"""Sentinel Hub metadata extraction functions."""

import logging
import json
from typing import Dict, Any, Optional, Tuple, List, Union

from sentinelhub_downloader.api.client import SentinelHubClient

logger = logging.getLogger("sentinelhub_downloader")

class MetadataAPI:
    """Functions for extracting and processing metadata from Sentinel Hub APIs."""
    
    def __init__(self, client: SentinelHubClient):
        """Initialize the Metadata API.
        
        Args:
            client: SentinelHubClient instance
        """
        self.client = client
    
    def get_stac_info(self, collection_id: str) -> Dict[str, Any]:
        """Get STAC collection information.
        
        Args:
            collection_id: Collection ID
            
        Returns:
            Collection metadata
        """
        response = self.client.get(f"{self.client.catalog_url}/collections/{collection_id}")
        return response.json()
    
    def get_byoc_info(self, byoc_id: str) -> Dict[str, Any]:
        """Get information about a BYOC collection using the BYOC API.
        
        Args:
            byoc_id: BYOC collection ID
            
        Returns:
            Collection metadata
        """
        logger.debug(f"Getting BYOC information for collection: {byoc_id}")
        
        try:
            # Make API request to BYOC API
            response = self.client.get(f"{self.client.byoc_url}/collections/{byoc_id}")
            return response.json()
        except Exception as e:
            logger.error(f"Error getting BYOC information: {e}")
            raise
    
    def extract_band_info(self, collection_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract band information from collection metadata.
        
        Args:
            collection_info: Collection metadata from STAC or BYOC API
            
        Returns:
            Dictionary of band information keyed by band name
        """
        band_info = {}
        
        # Try STAC format first
        if "item_assets" in collection_info:
            item_assets = collection_info["item_assets"]
            for asset_name, asset_info in item_assets.items():
                # Skip non-band assets
                if "roles" in asset_info and "data" not in asset_info["roles"]:
                    continue
                
                band_data = {
                    "name": asset_name,
                    "description": asset_info.get("title", ""),
                    "data_type": None,
                    "nodata": None,
                    "unit": None,
                    "scale": None,
                    "offset": None
                }
                
                # Extract raster band information if available
                if "raster:bands" in asset_info and asset_info["raster:bands"]:
                    band = asset_info["raster:bands"][0]  # Get first band
                    band_data.update({
                        "data_type": band.get("data_type"),
                        "nodata": band.get("nodata"),
                        "unit": band.get("unit"),
                        "scale": band.get("scale"),
                        "offset": band.get("offset")
                    })
                
                band_info[asset_name] = band_data
        
        # Try BYOC format
        elif "data" in collection_info and "additionalData" in collection_info["data"]:
            additional_data = collection_info["data"]["additionalData"]
            if "bands" in additional_data:
                bands = additional_data["bands"]
                for band_name, band_data in bands.items():
                    info = {
                        "name": band_name,
                        "description": "",
                        "data_type": band_data.get("sampleFormat", ""),
                        "nodata": band_data.get("noData"),
                        "unit": "",
                        "scale": None,
                        "offset": None,
                        "band_index": band_data.get("bandIndex"),
                        "source": band_data.get("source", "")
                    }
                    band_info[band_name] = info
        
        return band_info
    
    def get_collection_data_type(self, collection_info: Dict[str, Any]) -> str:
        """Extract default data type from collection metadata.
        
        Args:
            collection_info: Collection metadata
            
        Returns:
            Data type string or "AUTO" if not found
        """
        band_info = self.extract_band_info(collection_info)
        
        # Look for a consistent data type across bands
        data_types = set()
        for band, info in band_info.items():
            if info.get("data_type"):
                data_types.add(info["data_type"])
        
        # Return the first data type if there's only one
        if len(data_types) == 1:
            return next(iter(data_types))
        
        # Otherwise return AUTO
        return "AUTO"
    
    def get_collection_band_names(self, collection_info: Dict[str, Any]) -> List[str]:
        """Extract band names from collection metadata.
        
        Args:
            collection_info: Collection metadata
            
        Returns:
            List of band names
        """
        band_info = self.extract_band_info(collection_info)
        return list(band_info.keys())
    
    def get_collection_nodata_value(self, collection_info: Dict[str, Any]) -> Optional[float]:
        """Extract common nodata value from collection metadata.
        
        Args:
            collection_info: Collection metadata
            
        Returns:
            Nodata value or None if not found or inconsistent
        """
        band_info = self.extract_band_info(collection_info)
        
        # Look for a consistent nodata value across bands
        nodata_values = set()
        for band, info in band_info.items():
            if info.get("nodata") is not None:
                nodata_values.add(info["nodata"])
        
        # Return the first nodata value if there's only one
        if len(nodata_values) == 1:
            return next(iter(nodata_values))
        
        # Otherwise return None
        return None 