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
        
        # Try STAC format first (item_assets format)
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
        
        # Try BYOC STAC format (summaries with eo:bands and raster:bands)
        elif "summaries" in collection_info and "eo:bands" in collection_info["summaries"]:
            eo_bands = collection_info["summaries"]["eo:bands"]
            raster_bands = collection_info["summaries"].get("raster:bands", [])
            
            # Create band info for each band
            for i, band_meta in enumerate(eo_bands):
                band_name = band_meta.get("name")
                if not band_name:
                    logger.warning(f"Band at index {i} has no name, skipping")
                    continue
                    
                # Get corresponding raster band info if available
                raster_info = {} 
                if i < len(raster_bands):
                    raster_info = raster_bands[i]
                    
                band_data = {
                    "name": band_name,
                    "description": band_meta.get("description", ""),
                    "data_type": raster_info.get("data_type"),
                    "nodata": raster_info.get("nodata"),
                    "unit": band_meta.get("unit"),
                    "scale": raster_info.get("scale"),
                    "offset": raster_info.get("offset")
                }
                
                band_info[band_name] = band_data
        
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
        
        # Log the extraction result
        if band_info:
            logger.debug(f"Extracted band info for {len(band_info)} bands: {list(band_info.keys())}")
        else:
            logger.debug("No band information found in collection metadata")
        
        return band_info
    
    def get_collection_data_type(self, collection_info: Dict[str, Any]) -> str:
        """Extract the data type from collection metadata.
        
        Args:
            collection_info: Collection metadata from STAC or BYOC API
            
        Returns:
            Data type if available, "AUTO" otherwise
        """
        # Try STAC format with item_assets
        if "item_assets" in collection_info:
            for asset_name, asset_info in collection_info["item_assets"].items():
                if "raster:bands" in asset_info and asset_info["raster:bands"]:
                    band = asset_info["raster:bands"][0]
                    if "data_type" in band:
                        return band["data_type"].upper()
        
        # Try BYOC STAC format with summaries
        if "summaries" in collection_info and "raster:bands" in collection_info["summaries"]:
            raster_bands = collection_info["summaries"]["raster:bands"]
            if raster_bands and len(raster_bands) > 0:
                # For simplicity, use the first band's data type
                if "data_type" in raster_bands[0]:
                    data_type = raster_bands[0]["data_type"].upper()
                    logger.debug(f"Found data type in STAC summaries: {data_type}")
                    return data_type
        
        # Try BYOC format
        if "data" in collection_info and "additionalData" in collection_info["data"]:
            additional_data = collection_info["data"]["additionalData"]
            if "bands" in additional_data:
                bands = additional_data["bands"]
                for band_name, band_data in bands.items():
                    if "sampleFormat" in band_data:
                        sample_format = band_data["sampleFormat"]
                        # Map sample format to data type
                        if sample_format == "UINT8":
                            return "UINT8"
                        elif sample_format == "UINT16":
                            return "UINT16"
                        elif sample_format in ["FLOAT32", "FLOAT64"]:
                            return "FLOAT32"
        
        logger.debug("No data type found in collection metadata, using AUTO")
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
        """Extract the nodata value from collection metadata.
        
        Args:
            collection_info: Collection metadata from STAC or BYOC API
            
        Returns:
            Nodata value if available, None otherwise
        """
        # Try STAC format with item_assets
        if "item_assets" in collection_info:
            for asset_name, asset_info in collection_info["item_assets"].items():
                if "raster:bands" in asset_info and asset_info["raster:bands"]:
                    band = asset_info["raster:bands"][0]
                    if "nodata" in band:
                        return band["nodata"]
        
        # Try BYOC STAC format with summaries
        if "summaries" in collection_info and "raster:bands" in collection_info["summaries"]:
            raster_bands = collection_info["summaries"]["raster:bands"]
            if raster_bands and len(raster_bands) > 0:
                # For simplicity, use the first band's nodata value
                # In a more sophisticated implementation, we might want to handle multiple different nodata values
                if "nodata" in raster_bands[0]:
                    nodata_value = raster_bands[0]["nodata"]
                    logger.debug(f"Found nodata value in STAC summaries: {nodata_value}")
                    return nodata_value
        
        # Try BYOC format
        if "data" in collection_info and "additionalData" in collection_info["data"]:
            additional_data = collection_info["data"]["additionalData"]
            if "bands" in additional_data:
                bands = additional_data["bands"]
                for band_name, band_data in bands.items():
                    if "noData" in band_data:
                        return band_data["noData"]
        
        logger.debug("No nodata value found in collection metadata")
        return None 