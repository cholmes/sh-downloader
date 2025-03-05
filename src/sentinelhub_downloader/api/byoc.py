"""Sentinel Hub BYOC-specific API functions."""

import logging
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

from sentinelhub import DataCollection, SentinelHubRequest, MimeType, BBox, CRS

from sentinelhub_downloader.api.client import SentinelHubClient
from sentinelhub_downloader.api.process import ProcessAPI
from sentinelhub_downloader.api.catalog import CatalogAPI
from sentinelhub_downloader.api.metadata import MetadataAPI

logger = logging.getLogger("sentinelhub_downloader")

class BYOCAPI:
    """Functions for working with Bring Your Own Collection (BYOC) data."""
    
    def __init__(
        self, 
        client: SentinelHubClient, 
        process_api: ProcessAPI,
        catalog_api: CatalogAPI,
        metadata_api: MetadataAPI
    ):
        """Initialize the BYOC API."""
        self.client = client
        self.process_api = process_api
        self.catalog_api = catalog_api
        self.metadata_api = metadata_api
        self.sh_config = client.sh_config
    
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
        """Download a time series of images from a BYOC collection."""
        logger.debug(f"Downloading BYOC time series for collection ID: {byoc_id}")
        logger.debug(f"Time interval: {time_interval[0]} to {time_interval[1]}")
        
        # Get available dates
        available_dates = self.catalog_api.get_available_dates(
            collection="byoc",
            byoc_id=byoc_id,
            time_interval=time_interval,
            bbox=bbox,
            time_difference_days=time_difference_days
        )
        
        if not available_dates:
            logger.warning("No available dates found for the specified parameters")
            return []
        
        # Default output directory
        if output_dir is None:
            output_dir = "./downloads"
        os.makedirs(output_dir, exist_ok=True)
        
        # Default filename template
        if filename_template is None:
            filename_template = "BYOC_{date}.tiff"
        
        # If no evalscript is provided, create one based on the bands
        if not evalscript:
            if auto_discover_bands and not specified_bands:
                # Try to discover bands from metadata
                try:
                    band_info = self.metadata_api.get_byoc_info(byoc_id)
                    if band_info and 'bands' in band_info:
                        specified_bands = [b['name'] for b in band_info['bands']]
                        logger.debug(f"Auto-discovered bands: {specified_bands}")
                except Exception as e:
                    logger.warning(f"Failed to auto-discover bands: {e}")
                    
                    # Fallback to specified bands if provided
                    if not specified_bands:
                        logger.debug("Using default bands")
                        specified_bands = ["B04", "B03", "B02"]  # Default fallback
            
            # Use specified bands for evalscript creation
            if specified_bands:
                logger.debug(f"Using specified bands: {specified_bands}")
                evalscript = self.process_api.create_dynamic_evalscript(specified_bands, data_type=data_type)
            else:
                # Use default evalscript
                evalscript = self.process_api._get_default_evalscript()
        
        logger.info(f"Found {len(available_dates)} dates with images")
        if specified_bands:
            logger.info(f"Using specified bands: {specified_bands}")
        
        downloaded_files = []
        
        # Download each date
        for date in available_dates:
            date_str = date.strftime("%Y-%m-%d")
            
            # Create filename from template
            filename = filename_template.format(
                collection=f"BYOC_{byoc_id[:8]}",
                date=date_str
            )
            
            try:
                # Use the output_path directly with process_image
                output_path = os.path.join(output_dir, filename)
                
                # Download the image using the sentinelhub-py library
                downloaded_path = self.process_api.process_image(
                    collection="byoc",
                    bbox=bbox,
                    output_path=output_path,
                    date=date_str,
                    size=size,
                    evalscript=evalscript,
                    byoc_id=byoc_id,
                    specified_bands=specified_bands,
                    data_type=data_type,
                    nodata_value=nodata_value,
                    scale_metadata=scale_metadata
                )
                
                downloaded_files.append(downloaded_path)
                logger.debug(f"Downloaded image for {date_str}: {downloaded_path}")
            except Exception as e:
                logger.error(f"Failed to download image for {date_str}: {e}")
        
        return downloaded_files 