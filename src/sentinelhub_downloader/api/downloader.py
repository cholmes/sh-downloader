"""Sentinel Hub image download functions."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

from sentinelhub_downloader.api.client import SentinelHubClient
from sentinelhub_downloader.api.process import ProcessAPI

logger = logging.getLogger("sentinelhub_downloader")

class DownloaderAPI:
    """Functions for downloading images from Sentinel Hub."""
    
    def __init__(self, client: SentinelHubClient, process_api: ProcessAPI):
        """Initialize the Downloader API."""
        self.client = client
        self.process_api = process_api
    
    def download_image(
        self,
        image_id: Optional[str],
        collection: str,
        bbox: Tuple[float, float, float, float],
        output_dir: str = "./downloads",
        byoc_id: Optional[str] = None,
        date: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        evalscript: Optional[str] = None,
        specified_bands: Optional[List[str]] = None,
        data_type: str = "AUTO",
        filename_template: Optional[str] = None,
        nodata_value: Optional[float] = None,
        scale_metadata: Optional[float] = None,
    ) -> str:
        """Download a specific image from Sentinel Hub."""
        # Check required parameters
        if collection.lower() == "byoc" and not byoc_id:
            raise ValueError("BYOC collection ID is required when collection is 'byoc'")
        
        if not image_id and not date:
            raise ValueError("Either image_id or date must be provided")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        if not filename_template:
            if image_id:
                filename_template = "{collection}_{image_id}.tiff"
            else:
                filename_template = "{collection}_{date}.tiff"
        
        # Format filename
        date_str = date.split("T")[0] if date else datetime.now().strftime("%Y-%m-%d")
        filename = filename_template.format(
            collection=collection,
            image_id=image_id or "unknown",
            date=date_str,
            byoc_id=byoc_id or ""
        )
        
        output_path = os.path.join(output_dir, filename)
        
        # Process and download the image using the sentinelhub-py library
        return self.process_api.process_image(
            collection=collection,
            image_id=image_id,
            bbox=bbox,
            output_path=output_path,
            date=date,
            size=size,
            evalscript=evalscript,
            byoc_id=byoc_id,
            specified_bands=specified_bands,
            data_type=data_type,
            nodata_value=nodata_value,
            scale_metadata=scale_metadata
        ) 