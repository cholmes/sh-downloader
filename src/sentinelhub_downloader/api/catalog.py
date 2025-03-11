"""Sentinel Hub Catalog API functions."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union

from sentinelhub_downloader.api.client import SentinelHubClient
from sentinelhub_downloader.utils import format_time_interval
from sentinelhub import SentinelHubCatalog, BBox, CRS

logger = logging.getLogger("sentinelhub_downloader")

class CatalogAPI:
    """Functions for interacting with the Sentinel Hub Catalog API."""
    
    def __init__(self, client: SentinelHubClient):
        """Initialize the Catalog API.
        
        Args:
            client: SentinelHubClient instance
        """
        self.client = client
        
    def search_images(
        self,
        collection: str,
        time_interval: Tuple[datetime, datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        max_cloud_cover: Optional[float] = None,
        byoc_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search the Sentinel Hub Catalog for available images."""
        
        # Validate inputs
        if collection.lower() == "byoc" and not byoc_id:
            raise ValueError("BYOC collection ID is required when collection is 'byoc'")
               
        # Build catalog ID
        catalog_id = collection.lower()
        if catalog_id == "byoc":
            catalog_id = f"byoc-{byoc_id}"
        
        # Add bounding box if provided, otherwise use global bbox
        if bbox:
            # Make sure bbox is a tuple of 4 floats
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}. Expected (min_lon, min_lat, max_lon, max_lat)")
            search_bbox = BBox(list(bbox), crs=CRS.WGS84)
        else:
            # Global bounding box: [-180, -90, 180, 90]
            search_bbox = BBox((-180, -90, 180, 90), crs=CRS.WGS84)
  

        catalog = SentinelHubCatalog(config=self.client.get_sh_config())        

        if max_cloud_cover is not None:
            if catalog_id == "sentinel-2-l2a" or catalog_id == "sentinel-2-l1c":
                filter = "eo:cloud_cover < " + str(max_cloud_cover)
            else:
                logger.warning(f"Cloud cover filtering is not supported for {catalog_id} collection")
                filter = None
        else:
            filter = None
            
        search_iterator = catalog.search(
            catalog_id,
            bbox=search_bbox,
            time=time_interval,
            filter=filter,
            fields={ "exclude": []},
        )


        results = list(search_iterator)
    
        # if there are results, log the first result
        if results:
            logger.debug("first result: " + str(results[0]))
        else:
            logger.debug("no results found")

        # Extract relevant metadata from each result
        processed_results = []
        for result in results:
            try:
                processed_result = {
                    'datetime': datetime.fromisoformat(result['properties']['datetime'].replace('Z', '+00:00')),
                    'geometry': result.get('geometry'),
                    'id': result.get('id'),
                    'bbox': result.get('bbox'),
                    'properties': result.get('properties', {})
                }
                processed_results.append(processed_result)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to process result: {e}")
        
        return processed_results

    def get_available_dates(
        self,
        collection: str,
        time_interval: Tuple[datetime, datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        byoc_id: Optional[str] = None,
        time_difference_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get available dates and metadata for a collection in a time interval.
        
        Args:
            collection: Collection name
            time_interval: Tuple of (start_date, end_date)
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            byoc_id: BYOC collection ID (required when collection is 'byoc')
            time_difference_days: If specified, filter dates to have at least this many days between them
            
        Returns:
            List of dictionaries containing datetime, geometry, and id for each result
        """
        logger.debug(f"Getting available dates for {collection} from {time_interval[0]} to {time_interval[1]}")
        
        # Search for images
        search_results = self.search_images(
            collection=collection,
            time_interval=time_interval,
            bbox=bbox,
            byoc_id=byoc_id,
            limit=1000
        )
        
        if not search_results:
            logger.debug("No images found for the specified parameters")
            return []
        
        # Sort results by date
        search_results.sort(key=lambda x: x['datetime'])
        
        # Apply time difference filter if specified
        if time_difference_days is not None and time_difference_days > 0:
            filtered_results = []
            last_added = None
            
            for result in search_results:
                if last_added is None or (result['datetime'] - last_added).days >= time_difference_days:
                    filtered_results.append(result)
                    last_added = result['datetime']
            
            logger.debug(f"Filtered from {len(search_results)} to {len(filtered_results)} results")
            search_results = filtered_results
        
        logger.debug(f"Found {len(search_results)} available dates")
        return search_results

    def get_stac_feature(self, collection_id: str, feature_id: str) -> Dict[str, Any]:
        """Get STAC feature information.
        
        Args:
            collection_id: Collection ID (e.g., sentinel-2-l2a or byoc-uuid)
            feature_id: Feature ID to retrieve
            
        Returns:
            Dictionary containing the feature information
        """
        response = self.client.get(
            f"{self.client.catalog_url}/collections/{collection_id}/items/{feature_id}"
        )
        return response.json()