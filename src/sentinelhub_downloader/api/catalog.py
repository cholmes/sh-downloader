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
        
        # Format time interval
        time_from, time_to = format_time_interval(time_interval)
        
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
    
        logger.debug("first result: " + str(results[0]))

        # Sort results by datetime in descending order
        results.sort(
            key=lambda x: x.get("properties", {}).get("datetime", ""),
            reverse=True
        )
        
        return results
    
    def get_available_dates(
        self,
        collection: str,
        time_interval: Tuple[datetime, datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        byoc_id: Optional[str] = None,
        time_difference_days: Optional[int] = None,
    ) -> List[datetime]:
        """Get a list of dates with available images."""
        # Use a maximum limit of 100 per request (STAC API restriction)
        max_per_page = 100
        all_results = []
        
        # Implement pagination
        # First fetch with initial limit
        search_results = self.search_images(
            collection=collection,
            time_interval=time_interval,
            bbox=bbox,
            byoc_id=byoc_id,
            limit=max_per_page
        )
        all_results.extend(search_results)
        
        print(f"Found {len(search_results)} images in first page")
        # If we got a full page, we might need more pages
        while len(search_results) == max_per_page and len(all_results) < 2000:
            # Find the datetime of the last item to use as a filter
            if not search_results:
                break
            
            last_datetime = search_results[-1].get("properties", {}).get("datetime")
            #print(f"Last datetime: {last_datetime}")
            if not last_datetime:
                break
            
            # Parse the datetime
            try:
                last_date = datetime.strptime(last_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                try:
                    last_date = datetime.strptime(last_datetime, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    logger.warning(f"Could not parse date: {last_datetime}")
                    break
            
            # Create a new time interval from the last date to the end date
            # Add a tiny offset to exclude the last result from previous page
            new_start = last_date + timedelta(milliseconds=1)
            _, end_date = time_interval
            print(f"New start: {new_start}, End date: {end_date}")
            # If new start is after end date, we're done
            if new_start > end_date:
                break
            
            # print(f"Fetching next page from {new_start} to {end_date}")
            # Fetch the next page
            search_results = self.search_images(
                collection=collection,
                time_interval=(new_start, end_date),
                bbox=bbox,
                byoc_id=byoc_id,
                limit=max_per_page
            )
            all_results.extend(search_results)
        
        # Extract dates from all results
        dates = []
        for result in all_results:
            datetime_str = result.get("properties", {}).get("datetime")
            if datetime_str:
                try:
                    date = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                    dates.append(date)
                except ValueError:
                    try:
                        date = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
                        dates.append(date)
                    except ValueError:
                        logger.warning(f"Could not parse date: {datetime_str}")
        
        # Sort dates
        dates.sort()
        
        # Filter by time difference if requested
        if time_difference_days is not None and time_difference_days > 0:
            filtered_dates = []
            last_date = None
            
            for date in dates:
                if last_date is None or (date - last_date).days >= time_difference_days:
                    filtered_dates.append(date)
                    last_date = date
            
            return filtered_dates
        
        return dates 