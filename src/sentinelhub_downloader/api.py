"""API client for interacting with Sentinel Hub."""

import datetime
import json
import logging
import os
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile
import shutil
import numpy as np

import requests
import rasterio
from sentinelhub import CRS, BBox, SHConfig, SentinelHubCatalog, SentinelHubRequest
from sentinelhub import DataCollection as SHDataCollection
from sentinelhub import MimeType, SentinelHubDownloadClient, filter_times
from shapely.geometry import Polygon, box
from tqdm import tqdm

# Try to import GDAL
try:
    from osgeo import gdal
    # Set GDAL to not use exceptions by default (maintain backward compatibility)
    gdal.DontUseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from sentinelhub_downloader.config import Config

# Set up logger
logger = logging.getLogger("sentinelhub_downloader")


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


class SentinelHubAPI:
    """Client for the Sentinel Hub API."""

    # Mapping from user-friendly collection names to SentinelHub DataCollection attributes
    COLLECTION_MAPPING = {
        "sentinel-1-grd": "SENTINEL1_GRD",
        "sentinel-2-l1c": "SENTINEL2_L1C",
        "sentinel-2-l2a": "SENTINEL2_L2A",
        "sentinel-3-olci": "SENTINEL3_OLCI",
        "sentinel-5p-l2": "SENTINEL5P",
    }

    def __init__(self, config: Optional[Config] = None, debug: bool = False):
        """Initialize the API client with configuration."""
        self.config = config or Config()
        self.debug = debug
        self.sh_config = self._initialize_sh_config()
        self.catalog = SentinelHubCatalog(config=self.sh_config)
        self.download_client = SentinelHubDownloadClient(config=self.sh_config)
        
        if self.debug:
            logger.debug("SentinelHubAPI initialized with debug mode enabled")
            logger.debug(f"Using client_id: {self.config.get('client_id')[:5]}...")
            logger.debug(f"Using instance_id: {self.config.get('instance_id')[:5]}...")

    def _initialize_sh_config(self) -> SHConfig:
        """Initialize SentinelHub configuration."""
        sh_config = SHConfig()
        sh_config.sh_client_id = self.config.get("client_id")
        sh_config.sh_client_secret = self.config.get("client_secret")
        sh_config.instance_id = self.config.get("instance_id")
        
        # Set a custom JSON encoder that can handle datetime objects
        sh_config.json_encoder = DateTimeEncoder
        
        return sh_config

    def _get_data_collection(self, collection: str, byoc_id: Optional[str] = None):
        """
        Get the proper DataCollection enum value from the collection name.
        
        Args:
            collection: The user-friendly collection name (e.g., 'sentinel-2-l2a' or 'byoc')
            byoc_id: The BYOC collection ID (required if collection is 'byoc')
            
        Returns:
            The appropriate DataCollection enum value
        """
        if collection.lower() == "byoc":
            if not byoc_id:
                raise ValueError("BYOC collection ID is required for BYOC collections")
            
            if self.debug:
                logger.debug(f"Using BYOC collection with ID: {byoc_id}")
            
            return SHDataCollection.define_byoc(byoc_id)
        
        collection_id = self.COLLECTION_MAPPING.get(collection.lower())
        if not collection_id:
            raise ValueError(f"Unknown collection: {collection}")
        
        if self.debug:
            logger.debug(f"Mapped collection '{collection}' to '{collection_id}'")
        
        return getattr(SHDataCollection, collection_id)
    
    def _convert_datetime_to_str(self, time_interval: Tuple[datetime.datetime, datetime.datetime]) -> Tuple[str, str]:
        """
        Convert datetime objects to ISO format strings.
        
        Args:
            time_interval: A tuple of (start_time, end_time) as datetime objects
            
        Returns:
            A tuple of (start_time, end_time) as ISO format strings
        """
        start_time, end_time = time_interval
        return start_time.isoformat(), end_time.isoformat()

    def get_available_dates(
        self,
        collection: str,
        time_interval: Tuple[datetime.datetime, datetime.datetime],
        bbox: Tuple[float, float, float, float],
        byoc_id: Optional[str] = None,
        time_difference_days: Optional[int] = None
    ) -> List[datetime.datetime]:
        """
        Get available dates for a collection within a time interval and bounding box.
        
        Args:
            collection: The data collection name (e.g., 'sentinel-2-l2a' or 'byoc')
            time_interval: A tuple of (start_time, end_time) as datetime objects
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            byoc_id: The BYOC collection ID (required if collection is 'byoc')
            time_difference_days: Minimum days between returned dates (if None, return all dates)
            
        Returns:
            List of available dates as datetime objects
        """
        if self.debug:
            logger.debug(f"Getting available dates for collection: {collection}")
            logger.debug(f"Time interval: {time_interval[0]} to {time_interval[1]}")
            logger.debug(f"Bounding box: {bbox}")
            if byoc_id:
                logger.debug(f"BYOC ID: {byoc_id}")
            if time_difference_days is not None:
                logger.debug(f"Time difference filter: {time_difference_days} days")
        
        # Get the data collection
        data_collection = self._get_data_collection(collection, byoc_id)
        
        # Create a BBox object
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        
        # Search for the data using the format from the working code
        search_iterator = self.catalog.search(
            collection=data_collection,
            bbox=bbox_obj,
            time=time_interval  # Pass the time interval directly
        )
        
        if self.debug:
            logger.debug("Search request sent to Sentinel Hub API")
        
        # Convert to list to see if we got any results
        search_results = list(search_iterator)
        
        if self.debug:
            logger.debug(f"Found {len(search_results)} results")
            if search_results:
                logger.debug(f"First result: {search_results[0]}")
        
        # Extract timestamps from search results using the approach from the working code
        timestamps = []
        for item in search_results:
            try:
                # Get the datetime string from properties
                date_str = item['properties']['datetime']
                
                # Handle 'Z' timezone indicator
                if 'Z' in date_str:
                    date_str = date_str.replace('Z', '+00:00')
                    
                # Convert to datetime object
                timestamp = datetime.datetime.fromisoformat(date_str)
                timestamps.append(timestamp)
                
                if self.debug:
                    logger.debug(f"Extracted timestamp: {timestamp}")
            except (KeyError, ValueError) as e:
                if self.debug:
                    logger.warning(f"Could not parse date from item: {e}")
                    logger.warning(f"Item: {item}")
                continue
        
        # Sort timestamps chronologically
        timestamps.sort()
        
        # Filter timestamps based on time difference if specified
        if time_difference_days is not None:
            time_difference = datetime.timedelta(days=time_difference_days)
            filtered_dates = filter_times(timestamps, time_difference=time_difference)
            
            if self.debug:
                logger.debug(f"Filtered to {len(filtered_dates)} dates with minimum {time_difference_days} day(s) difference")
                if filtered_dates:
                    logger.debug(f"Available dates: {[d.strftime('%Y-%m-%d') for d in filtered_dates]}")
            
            return filtered_dates
        else:
            if self.debug:
                logger.debug(f"Returning all {len(timestamps)} dates without filtering")
                logger.debug(f"Available dates: {[d.strftime('%Y-%m-%d') for d in timestamps]}")
            
            return timestamps

    def search_catalog(
        self,
        collection: str,
        time_interval: Tuple[datetime.datetime, datetime.datetime],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        max_cloud_cover: Optional[float] = None,
        byoc_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search the Sentinel Hub Catalog for available images.

        Args:
            collection: The data collection name (e.g., 'sentinel-2-l2a' or 'byoc')
            time_interval: A tuple of (start_time, end_time) as datetime objects
            bbox: Optional bounding box as (min_lon, min_lat, max_lon, max_lat)
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            byoc_id: The BYOC collection ID (required if collection is 'byoc')

        Returns:
            List of image metadata dictionaries
        """
        if self.debug:
            logger.debug(f"Searching catalog for collection: {collection}")
            logger.debug(f"Time interval: {time_interval[0]} to {time_interval[1]}")
            logger.debug(f"Bounding box: {bbox}")
            logger.debug(f"Max cloud cover: {max_cloud_cover}")
            if byoc_id:
                logger.debug(f"BYOC ID: {byoc_id}")
        
        # Get the data collection
        data_collection = self._get_data_collection(collection, byoc_id)
        
        # Create search parameters
        search_params = {}
        
        # If no bbox is provided, use a global bounding box
        if bbox:
            bbox_obj = BBox(bbox, crs=CRS.WGS84)
        else:
            # Use a global bounding box (entire world)
            bbox_obj = BBox((-180, -90, 180, 90), crs=CRS.WGS84)
        
        # Add cloud cover filter if specified and applicable
        if max_cloud_cover is not None and "SENTINEL2" in str(data_collection):
            search_params["filter"] = {
                "op": "lte",
                "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
            }
        
        if self.debug:
            logger.debug("Search parameters:")
            logger.debug(f"Collection: {data_collection}")
            logger.debug(f"BBox: {bbox_obj}")
            logger.debug(f"Time interval: {time_interval}")
            logger.debug(f"Additional params: {search_params}")
        
        try:
            # Search using the format from the working code
            search_iterator = self.catalog.search(
                collection=data_collection,
                bbox=bbox_obj,
                time=time_interval,
                **search_params
            )
            
            if self.debug:
                logger.debug("Search request sent to Sentinel Hub API")
            
            # Convert the iterator to a list
            search_results = list(search_iterator)
            
            if self.debug:
                logger.debug(f"Received {len(search_results)} results from API")
                if search_results:
                    logger.debug("First result:")
                    logger.debug(f"{search_results[0]}")
                else:
                    logger.debug("No results returned from API")
            
            # Process the results to make them JSON serializable
            processed_results = []
            for result in search_results:
                # Create a copy of the result that we can modify
                processed_result = {}
                for key, value in result.items():
                    # Convert datetime objects to ISO format strings
                    if isinstance(value, datetime.datetime):
                        processed_result[key] = value.strftime("%Y-%m-%d")
                    # Handle nested dictionaries
                    elif isinstance(value, dict):
                        processed_result[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, datetime.datetime):
                                processed_result[key][sub_key] = sub_value.strftime("%Y-%m-%d")
                            else:
                                processed_result[key][sub_key] = sub_value
                    else:
                        processed_result[key] = value
                processed_results.append(processed_result)
            
            if self.debug:
                logger.debug(f"Processed {len(processed_results)} results")
            
            return processed_results
        except Exception as e:
            error_msg = f"Error searching catalog: {e}"
            print(error_msg)
            if self.debug:
                logger.error(error_msg)
                logger.exception(e)
            # Return an empty list on error
            return []

    def download_image(
        self,
        image_id: Optional[str],
        collection: str,
        bbox: Tuple[float, float, float, float],
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        date: Optional[Union[str, datetime.datetime]] = None,
        size: Tuple[int, int] = (512, 512),
        byoc_id: Optional[str] = None,
        evalscript: Optional[str] = None,
        nodata_value: Optional[float] = None,
        scale_metadata: Optional[float] = None,
    ) -> str:
        """
        Download a single image from Sentinel Hub.
        
        Args:
            image_id: The ID of the image to download (optional if date is provided)
            collection: The data collection name (e.g., 'sentinel-2-l2a')
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save the downloaded image
            filename: Filename for the downloaded image
            date: Specific date to download (optional if image_id is provided)
            size: Size of the output image as (width, height)
            byoc_id: The BYOC collection ID (required if collection is 'byoc')
            evalscript: Custom evalscript for processing the data
            nodata_value: Value to use for nodata pixels in the output GeoTIFF
            scale_metadata: Value to set as scale factor in the output GeoTIFF
            
        Returns:
            Path to the downloaded file
        """
        if self.debug:
            logger.debug(f"Downloading image from collection: {collection}")
            if image_id:
                logger.debug(f"Image ID: {image_id}")
            if date:
                logger.debug(f"Date: {date}")
            logger.debug(f"Bounding box: {bbox}")
            if byoc_id:
                logger.debug(f"BYOC ID: {byoc_id}")
            if nodata_value is not None:
                logger.debug(f"Using nodata value: {nodata_value}")
            if scale_metadata is not None:
                logger.debug(f"Setting scale metadata: {scale_metadata}")
        
        # Get the data collection
        data_collection = self._get_data_collection(collection, byoc_id)
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = self.config.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default filename if not provided
        if filename is None:
            if date:
                if isinstance(date, datetime.datetime):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = date
                filename = f"{collection}_{date_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"
            else:
                filename = f"{collection}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"
        
        output_path = Path(output_dir) / filename
        
        if self.debug:
            logger.debug(f"Output directory: {output_dir}")
            logger.debug(f"Output filename: {filename}")
        
        # Create bounding box for the request
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        
        try:
            # Define the request
            if evalscript:
                # Use the provided evalscript
                if self.debug:
                    logger.debug("Using custom evalscript")
            else:
                # Get the default evalscript for the collection
                evalscript = self._get_evalscript_for_collection(collection)
            
            if self.debug:
                logger.debug("Using evalscript:")
                logger.debug(evalscript)
            
            # Determine the output ID from the evalscript
            output_id = self._get_output_id_from_evalscript(evalscript)
            
            if self.debug:
                logger.debug(f"Using output ID: {output_id}")
            
            # Prepare input data based on whether we have an image_id or date
            if image_id:
                input_data = [
                    SentinelHubRequest.input_data(
                        data_collection=data_collection,
                        identifier=image_id,
                    )
                ]
            elif date:
                # Convert date to string if it's a datetime object
                if isinstance(date, datetime.datetime):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = date
                
                input_data = [
                    SentinelHubRequest.input_data(
                        data_collection=data_collection,
                        time_interval=(date_str, date_str)
                    )
                ]
            else:
                raise ValueError("Either image_id or date must be provided")
            
            # Create the request to get data as a NumPy array
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=input_data,
                responses=[SentinelHubRequest.output_response(output_id, MimeType.TIFF)],
                bbox=bbox_obj,
                size=size,
                config=self.sh_config,
            )
            
            if self.debug:
                logger.debug("Download request created")
                logger.debug("Sending download request to Sentinel Hub API...")
            
            # Get the data as a NumPy array
            response = request.get_data(save_data=False)
            
            if self.debug:
                logger.debug(f"Download response: {response}")
            
            # Check if we got a response
            if not response or len(response) == 0:
                if self.debug:
                    logger.warning("No response received from Sentinel Hub API")
                return str(output_path)
            
            # Process the NumPy array response
            image_data = response[0]
            
            if self.debug:
                logger.debug(f"Received data with shape: {image_data.shape}")
            
            # Get the number of bands
            if len(image_data.shape) == 3:
                # Multi-band image
                height, width, bands = image_data.shape
            else:
                # Single-band image
                height, width = image_data.shape
                bands = 1
            
            # Calculate statistics for each band
            stats = []
            if bands == 1:
                # Single band
                valid_mask = image_data != nodata_value if nodata_value is not None else np.ones_like(image_data, dtype=bool)
                valid_data = image_data[valid_mask]
                if len(valid_data) > 0:
                    stats.append({
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'valid_percent': float(np.sum(valid_mask) / valid_mask.size * 100)
                    })
                else:
                    stats.append(None)
            else:
                # Multi-band
                for b in range(bands):
                    band_data = image_data[:, :, b]
                    valid_mask = band_data != nodata_value if nodata_value is not None else np.ones_like(band_data, dtype=bool)
                    valid_data = band_data[valid_mask]
                    if len(valid_data) > 0:
                        stats.append({
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'valid_percent': float(np.sum(valid_mask) / valid_mask.size * 100)
                        })
                    else:
                        stats.append(None)
            
            if self.debug and stats:
                logger.debug(f"Calculated statistics for {len(stats)} bands")
                for i, s in enumerate(stats):
                    if s:
                        logger.debug(f"Band {i+1}: min={s['min']}, max={s['max']}, mean={s['mean']:.2f}, std={s['std']:.2f}, valid={s['valid_percent']:.2f}%")
            
            # First create a temporary GeoTIFF
            temp_path = str(output_path) + ".temp.tiff"
            
            # Create a GeoTIFF with the correct number of bands
            with rasterio.open(
                temp_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=image_data.dtype,
                crs='EPSG:4326',
                transform=rasterio.transform.from_bounds(
                    bbox[0], bbox[1], bbox[2], bbox[3], width, height
                ),
                nodata=nodata_value,
                # Add COG creation options
                compress='deflate',
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256
            ) as dst:
                # Write the data
                if bands == 1:
                    dst.write(image_data, 1)
                else:
                    for b in range(bands):
                        dst.write(image_data[:, :, b], b + 1)
            
            # Ensure the file is fully written before proceeding
            import time
            time.sleep(0.5)  # Small delay to ensure file system operations complete
            
            # Now convert to COG and add statistics using GDAL
            try:
                # Import GDAL here to avoid making it a hard dependency
                try:
                    from osgeo import gdal
                except ImportError:
                    if self.debug:
                        logger.warning("GDAL not available, cannot create COG or set metadata")
                    # If GDAL is not available, just rename the temp file to the output file
                    os.rename(temp_path, str(output_path))
                    return str(output_path)
                
                if self.debug:
                    logger.debug("Adding metadata and converting to Cloud Optimized GeoTIFF")
                
                # Force Python garbage collection to ensure file handles are released
                import gc
                gc.collect()
                
                # First, add statistics and scale to the temporary file
                ds_temp = gdal.OpenEx(temp_path, gdal.GA_Update)
                if ds_temp is not None:
                    if self.debug:
                        logger.debug("Adding statistics to temporary file")
                    
                    # Add statistics for each band
                    for band_number in range(1, ds_temp.RasterCount + 1):
                        band = ds_temp.GetRasterBand(band_number)
                        if band is None:
                            continue
                        
                        # Add statistics metadata if available
                        if stats and band_number <= len(stats) and stats[band_number-1]:
                            s = stats[band_number-1]
                            band.SetMetadataItem("STATISTICS_MINIMUM", str(s['min']))
                            band.SetMetadataItem("STATISTICS_MAXIMUM", str(s['max']))
                            band.SetMetadataItem("STATISTICS_MEAN", str(s['mean']))
                            band.SetMetadataItem("STATISTICS_STDDEV", str(s['std']))
                            band.SetMetadataItem("STATISTICS_VALID_PERCENT", str(s['valid_percent']))
                        
                        # Set the scale and offset if provided
                        if scale_metadata is not None:
                            band.SetScale(float(scale_metadata))
                            band.SetOffset(0.0)
                    
                    # Flush changes
                    ds_temp.FlushCache()
                    ds_temp = None
                
                # Set COG creation options
                translate_options = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=[
                        "COMPRESS=LZW",  # Use LZW compression instead of DEFLATE
                        "PREDICTOR=2",
                        "TILED=YES",
                        "BLOCKXSIZE=256",
                        "BLOCKYSIZE=256",
                    ],
                    metadataOptions=["COPY_SRC_METADATA=YES"]  # Ensure metadata is copied
                )
                
                # Convert to GeoTIFF with the metadata
                gdal.Translate(str(output_path), temp_path, options=translate_options)
                
                # Remove the temporary file
                os.remove(temp_path)
                
                # Now add overviews to make it a proper COG
                if self.debug:
                    logger.debug("Adding overviews to create a proper COG")
                
                # Open the output file to add overviews
                ds_out = gdal.Open(str(output_path), gdal.GA_Update)
                if ds_out is not None:
                    # Add overviews
                    overview_list = [2, 4, 8, 16]
                    ds_out.BuildOverviews("NEAREST", overview_list)
                    
                    # Close the dataset
                    ds_out = None
                    
                    if self.debug:
                        logger.debug(f"Added overviews at factors: {overview_list}")
                
                # Verify the metadata was set correctly
                if self.debug:
                    logger.debug("Verifying metadata in output file")
                    ds = gdal.Open(str(output_path))
                    if ds:
                        for band_number in range(1, ds.RasterCount + 1):
                            band = ds.GetRasterBand(band_number)
                            scale = band.GetScale()
                            offset = band.GetOffset()
                            metadata = band.GetMetadata()
                            logger.debug(f"Band {band_number}: Scale={scale}, Offset={offset}")
                            logger.debug(f"Band {band_number} metadata: {metadata}")
                        ds = None
            except Exception as e:
                # Log the error but continue
                if self.debug:
                    logger.warning(f"Failed to create COG or set metadata: {e}")
                
                # If there was an error, make sure we still have an output file
                if os.path.exists(temp_path) and not os.path.exists(str(output_path)):
                    os.rename(temp_path, str(output_path))
            
            if self.debug:
                logger.debug(f"Saved image data to: {output_path}")
            
            return str(output_path)
        except Exception as e:
            # Log the error and re-raise it
            error_msg = f"Error downloading image: {e}"
            print(error_msg)
            if self.debug:
                logger.error(error_msg)
                logger.exception(e)
            raise
    
    def download_byoc_timeseries(
        self,
        byoc_id: str,
        bbox: Tuple[float, float, float, float],
        time_interval: Tuple[datetime.datetime, datetime.datetime],
        output_dir: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        time_difference_days: Optional[int] = 1,
        filename_template: Optional[str] = None,
        evalscript: Optional[str] = None,
        auto_discover_bands: bool = True,
        specified_bands: Optional[List[str]] = None,
        nodata_value: Optional[float] = None,
        scale_metadata: Optional[float] = None,
    ) -> List[str]:
        """
        Download a time series of images from a BYOC collection.
        
        Args:
            byoc_id: The BYOC collection ID
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            time_interval: A tuple of (start_time, end_time) as datetime objects
            output_dir: Directory to save the downloaded images
            size: Size of the output images as (width, height)
            time_difference_days: Minimum days between downloaded images (if None, download all images)
            filename_template: Template for filenames (default: "{collection}_{date}.tiff")
            evalscript: Custom evalscript for processing the BYOC data
            auto_discover_bands: Whether to automatically discover and include all bands
            specified_bands: List of specific band names to download
            nodata_value: Value to use for nodata pixels in the output GeoTIFF
            scale_metadata: Value to set as SCALE metadata in the output GeoTIFF
            
        Returns:
            List of paths to the downloaded files
        """
        if self.debug:
            logger.debug(f"Downloading BYOC time series for collection ID: {byoc_id}")
            logger.debug(f"Time interval: {time_interval[0]} to {time_interval[1]}")
            logger.debug(f"Bounding box: {bbox}")
            if evalscript:
                logger.debug("Using custom evalscript")
            logger.debug(f"Auto discover bands: {auto_discover_bands}")
            if specified_bands:
                logger.debug(f"Specified bands: {specified_bands}")
            if nodata_value is not None:
                logger.debug(f"Using nodata value: {nodata_value}")
            if scale_metadata is not None:
                logger.debug(f"Setting SCALE metadata: {scale_metadata}")
        
        # Get available dates
        available_dates = self.get_available_dates(
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
            output_dir = self.config.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        
        # Default filename template
        if filename_template is None:
            filename_template = "BYOC_{date}.tiff"
        
        # If no custom evalscript is provided, create one based on specified bands or auto-discovery
        if not evalscript:
            if specified_bands:
                # Use the specified bands
                evalscript = self.create_dynamic_evalscript(specified_bands)
                if self.debug:
                    logger.debug(f"Created evalscript for specified bands: {specified_bands}")
            elif auto_discover_bands:
                # Try to discover bands
                bands = self.get_byoc_bands(byoc_id, time_interval, bbox)
                if bands:
                    evalscript = self.create_dynamic_evalscript(bands)
                    if self.debug:
                        logger.debug(f"Using auto-discovered bands: {bands}")
        
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
                # Download the image
                output_path = self.download_image(
                    image_id=None,
                    collection="byoc",
                    byoc_id=byoc_id,
                    bbox=bbox,
                    output_dir=output_dir,
                    filename=filename,
                    date=date,
                    size=size,
                    evalscript=evalscript,
                    nodata_value=nodata_value,
                    scale_metadata=scale_metadata
                )
                
                downloaded_files.append(output_path)
                
                if self.debug:
                    logger.debug(f"Downloaded image for date {date_str}: {output_path}")
            except Exception as e:
                logger.error(f"Error downloading image for date {date_str}: {e}")
                continue
        
        return downloaded_files
    
    def _get_evalscript_for_collection(self, collection: str) -> str:
        """Get the appropriate evalscript for the collection."""
        if collection.lower() == "byoc":
            # More flexible evalscript for BYOC collections
            # This handles both single-band and multi-band data
            return """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B0", "B1", "B2"],
                        units: "DN"
                    }],
                    output: {
                        id: "default",
                        bands: 3,
                        sampleType: "AUTO"
                    }
                };
            }

            function evaluatePixel(sample) {
                // Check if we have all three bands
                if (sample.B0 !== undefined && sample.B1 !== undefined && sample.B2 !== undefined) {
                    return [sample.B0, sample.B1, sample.B2];
                }
                // If we only have one band, return it as grayscale
                else if (sample.B0 !== undefined) {
                    return [sample.B0, sample.B0, sample.B0];
                }
                // Fallback to zeros if no bands are available
                else {
                    return [0, 0, 0];
                }
            }
            """
        elif "sentinel-2" in collection.lower():
            # Basic RGB composite for Sentinel-2
            return """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3,
                        sampleType: "UINT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
            """
        elif "sentinel-1" in collection.lower():
            # Basic VV-VH composite for Sentinel-1
            return """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["VV", "VH"]
                    }],
                    output: {
                        bands: 2,
                        sampleType: "FLOAT32"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.VV, sample.VH];
            }
            """
        else:
            # Generic evalscript for other collections
            return """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B0", "B1", "B2"]
                    }],
                    output: {
                        bands: 3,
                        sampleType: "UINT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B0, sample.B1, sample.B2];
            }
            """ 

    def _get_output_id_from_evalscript(self, evalscript: str) -> str:
        """
        Extract the output ID from an evalscript.
        
        Args:
            evalscript: The evalscript to parse
            
        Returns:
            The output ID (defaults to "default" if not found)
        """
        # Look for output ID in the evalscript
        if "id:" in evalscript:
            # Try to extract the ID from the output section
            try:
                # Find the output section
                output_start = evalscript.find("output:")
                if output_start != -1:
                    # Find the id field within the output section
                    id_start = evalscript.find("id:", output_start)
                    if id_start != -1:
                        # Extract the value after "id:"
                        id_value_start = id_start + 3
                        # Find the next quote or comma
                        quote_start = evalscript.find('"', id_value_start)
                        if quote_start != -1:
                            quote_end = evalscript.find('"', quote_start + 1)
                            if quote_end != -1:
                                return evalscript[quote_start + 1:quote_end]
                        
                        # Try with single quotes
                        quote_start = evalscript.find("'", id_value_start)
                        if quote_start != -1:
                            quote_end = evalscript.find("'", quote_start + 1)
                            if quote_end != -1:
                                return evalscript[quote_start + 1:quote_end]
            except Exception as e:
                if self.debug:
                    logger.warning(f"Error parsing evalscript for output ID: {e}")
        
        # Default to "default" if we couldn't find an ID
        return "default"

    def get_byoc_bands(
        self,
        byoc_id: str,
        time_interval: Tuple[datetime.datetime, datetime.datetime],
        bbox: Tuple[float, float, float, float],
    ) -> List[str]:
        """
        Discover available bands in a BYOC collection.
        
        Args:
            byoc_id: The BYOC collection ID
            time_interval: A tuple of (start_time, end_time) as datetime objects
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            List of band names available in the collection
        """
        if self.debug:
            logger.debug(f"Discovering bands for BYOC collection: {byoc_id}")
        
        # Get the data collection
        data_collection = self._get_data_collection("byoc", byoc_id)
        
        # Create a BBox object
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        
        # Search for the data
        search_iterator = self.catalog.search(
            collection=data_collection,
            bbox=bbox_obj,
            time=time_interval
        )
        
        # Get the first result to examine its properties
        search_results = list(search_iterator)
        
        if not search_results:
            if self.debug:
                logger.warning("No results found for band discovery")
            return []
        
        # Try to extract band information from the first result
        first_result = search_results[0]
        
        # Look for band information in various places
        bands = []
        
        # Check in properties.eo:bands
        if 'properties' in first_result and 'eo:bands' in first_result['properties']:
            eo_bands = first_result['properties']['eo:bands']
            if isinstance(eo_bands, list):
                for band in eo_bands:
                    if isinstance(band, dict) and 'name' in band:
                        bands.append(band['name'])
        
        # Check in assets
        if not bands and 'assets' in first_result:
            assets = first_result['assets']
            if isinstance(assets, dict):
                # Extract band names from asset keys
                for asset_name in assets.keys():
                    # Skip common non-band assets
                    if asset_name.lower() not in ['thumbnail', 'overview', 'metadata']:
                        bands.append(asset_name)
        
        # If we still don't have bands, try to test common bands individually
        if not bands:
            if self.debug:
                logger.debug("No bands found in metadata, testing common bands individually")
            
            # Try some common band names
            potential_bands = [
                # Common optical bands
                'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12',
                # Simplified band names
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
                # Generic band names
                'B0', 'B1', 'B2',
                # SAR bands
                'VV', 'VH', 'HH', 'HV',
                # Common derived bands
                'SCL', 'CLD', 'SNW', 'QA60',
                # Custom bands that might be in BYOC collections
                'SWC', 'NDVI', 'EVI', 'NDWI', 'NDSI'
            ]
            
            # Test each band individually
            verified_bands = []
            for band in potential_bands:
                # Create a test evalscript for this single band
                test_evalscript = f"""
                //VERSION=3
                function setup() {{
                    return {{
                        input: [{{
                            bands: ["{band}"],
                            units: "DN"
                        }}],
                        output: {{
                            id: "default",
                            bands: 1,
                            sampleType: "AUTO"
                        }}
                    }};
                }}

                function evaluatePixel(sample) {{
                    return [sample.{band}];
                }}
                """
                
                # Try to create a request with this band
                try:
                    # Get a date from the available dates
                    if len(search_results) > 0 and 'properties' in search_results[0] and 'datetime' in search_results[0]['properties']:
                        date_str = search_results[0]['properties']['datetime']
                        if 'Z' in date_str:
                            date_str = date_str.replace('Z', '+00:00')
                        date = datetime.datetime.fromisoformat(date_str)
                        date_str = date.strftime("%Y-%m-%d")
                    else:
                        # Use the start of the time interval
                        date_str = time_interval[0].strftime("%Y-%m-%d")
                    
                    # Create a small request to test if the band exists
                    request = SentinelHubRequest(
                        evalscript=test_evalscript,
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=data_collection,
                                time_interval=(date_str, date_str)
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        bbox=bbox_obj,
                        size=(10, 10),  # Small size for quick testing
                        config=self.sh_config,
                    )
                    
                    # Try to get data without actually downloading
                    # This will raise an exception if the band doesn't exist
                    request.get_data(save_data=False)
                    
                    # If we get here, the band exists
                    verified_bands.append(band)
                    if self.debug:
                        logger.debug(f"Verified band exists: {band}")
                    
                except Exception as e:
                    # Band doesn't exist or other error
                    if self.debug:
                        logger.debug(f"Band {band} not available: {str(e)}")
                    continue
            
            bands = verified_bands
        
        if self.debug:
            logger.debug(f"Discovered bands: {bands}")
        
        return bands

    def create_dynamic_evalscript(self, bands: List[str]) -> str:
        """
        Create a dynamic evalscript that includes all specified bands.
        
        Args:
            bands: List of band names to include
            
        Returns:
            An evalscript that will return all specified bands
        """
        if not bands:
            if self.debug:
                logger.warning("No bands provided for evalscript creation")
            return self._get_evalscript_for_collection("byoc")
        
        # Create the input section with all bands
        bands_str = ', '.join([f'"{band}"' for band in bands])
        
        # For a single band, create a grayscale image
        if len(bands) == 1:
            band = bands[0]
            evalscript = f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: ["{band}"],
                        units: "DN"
                    }}],
                    output: {{
                        id: "default",
                        bands: 1,
                        sampleType: "AUTO"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [sample.{band}];
            }}
            """
        else:
            # For multiple bands, include all of them
            evalscript = f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: [{bands_str}],
                        units: "DN"
                    }}],
                    output: {{
                        id: "default",
                        bands: {len(bands)},
                        sampleType: "AUTO"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'sample.{band}' for band in bands])}];
            }}
            """
        
        if self.debug:
            logger.debug("Created dynamic evalscript:")
            logger.debug(evalscript)
        
        return evalscript 