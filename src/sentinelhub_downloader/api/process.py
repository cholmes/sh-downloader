"""Sentinel Hub Processing API functions."""

import logging
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

import numpy as np
from sentinelhub import (
    SentinelHubRequest, 
    BBox, 
    CRS, 
    MimeType, 
    DataCollection
)

from sentinelhub_downloader.api.client import SentinelHubClient

logger = logging.getLogger("sentinelhub_downloader")

class ProcessAPI:
    """Functions for interacting with the Sentinel Hub Processing API."""
    
    def __init__(self, client: SentinelHubClient):
        """Initialize the Process API.
        
        Args:
            client: SentinelHubClient instance
        """
        self.client = client
        self.sh_config = client.sh_config
    
    def create_dynamic_evalscript(
        self, 
        bands: List[str], 
        data_type: str = "AUTO"
    ) -> str:
        """Create a dynamic evalscript to extract specified bands."""
        if not bands:
            logger.warning("No bands provided for evalscript creation")
            return self._get_default_evalscript()
        
        # Create the input section with all bands
        bands_str = ', '.join([f'"{band}"' for band in bands])
        
        # Ensure data_type is uppercase
        data_type = data_type.upper()
        
        # For a single band, create a simple evalscript
        if len(bands) == 1:
            band = bands[0]
            return f"""
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
                        sampleType: "{data_type}"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [sample.{band}];
            }}
            """
        else:
            # For multiple bands, create a multi-band output
            return f"""
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
                        sampleType: "{data_type}"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'sample.{band}' for band in bands])}];
            }}
            """
    
    def _get_default_evalscript(self) -> str:
        """Get a default evalscript for fallback."""
        return """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"],
                    units: "DN"
                }],
                output: {
                    bands: 3,
                    sampleType: "AUTO"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """
    
    def process_image(
        self,
        collection: str,
        output_path: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        image_id: Optional[str] = None,
        date: Optional[str] = None,
        size: Tuple[int, int] = (512, 512),
        evalscript: Optional[str] = None,
        byoc_id: Optional[str] = None,
        specified_bands: Optional[List[str]] = None,
        data_type: str = "AUTO",
        nodata_value: Optional[float] = None,
        scale_metadata: Optional[float] = None,
    ) -> str:
        """Process and download an image using the sentinelhub-py library."""
        # Silence GDAL warnings about exceptions
        try:
            from osgeo import gdal
            gdal.UseExceptions()  # Explicitly enable exceptions
        except ImportError:
            pass  # GDAL not available, no need to worry about warnings
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine the data collection
        if collection.lower() == "byoc" and byoc_id:
            data_collection = DataCollection.define_byoc(byoc_id)
        else:
            try:
                data_collection = getattr(DataCollection, collection.upper().replace("-", "_"))
            except AttributeError:
                logger.warning(f"Collection {collection} not found in DataCollection, using byoc")
                if byoc_id:
                    data_collection = DataCollection.define_byoc(byoc_id)
                else:
                    raise ValueError(f"Unsupported collection: {collection}")
        
        # If bbox is not provided, try to get it from the image metadata
        if bbox is None:
            if image_id:
                logger.debug(f"No bbox provided, attempting to retrieve from image metadata for {image_id}")
                bbox = self._get_bbox_from_image_id(collection, image_id, byoc_id)
            elif date:
                logger.warning("No bbox provided and cannot automatically determine bbox from date alone")
                raise ValueError("bbox must be provided when using date parameter without image_id")
            else:
                raise ValueError("Either bbox or image_id must be provided")
        
        # Create a BBox object
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        
        # Create evalscript if not provided
        if not evalscript:
            bands_to_use = specified_bands or ["B04", "B03", "B02"]
            evalscript = self.create_dynamic_evalscript(bands_to_use, data_type)
        
        logger.debug(f"Using evalscript: {evalscript}")
        
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
            if isinstance(date, datetime):
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
        
        # Create the request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=input_data,
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=size,
            config=self.sh_config,
        )
        
        # Get the data - using the correct parameters
        logger.debug("Sending request to Sentinel Hub API...")
        
        # First get the data as bytes or numpy array
        data = request.get_data()
        
        if not data or len(data) == 0:
            logger.warning("No data received from Sentinel Hub API")
            return output_path
        
        # Write the data to file
        import rasterio
        from rasterio.transform import from_bounds
        
        # First item in the response contains our image data
        image_data = data[0]
        
        # Determine the number of bands
        if isinstance(image_data, np.ndarray):
            if len(image_data.shape) == 3:
                # Multi-band image
                height, width, bands = image_data.shape
            else:
                # Single-band image
                height, width = image_data.shape
                bands = 1
                # Reshape to 3D for consistent handling
                image_data = image_data.reshape((height, width, 1))
            
            # Create a GeoTIFF with the data
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=image_data.dtype,
                crs='EPSG:4326',
                transform=from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height),
                nodata=nodata_value
            ) as dst:
                # Write each band
                for b in range(bands):
                    dst.write(image_data[:, :, b], b + 1)
        else:
            # If it's not a numpy array, it might be bytes that we can write directly
            with open(output_path, 'wb') as f:
                f.write(image_data)
        
        # If we need to add scale metadata and GDAL is available, post-process the file
        if scale_metadata is not None:
            try:
                # Open the file to add metadata
                ds = gdal.Open(output_path, gdal.GA_Update)
                if ds is not None:
                    for band_number in range(1, ds.RasterCount + 1):
                        band = ds.GetRasterBand(band_number)
                        if band is None:
                            continue
                        
                        # Set scale metadata
                        band.SetScale(float(scale_metadata))
                        band.SetOffset(0.0)
                    
                    # Flush changes
                    ds.FlushCache()
                    ds = None
            except ImportError:
                logger.warning("GDAL not available, cannot set scale metadata")
            except Exception as e:
                logger.warning(f"Error setting metadata: {e}")
        
        # After writing the initial GeoTIFF, convert it to a COG
        try:
            from osgeo import gdal
            
            # Create a temporary filename for the COG
            temp_output_path = output_path + ".temp.tif"
            
            # Move the original file to the temporary path
            os.rename(output_path, temp_output_path)
            
            # Calculate statistics on the original file before converting to COG
            ds = gdal.Open(temp_output_path)
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                band.ComputeStatistics(False)  # False = accurate statistics
            ds = None
            
            # Convert to COG
            gdal_options = [
                '-co', 'COMPRESS=LZW',
                '-co', 'PREDICTOR=2',
                '-co', 'TILED=YES',
                '-co', 'BLOCKXSIZE=256',
                '-co', 'BLOCKYSIZE=256',
                '-co', 'COPY_SRC_OVERVIEWS=YES',
                '-co', 'BIGTIFF=IF_SAFER',
                '-co', 'INTERLEAVE=PIXEL',  # Include statistics in the output
                '-stats'  # Include statistics in the output
            ]
            
            # Add overviews if they don't exist
            ds = gdal.Open(temp_output_path)
            if ds.GetRasterBand(1).GetOverviewCount() == 0:
                gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
                ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
            ds = None
            
            # Create the COG
            gdal.Translate(output_path, temp_output_path, options=gdal_options)
            
            # Clean up temporary files
            os.remove(temp_output_path)
            
            # Clean up auxiliary files
            aux_xml = temp_output_path + ".aux.xml"
            if os.path.exists(aux_xml):
                os.remove(aux_xml)
            
            ovr_file = temp_output_path + ".ovr"
            if os.path.exists(ovr_file):
                os.remove(ovr_file)
            
            # Check for any other auxiliary files with similar patterns
            temp_dir = os.path.dirname(temp_output_path)
            temp_base = os.path.basename(temp_output_path)
            for filename in os.listdir(temp_dir):
                if filename.startswith(temp_base + "."):
                    full_path = os.path.join(temp_dir, filename)
                    logger.debug(f"Removing auxiliary file: {full_path}")
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove auxiliary file {full_path}: {e}")
            
            logger.debug(f"Converted output to Cloud-Optimized GeoTIFF with statistics")
            
        except ImportError:
            logger.warning("GDAL not available, cannot convert to Cloud-Optimized GeoTIFF")
        except Exception as e:
            logger.warning(f"Error converting to Cloud-Optimized GeoTIFF: {e}")
            # If conversion fails, ensure the original file is preserved
            if os.path.exists(temp_output_path) and not os.path.exists(output_path):
                os.rename(temp_output_path, output_path)
        
        return output_path 
    
    def _get_bbox_from_image_id(
        self, 
        collection: str, 
        image_id: str,
        byoc_id: Optional[str] = None
    ) -> Tuple[float, float, float, float]:
        """Get the bounding box for an image from its metadata.
        
        Args:
            collection: The collection name
            image_id: The image identifier
            byoc_id: The BYOC collection ID (if applicable)
            
        Returns:
            Tuple containing (min_x, min_y, max_x, max_y) in WGS84 coordinates
        
        Raises:
            ValueError: If the bbox cannot be determined
        """
        try:
            # Try to use the Catalog API if available
            from sentinelhub import CatalogAPI
            
            catalog = CatalogAPI(config=self.sh_config)
            
            # Determine the collection ID based on the collection name
            if collection.lower() == "byoc" and byoc_id:
                collection_id = byoc_id
            else:
                # Map collection names to Catalog API collection IDs
                collection_mapping = {
                    "SENTINEL2_L1C": "sentinel-2-l1c",
                    "SENTINEL2_L2A": "sentinel-2-l2a",
                    "SENTINEL1_IW": "sentinel-1-grd",
                    # Add more mappings as needed
                }
                collection_id = collection_mapping.get(collection.upper().replace("-", "_"))
                
                if not collection_id:
                    raise ValueError(f"Cannot map collection {collection} to a Catalog API collection ID")
            
            # Get the item metadata
            response = catalog.get_collection_item(collection_id, image_id)
            
            if not response or "geometry" not in response:
                raise ValueError(f"No geometry found in metadata for image {image_id}")
            
            # Extract the bbox from the geometry
            geometry = response["geometry"]
            if "bbox" in response:
                # Use the bbox if directly available
                return tuple(response["bbox"])
            elif geometry["type"] == "Polygon":
                # Calculate bbox from polygon coordinates
                coords = geometry["coordinates"][0]  # Outer ring
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                return (min(lons), min(lats), max(lons), max(lats))
            else:
                raise ValueError(f"Unsupported geometry type: {geometry['type']}")
                
        except ImportError:
            logger.warning("CatalogAPI not available, cannot retrieve bbox from image ID")
            raise ValueError("Cannot determine bbox: CatalogAPI not available")
        except Exception as e:
            logger.error(f"Error retrieving bbox for image {image_id}: {str(e)}")
            raise ValueError(f"Cannot determine bbox for image {image_id}: {str(e)}")