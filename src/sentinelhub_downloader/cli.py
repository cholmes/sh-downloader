"""Command-line interface for Sentinel Hub Downloader."""

import datetime
import logging
import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import click
from tqdm import tqdm

from sentinelhub_downloader.api import SentinelHubAPI
from sentinelhub_downloader.config import Config
from sentinelhub_downloader.utils import get_date_range, parse_bbox

# Set up logging
logger = logging.getLogger("sentinelhub_downloader")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def is_uuid(collection_id: str) -> Tuple[bool, Optional[str]]:
    """Check if a string is a UUID and return the valid UUID part if it is.
    
    Args:
        collection_id: String to check
        
    Returns:
        Tuple of (is_uuid, uuid_part) where uuid_part is the valid UUID if found, None otherwise
    """
    if '-' in collection_id and len(collection_id) >= 32:
        try:
            # Try to parse the first 36 characters as a UUID
            uuid_part = collection_id[:36] if len(collection_id) > 36 else collection_id
            uuid.UUID(uuid_part)
            
            # If we successfully parsed a UUID but the string is longer, return the valid part
            if len(collection_id) > 36:
                logger.warning(f"Trimming collection ID to valid UUID format: {uuid_part}")
            
            return True, uuid_part
        except (ValueError, AttributeError):
            pass
    
    return False, None


def setup_api_from_context(ctx: click.Context) -> Tuple[SentinelHubAPI, bool]:
    """Set up the API client from the Click context.
    
    Args:
        ctx: Click context
        
    Returns:
        Tuple of (api_client, debug_flag)
    """
    debug = ctx.obj.get("DEBUG", False)
    config = Config()
    
    # Check if configured
    if not config.is_configured():
        click.echo("Sentinel Hub Downloader is not configured. Running configuration wizard...")
        config.configure_wizard()
    
    # Set up API client with debug flag
    api = SentinelHubAPI(config, debug=debug)
    
    # Set logger level based on debug flag
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    return api, debug


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False, 
    help="Enable debug logging",
)
@click.pass_context
def cli(ctx, debug):
    """Download satellite imagery from Sentinel Hub as GeoTIFFs."""
    # Set up context object to pass debug flag to commands
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    
    # Set logging level based on debug flag
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logger.setLevel(logging.INFO)


@cli.command()
@click.pass_context
def configure(ctx):
    """Configure the Sentinel Hub Downloader with your credentials."""
    config = Config()
    config.configure_wizard()


@cli.command()
@click.argument(
    "collection_id",
    required=True
)
@click.option(
    "--start",
    "-s",
    help="Start date (YYYY-MM-DD). Defaults to 30 days ago.",
)
@click.option(
    "--end",
    "-e",
    help="End date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--bbox",
    "-b",
    help="Bounding box as min_lon,min_lat,max_lon,max_lat. Default is global.",
    callback=parse_bbox,
)
@click.option(
    "--max-cloud-cover",
    "-m",
    type=float,
    help="Maximum cloud cover percentage (0-100). Only applies to optical sensors.",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Maximum number of results to display. Default is 10.",
)
@click.pass_context
def search(
    ctx,
    collection_id: str,
    start: Optional[str],
    end: Optional[str],
    bbox: Optional[str],
    max_cloud_cover: Optional[float],
    limit: int,
):
    """Search for available satellite imagery without downloading.
    
    COLLECTION_ID can be a standard collection ID (e.g., sentinel-2-l2a) or a BYOC UUID.
    """
    api, debug = setup_api_from_context(ctx)
    
    # Check if the collection ID is a UUID (BYOC collection)
    is_uuid_flag, byoc_id = is_uuid(collection_id)
    
    if debug:
        logger.debug(f"Collection ID '{collection_id}' identified as UUID: {is_uuid_flag}")
    
    # Determine the collection type
    collection = "byoc" if is_uuid_flag else collection_id
    
    # Parse date range
    start_date, end_date = get_date_range(start, end)
    click.echo(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Parse bounding box if provided
    bbox_tuple = bbox  # bbox is already parsed by the callback
    
    # Search for images
    if is_uuid_flag:
        click.echo(f"Searching for BYOC collection {byoc_id}...")
    else:
        click.echo(f"Searching for {collection} images...")
    
    search_results = api.search_images(
        collection=collection,
        time_interval=(start_date, end_date),
        bbox=bbox_tuple,
        max_cloud_cover=max_cloud_cover,
        byoc_id=byoc_id,
        limit=limit,
    )
    
    if not search_results:
        click.echo("No images found matching the criteria.")
        return
    
    # Display results
    click.echo(f"Found {len(search_results)} images. Showing first {min(limit, len(search_results))}:")
    
    for i, result in enumerate(search_results[:limit]):
        image_id = result["id"]
        date = result.get("properties", {}).get("datetime", "unknown_date")
        cloud_cover = result.get("properties", {}).get("eo:cloud_cover", "N/A")
        
        click.echo(f"[{i+1}] ID: {image_id}")
        click.echo(f"    Date: {date}")
        if cloud_cover != "N/A":
            click.echo(f"    Cloud Cover: {cloud_cover}%")
        click.echo("")


@cli.command()
@click.argument(
    "byoc_id",
    required=True
)
@click.option(
    "--image-id",
    "-i",
    help="Image ID to download",
)
@click.option(
    "--start",
    "-s",
    help="Start date (YYYY-MM-DD). Defaults to 30 days ago.",
)
@click.option(
    "--end",
    "-e",
    help="End date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--bbox",
    callback=parse_bbox,
    help="Bounding box in format: <min_lon> <min_lat> <max_lon> <max_lat>. If not provided, will download globally.",
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory for downloaded images",
)
@click.option(
    "--size",
    help="Size of the output image as width,height (default: 512,512)",
    default="512,512",
)
@click.option(
    "--time-difference",
    "-t",
    type=int,
    help="Minimum days between downloaded images (default: None - download all images)",
)
@click.option(
    "--all-dates/--filter-dates",
    default=False,
    help="Download all available dates without filtering (overrides --time-difference)",
)
@click.option(
    "--filename-template",
    "-f",
    help="Template for filenames (default: 'BYOC_{byoc_id}_{date}.tiff')",
)
@click.option(
    "--evalscript-file",
    help="Path to a file containing a custom evalscript",
)
@click.option(
    "--bands",
    help="Comma-separated list of band names to download (e.g., 'SWC,dataMask')",
)
@click.option(
    "--auto-discover-bands/--no-auto-discover-bands",
    default=True,
    help="Automatically discover and include all bands (default: True)",
)
@click.option(
    "--nodata",
    type=float,
    help="Value to use for nodata pixels in the output GeoTIFF",
)
@click.option(
    "--scale",
    type=float,
    help="Value to set as SCALE metadata in the output GeoTIFF",
)
@click.option(
    "--data-type",
    type=click.Choice([
        "auto",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "cint16",
        "cint32",
        "cfloat32",
        "cfloat64"
    ], case_sensitive=False),
    default="AUTO",
    help="Output data type (default: AUTO)",
)
@click.pass_context
def byoc(
    ctx,
    byoc_id: str,
    image_id: Optional[str],
    start: Optional[str],
    end: Optional[str],
    bbox: Optional[str],
    output_dir: Optional[str],
    size: str,
    time_difference: Optional[int],
    all_dates: bool,
    filename_template: Optional[str],
    evalscript_file: Optional[str],
    bands: Optional[str],
    auto_discover_bands: bool,
    nodata: Optional[float],
    scale: Optional[float],
    data_type: str,
):
    """Download images from a BYOC collection.
    
    BYOC_ID is the UUID of your Bring Your Own COG Collection.
    """
    api, debug = setup_api_from_context(ctx)
    
    # Validate BYOC ID is a UUID
    is_uuid_flag, valid_byoc_id = is_uuid(byoc_id)
    if not is_uuid_flag:
        click.echo(f"Error: {byoc_id} does not appear to be a valid UUID for a BYOC collection")
        return
    
    byoc_id = valid_byoc_id
    
    # Parse date range
    start_date, end_date = get_date_range(start, end)
    click.echo(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Parse bounding box
    bbox_tuple = None
    if bbox:
        bbox_tuple = parse_bbox(bbox)
        logger.debug(f"Bounding box: {bbox_tuple}")
    else:
        logger.debug("No bounding box provided - will use each image's own bbox")
    
    # If a specific image ID is provided, download just that image
    if image_id:
        click.echo(f"Downloading specific image: {image_id}")
        try:
            # Load evalscript from file if provided
            evalscript = None
            if evalscript_file:
                try:
                    with open(evalscript_file, "r") as f:
                        evalscript = f.read()
                    click.echo(f"Loaded evalscript from {evalscript_file}")
                except Exception as e:
                    click.echo(f"Error loading evalscript: {e}")
                    return
            
            # Parse bands if provided
            specified_bands = None
            if bands:
                specified_bands = [b.strip() for b in bands.split(",")]
                click.echo(f"Using specified bands: {specified_bands}")
            
            # If bbox is not provided, it will be retrieved from the image metadata
            output_path = api.download_image(
                image_id=image_id,
                collection="byoc",
                byoc_id=byoc_id,
                bbox=bbox_tuple,
                output_dir=output_dir,
                size=tuple(map(int, size.split(","))),
                evalscript=evalscript,
                specified_bands=specified_bands,
                nodata_value=nodata,
                scale_metadata=scale,
                data_type=data_type.upper()
            )
            click.echo(f"Image downloaded to: {output_path}")
            return
        except Exception as e:
            click.echo(f"Error downloading image: {e}")
            if debug:
                import traceback
                click.echo(traceback.format_exc())
            return
    
    # Parse size
    size_tuple = tuple(map(int, size.split(",")))

    # Determine time difference
    effective_time_difference = None
    if not all_dates:
        effective_time_difference = time_difference

    # For searching images, we need a bbox
    if not bbox_tuple:
        # If no bbox is provided for search, use the collection's bbox
        try:
            collection_info = api.get_stac_info(f"byoc-{byoc_id}")
            if "extent" in collection_info and "spatial" in collection_info["extent"]:
                spatial = collection_info["extent"]["spatial"]
                if "bbox" in spatial and spatial["bbox"]:
                    bbox_tuple = tuple(spatial["bbox"][0])
                    click.echo(f"Using collection's bbox for search: {bbox_tuple}")
                else:
                    click.echo("No bbox found in collection metadata")
                    return
            else:
                click.echo("No spatial extent found in collection metadata")
                return
        except Exception as e:
            click.echo(f"Error retrieving collection bbox: {e}")
            click.echo("Please provide a bbox for searching")
            return
    
    # Load evalscript from file if provided
    evalscript = None
    if evalscript_file:
        try:
            with open(evalscript_file, "r") as f:
                evalscript = f.read()
            click.echo(f"Loaded evalscript from {evalscript_file}")
        except Exception as e:
            click.echo(f"Error loading evalscript: {e}")
            return
    
    # Parse bands if provided
    specified_bands = None
    if bands:
        specified_bands = [b.strip() for b in bands.split(",")]
        click.echo(f"Using specified bands: {specified_bands}")
        # Disable auto-discovery if bands are specified
        auto_discover_bands = False
    
    # If no evalscript is provided but bands are specified, create a dynamic evalscript
    if not evalscript and specified_bands:
        evalscript = api.create_dynamic_evalscript(specified_bands, data_type=data_type)
        click.echo("Created dynamic evalscript for specified bands")
        if debug:
            click.echo(f"Evalscript:\n{evalscript}")
    
    # If we're using each image's own bbox, we need to modify the download approach
    if bbox_tuple is None:
        # Get available dates
        available_dates = api.get_available_dates(
            collection="byoc",
            byoc_id=byoc_id,
            time_interval=(start_date, end_date),
            bbox=bbox_tuple,  # Using collection bbox for search
            time_difference_days=effective_time_difference
        )
        
        if not available_dates:
            click.echo("No images found for the specified criteria")
            return
        
        click.echo(f"Found {len(available_dates)} dates with images")
        
        # Download each image individually using its own bbox
        downloaded_files = []
        for date in tqdm(available_dates, desc="Downloading images", unit="date"):
            try:
                # Search for images on this date
                search_results = api.search_images(
                    collection="byoc",
                    byoc_id=byoc_id,
                    time_interval=(date, date),
                    bbox=bbox_tuple  # Using collection bbox for search
                )
                
                if not search_results:
                    click.echo(f"No images found for date {date}")
                    continue
                
                # Download each image using its own bbox
                for result in search_results:
                    image_id = result["id"]
                    click.echo(f"Downloading image {image_id} from {date}...")
                    
                    # Extract bbox from the search result if available
                    image_bbox = None
                    if "bbox" in result:
                        image_bbox = tuple(result["bbox"])
                        click.echo(f"  Using bbox from search result: {image_bbox}")
                    elif "geometry" in result and result["geometry"]["type"] == "Polygon":
                        # Calculate bbox from polygon coordinates
                        coords = result["geometry"]["coordinates"][0]  # Outer ring
                        lons = [p[0] for p in coords]
                        lats = [p[1] for p in coords]
                        image_bbox = (min(lons), min(lats), max(lons), max(lats))
                        click.echo(f"  Calculated bbox from geometry: {image_bbox}")
                    
                    # The bbox will be retrieved from the image metadata if not available in search result
                    output_path = api.download_image(
                        image_id=image_id,
                        collection="byoc",
                        byoc_id=byoc_id,
                        bbox=image_bbox,  # Use bbox from search result if available
                        output_dir=output_dir,
                        size=size_tuple,
                        evalscript=evalscript,
                        specified_bands=specified_bands,
                        nodata_value=nodata,
                        scale_metadata=scale,
                        data_type=data_type.upper()
                    )
                    
                    downloaded_files.append(output_path)
                    click.echo(f"  Downloaded to: {output_path}")
            
            except Exception as e:
                click.echo(f"Error processing date {date}: {e}")
                if debug:
                    import traceback
                    click.echo(traceback.format_exc())
    else:
        # Use the existing timeseries download function
        downloaded_files = api.download_byoc_timeseries(
            byoc_id=byoc_id,
            bbox=bbox_tuple,
            time_interval=(start_date, end_date),
            output_dir=output_dir,
            size=size_tuple,
            time_difference_days=effective_time_difference,
            filename_template=filename_template,
            evalscript=evalscript,
            auto_discover_bands=auto_discover_bands,
            specified_bands=specified_bands,
            nodata_value=nodata,
            scale_metadata=scale,
            data_type=data_type.upper()
        )
    
    if downloaded_files:
        click.echo(f"Successfully downloaded {len(downloaded_files)} images")
    else:
        click.echo("No images were downloaded")


@cli.command()
@click.argument(
    "collection_id",
    required=True
)
@click.option(
    "--raw/--formatted",
    default=False,
    help="Display raw JSON or formatted output (default: formatted)",
)
@click.option(
    "--byoc-api/--stac-api",
    default=False,
    help="Use BYOC API instead of STAC API for BYOC collections (default: STAC API)",
)
@click.pass_context
def info(
    ctx,
    collection_id: str,
    raw: bool,
    byoc_api: bool,
):
    """Get information about a collection (standard or BYOC).
    
    COLLECTION_ID can be a standard collection ID (e.g., sentinel-2-l2a) or a BYOC UUID.
    """
    api, debug = setup_api_from_context(ctx)
    
    # Check if the collection ID is a UUID (BYOC collection)
    is_uuid_flag, valid_uuid = is_uuid(collection_id)
    
    if debug:
        logger.debug(f"Collection ID '{collection_id}' identified as UUID: {is_uuid_flag}")
    
    if is_uuid_flag:
        collection_id = valid_uuid
    
    # Determine which API to use
    if is_uuid_flag and byoc_api:
        # Use BYOC API for BYOC collections if requested
        click.echo(f"Getting BYOC information for collection: {collection_id}")
        try:
            collection_info = api.get_byoc_info(collection_id)
            
            if raw:
                # Print raw JSON
                click.echo(json.dumps(collection_info, indent=2))
            else:
                # Print formatted information
                click.echo(f"Collection ID: {collection_id}")
                click.echo(f"Name: {collection_info.get('name', 'N/A')}")
                click.echo(f"Description: {collection_info.get('description', 'N/A')}")
                click.echo(f"Type: BYOC")
                
                # Print spatial extent if available
                if "extent" in collection_info:
                    extent = collection_info["extent"]
                    if "spatial" in extent and "bbox" in extent["spatial"]:
                        bbox = extent["spatial"]["bbox"]
                        click.echo(f"Spatial Extent (bbox): {bbox}")
                    
                    if "temporal" in extent and "interval" in extent["temporal"]:
                        interval = extent["temporal"]["interval"]
                        click.echo(f"Temporal Extent: {interval}")
                
                # Print band information if available
                if "summaries" in collection_info and "eo:bands" in collection_info["summaries"]:
                    bands = collection_info["summaries"]["eo:bands"]
                    click.echo("\nBands:")
                    for band in bands:
                        name = band.get("name", "N/A")
                        description = band.get("description", "N/A")
                        click.echo(f"  - {name}: {description}")
        except Exception as e:
            click.echo(f"Error getting BYOC information: {e}")
            if debug:
                import traceback
                click.echo(traceback.format_exc())
    else:
        # Use STAC API for all collections (including BYOC)
        stac_id = collection_id
        if is_uuid_flag:
            stac_id = f"byoc-{collection_id}"
        
        click.echo(f"Getting STAC information for collection: {stac_id}")
        try:
            collection_info = api.get_stac_info(stac_id)
            
            if raw:
                # Print raw JSON
                click.echo(json.dumps(collection_info, indent=2))
            else:
                # Print formatted information
                click.echo(f"Collection ID: {stac_id}")
                click.echo(f"Title: {collection_info.get('title', 'N/A')}")
                click.echo(f"Description: {collection_info.get('description', 'N/A')}")
                click.echo(f"Type: {'BYOC' if is_uuid_flag else 'Standard'}")
                
                # Print license information if available
                if "license" in collection_info:
                    click.echo(f"License: {collection_info['license']}")
                
                # Print spatial extent if available
                if "extent" in collection_info:
                    extent = collection_info["extent"]
                    if "spatial" in extent and "bbox" in extent["spatial"]:
                        bbox = extent["spatial"]["bbox"]
                        click.echo(f"Spatial Extent (bbox): {bbox}")
                    
                    if "temporal" in extent and "interval" in extent["temporal"]:
                        interval = extent["temporal"]["interval"]
                        click.echo(f"Temporal Extent: {interval}")
                
                # Print band information if available
                band_info = api.extract_band_info(collection_info)
                if band_info:
                    click.echo("\nBands:")
                    for band_name, band_data in band_info.items():
                        description = band_data.get("description", "N/A")
                        click.echo(f"  - {band_name}: {description}")
                
                # Print data type if available
                data_type = api.get_collection_data_type(collection_info)
                if data_type != "AUTO":
                    click.echo(f"\nDefault Data Type: {data_type}")
                
                # Print nodata value if available
                nodata = api.get_collection_nodata_value(collection_info)
                if nodata is not None:
                    click.echo(f"NoData Value: {nodata}")
        except Exception as e:
            click.echo(f"Error getting collection information: {e}")
            if debug:
                import traceback
                click.echo(traceback.format_exc())


@cli.command()
@click.argument(
    "collection_id",
    required=True
)
@click.argument(
    "feature_id",
    required=True
)
@click.option(
    "--raw/--formatted",
    default=False,
    help="Display raw JSON or formatted output (default: formatted)",
)
@click.pass_context
def feature_info(
    ctx,
    collection_id: str,
    feature_id: str,
    raw: bool,
):
    """Get information about a specific feature in a collection.
    
    COLLECTION_ID can be a standard collection ID (e.g., sentinel-2-l2a) or a BYOC UUID.
    FEATURE_ID is the ID of the specific feature to retrieve information about.
    """
    api, debug = setup_api_from_context(ctx)
    
    # Check if the collection ID is a UUID (BYOC collection)
    is_uuid_flag, valid_uuid = is_uuid(collection_id)
    
    if debug:
        logger.debug(f"Collection ID '{collection_id}' identified as UUID: {is_uuid_flag}")
    
    # Format the collection ID for STAC API
    stac_id = f"byoc-{valid_uuid}" if is_uuid_flag else collection_id
    
    try:
        # Get the feature information
        feature_info = api.get_stac_feature(stac_id, feature_id)
        
        if raw:
            # Print raw JSON
            click.echo(json.dumps(feature_info, indent=2))
        else:
            # Print formatted information
            click.echo(f"Feature ID: {feature_id}")
            click.echo(f"Collection: {stac_id}")
            
            # Print datetime if available
            datetime_str = feature_info.get("properties", {}).get("datetime")
            if datetime_str:
                click.echo(f"DateTime: {datetime_str}")
            
            # Print bbox if available
            bbox = feature_info.get("bbox")
            if bbox:
                click.echo(f"Bounding Box: {bbox}")
            
            # Print cloud cover if available
            cloud_cover = feature_info.get("properties", {}).get("eo:cloud_cover")
            if cloud_cover is not None:
                click.echo(f"Cloud Cover: {cloud_cover}%")
            
            # Print other interesting properties
            props = feature_info.get("properties", {})
            interesting_props = [
                "platform", "instrument", "constellation",
                "gsd", "proj:epsg", "proj:shape",
                "sentinel:data_coverage"
            ]
            
            click.echo("\nProperties:")
            for prop in interesting_props:
                if prop in props:
                    click.echo(f"  {prop}: {props[prop]}")
            
            # Print asset information
            assets = feature_info.get("assets", {})
            if assets:
                click.echo("\nAssets:")
                for name, asset in assets.items():
                    click.echo(f"  {name}:")
                    if "title" in asset:
                        click.echo(f"    Title: {asset['title']}")
                    if "type" in asset:
                        click.echo(f"    Type: {asset['type']}")
                    if "roles" in asset:
                        click.echo(f"    Roles: {', '.join(asset['roles'])}")
    
    except Exception as e:
        click.echo(f"Error getting feature information: {e}")
        if debug:
            import traceback
            click.echo(traceback.format_exc())


if __name__ == "__main__":
    cli() 