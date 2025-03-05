"""Command-line interface for Sentinel Hub Downloader."""

import datetime
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple

import click
from tqdm import tqdm

from sentinelhub_downloader.api import SentinelHubAPI
from sentinelhub_downloader.config import Config
from sentinelhub_downloader.utils import get_date_range, parse_bbox

# Set up logging
logger = logging.getLogger("sentinelhub_downloader")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=True, 
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
@click.option(
    "--collection",
    "-c",
    type=click.Choice(
        [
            "sentinel-1-grd",
            "sentinel-2-l1c",
            "sentinel-2-l2a",
            "sentinel-3-olci",
            "sentinel-5p-l2",
            "byoc",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Sentinel data collection to download",
)
@click.option(
    "--byoc-id",
    help="BYOC collection ID (required if collection is 'byoc')",
)
@click.option(
    "--image-id",
    "-i",
    help="Image ID to download",
)
@click.option(
    "--date",
    "-d",
    help="Specific date to download (can be used instead of image-id for BYOC collections)",
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
)
@click.option(
    "--max-cloud-cover",
    "-m",
    type=float,
    help="Maximum cloud cover percentage (0-100). Only applies to optical sensors.",
)
@click.option(
    "--output-dir",
    "-o",
    help="Directory to save downloaded files. Defaults to ./downloads",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Maximum number of images to download. Default is 10.",
)
@click.pass_context
def download(
    ctx,
    collection: str,
    byoc_id: Optional[str],
    image_id: Optional[str],
    date: Optional[str],
    start: Optional[str],
    end: Optional[str],
    bbox: Optional[str],
    max_cloud_cover: Optional[float],
    output_dir: Optional[str],
    limit: int,
):
    """Download satellite imagery from Sentinel Hub."""
    debug = ctx.obj.get("DEBUG", False)
    config = Config()
    
    # Check if configured
    if not config.is_configured():
        click.echo("Sentinel Hub Downloader is not configured. Running configuration wizard...")
        config.configure_wizard()
    
    # Check if BYOC ID is provided for BYOC collection
    if collection.lower() == "byoc" and not byoc_id:
        click.echo("Error: BYOC collection ID (--byoc-id) is required when using BYOC collection")
        return
    
    # Set up API client with debug flag
    api = SentinelHubAPI(config, debug=debug)
    
    # Set default output directory if not provided
    if not output_dir:
        output_dir = config.get("output_dir")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If image_id is provided, download that specific image
    if image_id:
        if not bbox:
            click.echo("Error: Bounding box (--bbox) is required when downloading a specific image")
            return
        
        bbox_tuple = parse_bbox(bbox)
        click.echo(f"Downloading image {image_id} from {collection}...")
        
        try:
            output_path = api.download_image(
                image_id=image_id,
                collection=collection,
                byoc_id=byoc_id,
                bbox=bbox_tuple,
                output_dir=output_dir
            )
            click.echo(f"Image downloaded to: {output_path}")
        except Exception as e:
            click.echo(f"Error downloading image: {e}")
        
        return
    
    # If date is provided, download image for that specific date
    if date:
        if not bbox:
            click.echo("Error: Bounding box (--bbox) is required when downloading by date")
            return
        
        bbox_tuple = parse_bbox(bbox)
        click.echo(f"Downloading image for date {date} from {collection}...")
        
        try:
            output_path = api.download_image(
                image_id=None,
                collection=collection,
                byoc_id=byoc_id,
                bbox=bbox_tuple,
                output_dir=output_dir,
                date=date
            )
            click.echo(f"Image downloaded to: {output_path}")
        except Exception as e:
            click.echo(f"Error downloading image: {e}")
        
        return
    
    # Otherwise, search for images and download them
    # Parse date range
    start_date, end_date = get_date_range(start, end)
    click.echo(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Parse bounding box if provided
    bbox_tuple = None
    if bbox:
        bbox_tuple = parse_bbox(bbox)
        click.echo(f"Bounding box: {bbox_tuple}")
    else:
        click.echo("Bounding box: Global")
    
    # Search for images
    click.echo(f"Searching for {collection} images...")
    search_results = api.search_catalog(
        collection=collection,
        time_interval=(start_date, end_date),
        bbox=bbox_tuple,
        max_cloud_cover=max_cloud_cover,
        byoc_id=byoc_id,
    )
    
    if not search_results:
        click.echo("No images found matching the criteria.")
        return
    
    # Limit the number of images to download
    images_to_download = search_results[:limit]
    click.echo(f"Found {len(search_results)} images. Downloading first {len(images_to_download)}...")
    
    # Download each image
    for i, result in enumerate(images_to_download):
        image_id = result["id"]
        date = result.get("properties", {}).get("datetime", "unknown_date")
        
        click.echo(f"[{i+1}/{len(images_to_download)}] Downloading image {image_id} from {date}...")
        
        try:
            if not bbox_tuple:
                click.echo("  Skipping: Bounding box is required for download")
                continue
            
            output_path = api.download_image(
                image_id=image_id,
                collection=collection,
                byoc_id=byoc_id,
                bbox=bbox_tuple,
                output_dir=output_dir
            )
            click.echo(f"  Downloaded to: {output_path}")
        except Exception as e:
            click.echo(f"  Error: {e}")


@cli.command()
@click.option(
    "--collection",
    "-c",
    type=click.Choice(
        [
            "sentinel-1-grd",
            "sentinel-2-l1c",
            "sentinel-2-l2a",
            "sentinel-3-olci",
            "sentinel-5p-l2",
            "byoc",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Sentinel data collection to search",
)
@click.option(
    "--byoc-id",
    help="BYOC collection ID (required if collection is 'byoc')",
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
    collection: str,
    byoc_id: Optional[str],
    start: Optional[str],
    end: Optional[str],
    bbox: Optional[str],
    max_cloud_cover: Optional[float],
    limit: int,
):
    """Search for available satellite imagery without downloading."""
    debug = ctx.obj.get("DEBUG", False)
    config = Config()
    
    # Check if configured
    if not config.is_configured():
        click.echo("Sentinel Hub Downloader is not configured. Running configuration wizard...")
        config.configure_wizard()
    
    # Check if BYOC ID is provided for BYOC collection
    if collection.lower() == "byoc" and not byoc_id:
        click.echo("Error: BYOC collection ID (--byoc-id) is required when using BYOC collection")
        return
    
    # Set up API client with debug flag
    api = SentinelHubAPI(config, debug=debug)
    
    # Parse date range
    start_date, end_date = get_date_range(start, end)
    click.echo(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Parse bounding box if provided
    bbox_tuple = None
    if bbox:
        bbox_tuple = parse_bbox(bbox)
        click.echo(f"Bounding box: {bbox_tuple}")
    else:
        click.echo("Bounding box: Global")
    
    # Search for images
    click.echo(f"Searching for {collection} images...")
    search_results = api.search_catalog(
        collection=collection,
        time_interval=(start_date, end_date),
        bbox=bbox_tuple,
        max_cloud_cover=max_cloud_cover,
        byoc_id=byoc_id,
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
@click.option(
    "--byoc-id",
    required=True,
    help="BYOC collection ID",
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
    required=True,
    help="Bounding box as min_lon,min_lat,max_lon,max_lat",
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
        "AUTO", "auto",
        "INT8", "int8",
        "INT16", "int16",
        "INT32", "int32",
        "INT64", "int64",
        "UINT8", "uint8",
        "UINT16", "uint16",
        "UINT32", "uint32",
        "UINT64", "uint64",
        "FLOAT16", "float16",
        "FLOAT32", "float32",
        "FLOAT64", "float64",
        "CINT16", "cint16",
        "CINT32", "cint32",
        "CFLOAT32", "cfloat32",
        "CFLOAT64", "cfloat64"
    ], case_sensitive=False),
    default="AUTO",
    help="Output data type (default: AUTO)",
)
@click.pass_context
def byoc(
    ctx,
    byoc_id: str,
    start: Optional[str],
    end: Optional[str],
    bbox: str,
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
    """Download images from a BYOC collection."""
    debug = ctx.obj.get("DEBUG", False)
    config = Config()
    
    # Check if configured
    if not config.is_configured():
        click.echo("Sentinel Hub Downloader is not configured. Running configuration wizard...")
        config.configure_wizard()
    
    # Set up API client with debug flag
    api = SentinelHubAPI(config, debug=debug)
    
    # Parse date range
    start_date, end_date = get_date_range(start, end)
    click.echo(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Parse bounding box
    bbox_tuple = parse_bbox(bbox)
    click.echo(f"Bounding box: {bbox_tuple}")
    
    # Parse size
    size_tuple = tuple(map(int, size.split(",")))
    click.echo(f"Output size: {size_tuple}")
    
    # Determine time difference
    effective_time_difference = None
    if not all_dates:
        effective_time_difference = time_difference
    
    # Get available dates
    available_dates = api.get_available_dates(
        collection="byoc",
        byoc_id=byoc_id,
        time_interval=(start_date, end_date),
        bbox=bbox_tuple,
        time_difference_days=effective_time_difference
    )
    
    if not available_dates:
        click.echo("No images found for the specified criteria")
        return
    
    click.echo(f"Found {len(available_dates)} images")
    
    # Parse bands if provided
    specified_bands = None
    if bands:
        specified_bands = [b.strip() for b in bands.split(",")]
        click.echo(f"Using specified bands: {specified_bands}")
        # Disable auto-discovery if bands are specified
        auto_discover_bands = False
    
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
    
    # If no evalscript is provided but bands are specified, create a dynamic evalscript
    if not evalscript and specified_bands:
        evalscript = api.create_dynamic_evalscript(specified_bands, data_type=data_type)
        click.echo("Created dynamic evalscript for specified bands")
        if debug:
            click.echo(f"Evalscript:\n{evalscript}")
    
    # Download images
    click.echo(f"Downloading {len(available_dates)} images...")
    
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
        click.echo(f"Successfully downloaded {len(downloaded_files)} images:")
        for file_path in downloaded_files:
            click.echo(f"  - {file_path}")
    else:
        click.echo("No images were downloaded.")


@cli.command()
@click.option(
    "--collection-id",
    required=True,
    help="Collection ID to get information about (can be a standard collection ID or BYOC ID)",
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
    """Get information about a collection (standard or BYOC)."""
    debug = ctx.obj.get("DEBUG", False)
    config = Config()
    
    # Check if configured
    if not config.is_configured():
        click.echo("Sentinel Hub Downloader is not configured. Running configuration wizard...")
        config.configure_wizard()
    
    # Set up API client with debug flag
    api = SentinelHubAPI(config, debug=debug)
    
    # Check if the collection ID is a UUID (BYOC collection)
    is_uuid = False
    try:
        import uuid
        uuid.UUID(collection_id)
        is_uuid = True
    except (ValueError, AttributeError):
        pass
    
    # Determine which API to use
    if is_uuid and byoc_api:
        # Use BYOC API for BYOC collections if requested
        click.echo(f"Getting BYOC information for collection: {collection_id}")
        try:
            collection_info = api.get_byoc_info(collection_id)
            
            if raw:
                # Display raw JSON
                click.echo(json.dumps(collection_info, indent=2))
            else:
                # Display formatted information
                if "data" in collection_info:
                    # Use the data field if it exists
                    collection_data = collection_info["data"]
                else:
                    # Otherwise use the top level
                    collection_data = collection_info
                
                click.echo("\nCollection Information:")
                click.echo(f"  ID: {collection_data.get('id', 'N/A')}")
                click.echo(f"  Name: {collection_data.get('name', 'N/A')}")
                click.echo(f"  Created: {collection_data.get('created', 'N/A')}")
                
                # Display bands information if available
                if "additionalData" in collection_data and "bands" in collection_data["additionalData"]:
                    bands = collection_data["additionalData"]["bands"]
                    click.echo("\nBands:")
                    for band_name, band_info in bands.items():
                        click.echo(f"  - {band_name}:")
                        click.echo(f"      Sample Format: {band_info.get('sampleFormat', 'N/A')}")
                        click.echo(f"      Bit Depth: {band_info.get('bitDepth', 'N/A')}")
                        click.echo(f"      No Data Value: {band_info.get('noData', 'N/A')}")
                        click.echo(f"      Band Index: {band_info.get('bandIndex', 'N/A')}")
                        click.echo(f"      Source: {band_info.get('source', 'N/A')}")
                
                # Display temporal extent if available
                if "additionalData" in collection_data:
                    additional_data = collection_data["additionalData"]
                    
                    # Check for temporal information
                    if "fromSensingTime" in additional_data and "toSensingTime" in additional_data:
                        click.echo("\nTemporal Extent:")
                        click.echo(f"  Start: {additional_data.get('fromSensingTime', 'N/A')}")
                        click.echo(f"  End: {additional_data.get('toSensingTime', 'N/A')}")
                    
                    # Display spatial extent if available
                    if "extent" in additional_data:
                        extent = additional_data["extent"]
                        if "coordinates" in extent:
                            coords = extent["coordinates"][0]
                            min_lon = min(c[0] for c in coords)
                            min_lat = min(c[1] for c in coords)
                            max_lon = max(c[0] for c in coords)
                            max_lat = max(c[1] for c in coords)
                            
                            click.echo("\nSpatial Extent (BBOX):")
                            click.echo(f"  {min_lon},{min_lat},{max_lon},{max_lat}")
                    
                    # Display other metadata
                    click.echo("\nAdditional Metadata:")
                    for key, value in additional_data.items():
                        if key not in ["bands", "extent", "fromSensingTime", "toSensingTime"]:
                            click.echo(f"  {key}: {value}")
        
        except Exception as e:
            click.echo(f"Error getting BYOC information: {e}")
            if debug:
                import traceback
                click.echo(traceback.format_exc())
    
    else:
        # Use STAC API for all collections (default)
        # For BYOC collections, prepend "byoc-" to the UUID
        stac_collection_id = f"byoc-{collection_id}" if is_uuid else collection_id
        
        click.echo(f"Getting STAC information for collection: {stac_collection_id}")
        
        try:
            stac_info = api.get_stac_info(stac_collection_id)
            
            if raw:
                # Display raw JSON
                click.echo(json.dumps(stac_info, indent=2))
            else:
                # Display formatted information
                click.echo("\nSTAC Collection Information:")
                click.echo(f"  ID: {stac_info.get('id', 'N/A')}")
                click.echo(f"  Title: {stac_info.get('title', 'N/A')}")
                click.echo(f"  Description: {stac_info.get('description', 'N/A')}")
                click.echo(f"  License: {stac_info.get('license', 'N/A')}")
                click.echo(f"  Version: {stac_info.get('version', 'N/A')}")
                click.echo(f"  STAC Version: {stac_info.get('stac_version', 'N/A')}")
                
                # Display spatial extent
                if "extent" in stac_info:
                    extent = stac_info["extent"]
                    if "spatial" in extent:
                        spatial = extent["spatial"]
                        if "bbox" in spatial and spatial["bbox"]:
                            bbox = spatial["bbox"][0]
                            click.echo("\nSpatial Extent (BBOX):")
                            click.echo(f"  {bbox}")
                    
                    if "temporal" in extent:
                        temporal = extent["temporal"]
                        if "interval" in temporal and temporal["interval"]:
                            interval = temporal["interval"][0]
                            click.echo("\nTemporal Extent:")
                            click.echo(f"  Start: {interval[0] if interval[0] else 'N/A'}")
                            click.echo(f"  End: {interval[1] if interval[1] else 'N/A'}")
                
                # Display providers
                if "providers" in stac_info and stac_info["providers"]:
                    click.echo("\nProviders:")
                    for provider in stac_info["providers"]:
                        click.echo(f"  - {provider.get('name', 'N/A')}: {provider.get('description', 'N/A')}")
                        if "roles" in provider:
                            click.echo(f"    Roles: {', '.join(provider['roles'])}")
                        if "url" in provider:
                            click.echo(f"    URL: {provider['url']}")
                
                # Display summaries
                if "summaries" in stac_info:
                    summaries = stac_info["summaries"]
                    click.echo("\nSummaries:")
                    for key, value in summaries.items():
                        if isinstance(value, list) and len(value) > 10:
                            click.echo(f"  {key}: {value[:5]} ... (and {len(value)-5} more)")
                        else:
                            click.echo(f"  {key}: {value}")
                
                # Display assets
                if "assets" in stac_info:
                    assets = stac_info["assets"]
                    click.echo("\nAssets:")
                    for asset_name, asset_info in assets.items():
                        click.echo(f"  - {asset_name}:")
                        click.echo(f"      Title: {asset_info.get('title', 'N/A')}")
                        click.echo(f"      Type: {asset_info.get('type', 'N/A')}")
                        if "roles" in asset_info:
                            click.echo(f"      Roles: {', '.join(asset_info['roles'])}")
                
                # Display item assets
                if "item_assets" in stac_info:
                    item_assets = stac_info["item_assets"]
                    click.echo("\nItem Assets (Bands):")
                    for asset_name, asset_info in item_assets.items():
                        click.echo(f"  - {asset_name}:")
                        click.echo(f"      Title: {asset_info.get('title', 'N/A')}")
                        click.echo(f"      Type: {asset_info.get('type', 'N/A')}")
                        if "roles" in asset_info:
                            click.echo(f"      Roles: {', '.join(asset_info['roles'])}")
                        
                        # Display raster band information if available
                        if "raster:bands" in asset_info:
                            bands = asset_info["raster:bands"]
                            for i, band in enumerate(bands):
                                click.echo(f"      Band {i+1}:")
                                if "data_type" in band:
                                    click.echo(f"        Data Type: {band['data_type']}")
                                if "nodata" in band:
                                    click.echo(f"        No Data Value: {band['nodata']}")
                                if "unit" in band:
                                    click.echo(f"        Unit: {band['unit']}")
                                if "scale" in band:
                                    click.echo(f"        Scale: {band['scale']}")
                                if "offset" in band:
                                    click.echo(f"        Offset: {band['offset']}")
        
        except Exception as e:
            click.echo(f"Error getting STAC information: {e}")
            if debug:
                import traceback
                click.echo(traceback.format_exc())


if __name__ == "__main__":
    cli() 