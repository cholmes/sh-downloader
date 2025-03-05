"""Command-line interface for Sentinel Hub Downloader."""

import datetime
import logging
import os
import sys
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
    default=True,  # Default to True for now
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
    default=1,
    help="Minimum days between downloaded images (default: 1)",
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
    help="Value to use for nodata pixels in the output GeoTIFFs",
)
@click.option(
    "--scale",
    type=float,
    help="Scale factor to set in the output GeoTIFF (e.g., 0.001)",
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
    time_difference: int,
    filename_template: Optional[str],
    evalscript_file: Optional[str],
    bands: Optional[str],
    auto_discover_bands: bool,
    nodata: Optional[float],
    scale: Optional[float],
):
    """Download a time series from a BYOC (Bring Your Own Collection)."""
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
    width, height = map(int, size.split(","))
    size_tuple = (width, height)
    
    # Set default filename template if not provided
    if not filename_template:
        filename_template = f"BYOC_{byoc_id[:8]}_{{date}}.tiff"
    
    # Load custom evalscript if provided
    evalscript = None
    if evalscript_file:
        try:
            with open(evalscript_file, 'r') as f:
                evalscript = f.read()
            click.echo(f"Loaded custom evalscript from {evalscript_file}")
            # If a custom evalscript is provided, don't auto-discover bands
            auto_discover_bands = False
        except Exception as e:
            click.echo(f"Error loading evalscript file: {e}")
            return
    
    # If bands are specified, create an evalscript for those bands
    specified_bands = None
    if bands:
        specified_bands = [band.strip() for band in bands.split(',')]
        click.echo(f"Using specified bands: {specified_bands}")
        evalscript = api.create_dynamic_evalscript(specified_bands)
        # If bands are specified, don't auto-discover
        auto_discover_bands = False
    
    if auto_discover_bands and not evalscript and not specified_bands:
        click.echo("Auto-discovering bands in BYOC collection...")
    
    # Get available dates
    click.echo(f"Searching for available dates in BYOC collection {byoc_id}...")
    available_dates = api.get_available_dates(
        collection="byoc",
        byoc_id=byoc_id,
        time_interval=(start_date, end_date),
        bbox=bbox_tuple,
        time_difference_days=time_difference
    )
    
    if not available_dates:
        click.echo("No images found matching the criteria.")
        return
    
    click.echo(f"Found {len(available_dates)} dates with images.")
    click.echo(f"Available dates: {[d.strftime('%Y-%m-%d') for d in available_dates]}")
    
    # Download images
    click.echo(f"Downloading {len(available_dates)} images...")
    
    downloaded_files = api.download_byoc_timeseries(
        byoc_id=byoc_id,
        bbox=bbox_tuple,
        time_interval=(start_date, end_date),
        output_dir=output_dir,
        size=size_tuple,
        time_difference_days=time_difference,
        filename_template=filename_template,
        evalscript=evalscript,
        auto_discover_bands=auto_discover_bands,
        specified_bands=specified_bands,
        nodata_value=nodata,
        scale_metadata=scale
    )
    
    if downloaded_files:
        click.echo(f"Successfully downloaded {len(downloaded_files)} images:")
        for file_path in downloaded_files:
            click.echo(f"  - {file_path}")
    else:
        click.echo("No images were downloaded.")


if __name__ == "__main__":
    cli() 