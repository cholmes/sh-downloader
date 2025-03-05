"""Utility functions for Sentinel Hub Downloader."""

import datetime
from typing import List, Optional, Tuple, Union

import click
from shapely.geometry import box


def parse_date(date_str: str) -> datetime.datetime:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD)
        
    Returns:
        datetime object
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Parse a bounding box string into a tuple.
    
    Args:
        bbox_str: Bounding box string as "min_lon,min_lat,max_lon,max_lat"
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("Bounding box must have 4 values")
        
        min_lon, min_lat, max_lon, max_lat = parts
        
        # Validate the bounding box
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bounding box: min must be less than max")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
            
        return min_lon, min_lat, max_lon, max_lat
    except ValueError as e:
        raise click.BadParameter(f"Invalid bounding box format: {e}")


def get_date_range(
    start: Optional[str], end: Optional[str]
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Get a date range from start and end strings.
    
    If start is not provided, defaults to 30 days ago.
    If end is not provided, defaults to today.
    
    Args:
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    if end:
        end_date = parse_date(end)
    else:
        end_date = datetime.datetime.now()
    
    if start:
        start_date = parse_date(start)
    else:
        start_date = end_date - datetime.timedelta(days=30)
    
    # Ensure start is before end
    if start_date > end_date:
        raise click.BadParameter("Start date must be before end date")
    
    return start_date, end_date 