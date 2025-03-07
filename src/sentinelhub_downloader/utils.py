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


def parse_bbox(ctx=None, param=None, value=None):
    """Parse a bounding box string into a tuple of floats.
    
    Can be used as a Click callback or as a standalone function.
    
    Args:
        ctx: Click context (optional)
        param: Click parameter (optional)
        value: Bounding box string or value to parse
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) or None if value is None
    """
    # If called directly with a single argument, assume it's the value
    if ctx is not None and param is None and value is None:
        value = ctx
        ctx = None
    
    if value is None:
        return None
    
    try:
        # If value is already a tuple, just validate it
        if isinstance(value, tuple) and len(value) == 4:
            return value
            
        # Split by comma and convert to float
        parts = [float(x.strip()) for x in value.split(',')]
        
        # Ensure we have exactly 4 values
        if len(parts) != 4:
            raise ValueError("Bounding box must have exactly 4 values")
        
        # Return as tuple (min_lon, min_lat, max_lon, max_lat)
        return tuple(parts)
    except Exception as e:
        if ctx:  # If called as a Click callback
            raise click.BadParameter(f"Invalid bounding box format: {e}")
        else:
            raise ValueError(f"Invalid bounding box format: {e}")


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


def format_time_interval(time_interval):
    """Format a time interval for Sentinel Hub API requests.
    
    Args:
        time_interval: Tuple of (start_date, end_date) as datetime objects
        
    Returns:
        Tuple of (start_str, end_str) formatted as ISO 8601 strings
    """
    start_date, end_date = time_interval
    time_from = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_to = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    return time_from, time_to 