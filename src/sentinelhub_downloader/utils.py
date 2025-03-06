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
    """Parse a bounding box string into a tuple.
    
    Can be used as a Click callback or standalone function.
    
    Args:
        ctx: Click context (optional, for Click callback)
        param: Click parameter (optional, for Click callback)
        value: String in format "minx,miny,maxx,maxy" or None
        
    Returns:
        Tuple of (minx, miny, maxx, maxy) or None if value is None
    """
    if value is None:
        return None
        
    try:
        # Handle the case where value is already a tuple
        if isinstance(value, tuple) and len(value) == 4:
            return value
            
        # Parse string format
        parts = value.split(',')
        if len(parts) != 4:
            if ctx:  # If called as a Click callback
                raise click.BadParameter("Bounding box must be in format 'minx,miny,maxx,maxy'")
            else:
                raise ValueError("Bounding box must be in format 'minx,miny,maxx,maxy'")
                
        bbox = tuple(float(p.strip()) for p in parts)
        
        # Validate coordinates
        minx, miny, maxx, maxy = bbox
        if minx >= maxx or miny >= maxy:
            if ctx:
                raise click.BadParameter("Invalid bounding box: min values must be less than max values")
            else:
                raise ValueError("Invalid bounding box: min values must be less than max values")
                
        return bbox
    except ValueError as e:
        if ctx:
            raise click.BadParameter(f"Invalid bounding box format: {str(e)}")
        else:
            raise ValueError(f"Invalid bounding box format: {str(e)}")


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