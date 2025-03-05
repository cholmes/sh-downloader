# Sentinel Hub Downloader

A command-line tool to download GeoTIFF data from Sentinel Hub.

## Features

- Search and download Sentinel imagery based on time range and area of interest
- Access to different Sentinel collections (Sentinel-1, Sentinel-2, etc.)
- Easy-to-use command line interface
- Support for various spatial filters (bounding box, GeoJSON)
- Configurable output directory and file naming

## Dependencies

This package requires the following dependencies:

- Python 3.8+
- sentinelhub
- rasterio
- click
- tqdm
- shapely

For Cloud Optimized GeoTIFF (COG) creation and metadata handling:
- GDAL (optional but recommended)

GDAL is used for setting scale factors and creating optimized GeoTIFFs. If GDAL is not available, the tool will still work but with reduced functionality.

## Installation

```bash
pip install .
```