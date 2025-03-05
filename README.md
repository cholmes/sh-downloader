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

## Configuration

Before using the tool, you need to configure your Sentinel Hub credentials:

```
shdownload config --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
```

This will create a configuration file at `~/.sentinelhub/config.json`.

### Command Help

```
shdownload --help

Usage: shdownload [OPTIONS] COMMAND [ARGS]...

  Command line tool for downloading data from Sentinel Hub.

Options:
  --debug / --no-debug  Enable debug logging
  --help                Show this message and exit.

Commands:
  byoc        Download a time series of images from a BYOC collection.
  config      Configure Sentinel Hub credentials.
  download    Download a specific image from Sentinel Hub.
  info        Get information about a collection.
  search      Search for available images.
  timeseries  Download a time series of images.
```

## Usage

### Search for available images

Search for available Sentinel-2 L2A images in a specific area and time range:

```
shdownload search --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --time-from 2023-01-01 --time-to 2023-01-31
```

#### Search Command Help

```
shdownload search --help

Usage: shdownload search [OPTIONS]

  Search for available images.

Options:
  --collection TEXT              Collection name (e.g., sentinel-2-l2a)
                                 [required]
  --bbox FLOAT...                Bounding box as min_lon min_lat max_lon
                                 max_lat  [required]
  --time-from TEXT               Start date (YYYY-MM-DD)  [required]
  --time-to TEXT                 End date (YYYY-MM-DD)  [required]
  --byoc-id TEXT                 BYOC collection ID (required if collection is
                                 'byoc')
  --time-difference-days INTEGER
                                 Minimum days between images
  --help                         Show this message and exit.
```

Sample output:

Found 8 images:
2023-01-30T10:20:19Z - Cloud cover: 0.00%
2023-01-27T10:30:21Z - Cloud cover: 1.23%
2023-01-25T10:20:19Z - Cloud cover: 0.45%
2023-01-22T10:30:21Z - Cloud cover: 2.67%
2023-01-20T10:20:19Z - Cloud cover: 0.12%
2023-01-17T10:30:21Z - Cloud cover: 3.45%
2023-01-15T10:20:19Z - Cloud cover: 1.78%
2023-01-12T10:30:21Z - Cloud cover: 0.89%

### Download a specific image

Download a specific Sentinel-2 L2A image by date:

```
shdownload download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --output-dir ./images
```

#### Download Command Help

```
shdownload download --help

Usage: shdownload download [OPTIONS]

  Download a specific image from Sentinel Hub.

Options:
  --collection TEXT         Collection name (e.g., sentinel-2-l2a)  [required]
  --bbox FLOAT...           Bounding box as min_lon min_lat max_lon max_lat
                            [required]
  --date TEXT               Date to download (YYYY-MM-DD)
  --image-id TEXT           Specific image ID to download
  --output-dir TEXT         Directory to save the downloaded image
  --filename TEXT           Filename for the downloaded image
  --byoc-id TEXT            BYOC collection ID (required if collection is
                            'byoc')
  --size INTEGER...         Size of the output image as width height
  --bands TEXT              Comma-separated list of bands to include
  --data-type TEXT          Output data type (AUTO, UINT8, UINT16, FLOAT32)
  --nodata FLOAT            Value to use for nodata pixels
  --scale FLOAT             Scale factor for the output image
  --help                    Show this message and exit.
```

Sample output:

Downloading image for 2023-01-30...
Image saved to ./images/sentinel-2-l2a_2023-01-30.tiff

### Download a time series of images

Download all available Sentinel-2 L2A images in a specific area and time range:

```
shdownload timeseries --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --time-from 2023-01-01 --time-to 2023-01-31 --output-dir ./timeseries
```

#### Timeseries Command Help

```
shdownload timeseries --help

Usage: shdownload timeseries [OPTIONS]

  Download a time series of images.

Options:
  --collection TEXT              Collection name (e.g., sentinel-2-l2a)
                                   [required]
  --bbox FLOAT...                Bounding box as min_lon min_lat max_lon
                                   max_lat  [required]
  --time-from TEXT               Start date (YYYY-MM-DD)  [required]
  --time-to TEXT                 End date (YYYY-MM-DD)  [required]
  --output-dir TEXT              Directory to save the downloaded images
  --filename-template TEXT       Template for filenames
  --time-difference-days INTEGER
                                   Minimum days between images
  --size INTEGER...              Size of the output images as width height
  --bands TEXT                   Comma-separated list of bands to include
  --data-type TEXT               Output data type (AUTO, UINT8, UINT16,
                                   FLOAT32)
  --nodata FLOAT                 Value to use for nodata pixels
  --scale FLOAT                  Scale factor for the output images
  --help                         Show this message and exit.
```

Sample output:

Found 8 dates with images
Downloading image for 2023-01-12...
Downloading image for 2023-01-15...
Downloading image for 2023-01-17...
Downloading image for 2023-01-20...
Downloading image for 2023-01-22...
Downloading image for 2023-01-25...
Downloading image for 2023-01-27...
Downloading image for 2023-01-30...
Downloaded 8 images to ./timeseries

### Download BYOC (Bring Your Own Collection) data

Download data from a custom collection:

```
shdownload byoc --byoc-id YOUR_BYOC_ID --bbox 14.0 45.0 14.5 45.5 --time-from 2023-01-01 --time-to 2023-01-31 --output-dir ./byoc
```

#### BYOC Command Help

```
shdownload byoc --help

Usage: shdownload byoc [OPTIONS]

  Download a time series of images from a BYOC collection.

Options:
  --byoc-id TEXT                 BYOC collection ID  [required]
  --bbox FLOAT...                Bounding box as min_lon min_lat max_lon
                                   max_lat  [required]
  --time-from TEXT               Start date (YYYY-MM-DD)  [required]
  --time-to TEXT                 End date (YYYY-MM-DD)  [required]
  --output-dir TEXT              Directory to save the downloaded images
  --filename-template TEXT       Template for filenames
  --time-difference-days INTEGER
                                   Minimum days between images
  --size INTEGER...              Size of the output images as width height
  --bands TEXT                   Comma-separated list of bands to include
  --auto-discover-bands / --no-auto-discover-bands
                                   Automatically discover and include all bands
  --data-type TEXT               Output data type (AUTO, UINT8, UINT16,
                                   FLOAT32)
  --nodata FLOAT                 Value to use for nodata pixels
  --scale FLOAT                  Scale factor for the output images
  --help                         Show this message and exit.
```

Sample output:

Found 5 dates with images
Using specified bands: ['SWC', 'QF', 'SWC_MaskedPixels']
Downloading image for 2023-01-15...
Downloading image for 2023-01-20...
Downloading image for 2023-01-25...
Downloading image for 2023-01-27...
Downloading image for 2023-01-30...
Downloaded 5 images to ./byoc

### Get information about a collection

Get metadata about a collection:

```
shdownload info --collection sentinel-2-l2a
```

#### Info Command Help

```
shdownload info --help

Usage: shdownload info [OPTIONS]

  Get information about a collection.

Options:
  --collection TEXT  Collection name (e.g., sentinel-2-l2a)  [required]
  --byoc-id TEXT     BYOC collection ID (required if collection is 'byoc')
  --help             Show this message and exit.
```

Sample output:

Collection: sentinel-2-l2a
Description: Sentinel-2 L2A imagery
Available bands:
  - B01: Coastal aerosol (443 nm)
  - B02: Blue (490 nm)
  - B03: Green (560 nm)
  - B04: Red (665 nm)
  - B05: Vegetation Red Edge (705 nm)
  - B06: Vegetation Red Edge (740 nm)
  - B07: Vegetation Red Edge (783 nm)
  - B08: NIR (842 nm)
  - B8A: Vegetation Red Edge (865 nm)
  - B09: Water vapour (945 nm)
  - B11: SWIR (1610 nm)
  - B12: SWIR (2190 nm)
  - SCL: Scene classification

## Advanced Usage

### Specify bands to download

Download only specific bands from a collection:

```
shdownload download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --bands B04,B03,B02 --output-dir ./images
```

Sample output:

Downloading image for 2023-01-30...
Using specified bands: ['B04', 'B03', 'B02']
Image saved to ./images/sentinel-2-l2a_2023-01-30.tiff

### Set output data type

Specify the output data type for downloaded images:
    
```
shdownload download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --data-type uint16 --output-dir ./images
```

Sample output:

Downloading image for 2023-01-30...
Using data type: UINT16
Image saved to ./images/sentinel-2-l2a_2023-01-30.tiff

## License

This project is licensed under the MIT License - see the LICENSE file for details.