# Sentinel Hub Downloader

A command-line tool to download GeoTIFF data from Sentinel Hub. It is 
currently focused on BYOC data, so far only tested with Planet PV's of
Soil Water Content and Crop Biomass. But it attempts to work against
any BYOC collection, automatically discovering the bands and metadata.

It has only been tested against smaller files, and works just with the
process api. Making it work with larger areas will likely require batch API
and some iterative approach.

## Features

- Search and download BYOC and Sentinel imagery based on time range and area of interest
- Easy-to-use command line interface
- Support for various spatial filters (bounding box, GeoJSON)
- Configurable output directory 
- Automatically discovers bands and metadata for BYOC collections
- Lets users supply custom evalscripts
- Supports various output data types and nodata values

## Installation

Clone the repository and then install the package:

```bash
pip install sh-downloader
```

## Configuration

Before using the tool, you need to configure your Sentinel Hub credentials:

```
shdown config --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
```

This will create a configuration file at `~/.sentinelhub/config.json`.

### Command Help

```
shdown --help

Usage: shdown [OPTIONS] COMMAND [ARGS]...

  Command line tool for downloading data from Sentinel Hub.

Options:
  --debug / --no-debug  Enable debug logging
  --help                Show this message and exit.

Commands:
  byoc        Download a time series of images from a BYOC collection.
  config      Configure Sentinel Hub credentials.
  info        Get information about a collection.
  search      Search for available images.
```

## Usage

### Search for available images

Search for available Sentinel-2 L2A images in a specific area and time range:

```
shdown search --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --time-from 2023-01-01 --time-to 2023-01-31
```

#### Search Command Help

```
shdown search --help

Usage: shdown search [OPTIONS]

  Search for available satellite imagery without downloading.

Options:
  -c, --collection [sentinel-1-grd|sentinel-2-l1c|sentinel-2-l2a|sentinel-3-olci|sentinel-5p-l2|byoc]
                                  Sentinel data collection to search
                                  [required]
  --byoc-id TEXT                  BYOC collection ID (required if collection
                                  is 'byoc')
  -s, --start TEXT                Start date (YYYY-MM-DD). Defaults to 30 days
                                  ago.
  -e, --end TEXT                  End date (YYYY-MM-DD). Defaults to today.
  -b, --bbox TEXT                 Bounding box as
                                  min_lon,min_lat,max_lon,max_lat. Default is
                                  global.
  -m, --max-cloud-cover FLOAT     Maximum cloud cover percentage (0-100). Only
                                  applies to optical sensors.
  -l, --limit INTEGER             Maximum number of results to display.
                                  Default is 10.
  --help                          Show this message and exit.
```

Sample output:

```
Found 8 images:
2023-01-30T10:20:19Z - Cloud cover: 0.00%
2023-01-27T10:30:21Z - Cloud cover: 1.23%
2023-01-25T10:20:19Z - Cloud cover: 0.45%
2023-01-22T10:30:21Z - Cloud cover: 2.67%
2023-01-20T10:20:19Z - Cloud cover: 0.12%
2023-01-17T10:30:21Z - Cloud cover: 3.45%
2023-01-15T10:20:19Z - Cloud cover: 1.78%
2023-01-12T10:30:21Z - Cloud cover: 0.89%
```

### Download a specific image

Download a specific Sentinel-2 L2A image by date:

```
shdown download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --output-dir ./images
```

#### Download Command Help

```
shdown download --help

Usage: shdown download [OPTIONS]

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

### Download BYOC (Bring Your Own Collection) data

Download data from a custom collection:

```
shdown byoc --byoc-id YOUR_BYOC_COLLECTION_ID --time-from 2024-05-01 
```

#### BYOC Command Help

```
 % shdown byoc --help
Usage: shdown byoc [OPTIONS]

  Download images from a BYOC collection.

Options:
  --byoc-id TEXT                  BYOC collection ID  [required]
  -i, --image-id TEXT             Image ID to download
  -s, --start TEXT                Start date (YYYY-MM-DD). Defaults to 30 days
                                  ago.
  -e, --end TEXT                  End date (YYYY-MM-DD). Defaults to today.
  -b, --bbox TEXT                 Bounding box in format 'minx,miny,maxx,maxy'
                                  (WGS84). Optional - if not provided, will
                                  use each image's own bbox.
  -o, --output-dir TEXT           Output directory for downloaded images
  --size TEXT                     Size of the output image as width,height
                                  (default: 512,512)
  -t, --time-difference INTEGER   Minimum days between downloaded images
                                  (default: None - download all images)
  --all-dates / --filter-dates    Download all available dates without
                                  filtering (overrides --time-difference)
  -f, --filename-template TEXT    Template for filenames (default:
                                  'BYOC_{byoc_id}_{date}.tiff')
  --evalscript-file TEXT          Path to a file containing a custom
                                  evalscript
  --bands TEXT                    Comma-separated list of band names to
                                  download (e.g., 'SWC,dataMask')
  --auto-discover-bands / --no-auto-discover-bands
                                   Automatically discover and include all bands
                                  (default: True)
  --nodata FLOAT                  Value to use for nodata pixels in the output
                                  GeoTIFF
  --scale FLOAT                   Value to set as SCALE metadata in the output
                                  GeoTIFF
  --data-type [auto|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float16|float32|float64|cint16|cint32|cfloat32|cfloat64]
                                  Output data type (default: AUTO)
  --help                          Show this message and exit.                           
```


Get metadata about a collection:

```
shdown info --collection --collection byoc --byo
```

#### Info Command Help

```
shdown info --help

Usage: shdown info [OPTIONS]

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
shdown download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --bands B04,B03,B02 --output-dir ./images
```

Sample output:

Downloading image for 2023-01-30...
Using specified bands: ['B04', 'B03', 'B02']
Image saved to ./images/sentinel-2-l2a_2023-01-30.tiff

### Set output data type

Specify the output data type for downloaded images:
    
```
shdown download --collection sentinel-2-l2a --bbox 14.0 45.0 14.5 45.5 --date 2023-01-30 --data-type uint16 --output-dir ./images
```

Sample output:

Downloading image for 2023-01-30...
Using data type: UINT16
Image saved to ./images/sentinel-2-l2a_2023-01-30.tiff

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## TODO's

- Start issue tracker to put these in
- clean up extra gdal files
- option to provide a template for file naming
- show progress as images are downloading / completing
- make it so 'search' works nicer with byoc (same command, detect uuid and use it as id)