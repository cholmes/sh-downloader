# Sentinel Hub Downloader

A command-line tool to download GeoTIFF data from Sentinel Hub.

## Features

- Search and download Sentinel imagery based on time range and area of interest
- Access to different Sentinel collections (Sentinel-1, Sentinel-2, etc.)
- Easy-to-use command line interface
- Support for various spatial filters (bounding box, GeoJSON)
- Configurable output directory and file naming

## Installation

```bash
pip install .
```

## Authentication

Before using this tool, you need to obtain Sentinel Hub credentials. Visit [Sentinel Hub Dashboard](https://apps.sentinel-hub.com/dashboard/) to create an account and obtain your credentials.

Set up your credentials:

```bash
shdownload configure
```

## Usage

Basic usage:

```bash
shdownload download --collection sentinel-2-l2a
```

Download specific area:

```bash
shdownload download --collection sentinel-2-l2a --bbox 10,10,20,20
```

Specify time range and area:

```bash
shdownload download --collection sentinel-2-l2a --start 2023-01-01 --end 2023-01-31 --bbox 13.0,45.0,14.0,46.0
```

For more options:

```bash
shdownload download --help
```

## License

MIT
