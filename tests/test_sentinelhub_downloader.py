"""Tests for the Sentinel Hub Downloader package."""

import datetime
import os
from unittest import mock

import pytest

from sentinelhub_downloader import config, utils
from sentinelhub_downloader.api import SentinelHubAPI


def test_parse_date():
    """Test the date parsing function."""
    date = utils.parse_date("2023-04-15")
    assert date.year == 2023
    assert date.month == 4
    assert date.day == 15


def test_parse_bbox():
    """Test the bounding box parsing function."""
    bbox = utils.parse_bbox("10.0,45.0,12.0,47.0")
    assert bbox == (10.0, 45.0, 12.0, 47.0)


def test_get_date_range():
    """Test the date range function."""
    # Test with specific dates
    start, end = utils.get_date_range("2023-01-01", "2023-01-31")
    assert start.date() == datetime.date(2023, 1, 1)
    assert end.date() == datetime.date(2023, 1, 31)
    
    # Test with default end date (today)
    start, end = utils.get_date_range("2023-01-01", None)
    assert start.date() == datetime.date(2023, 1, 1)
    assert end.date() <= datetime.datetime.now().date()
    
    # Test with default start date (30 days before end)
    start, end = utils.get_date_range(None, "2023-01-31")
    assert end.date() == datetime.date(2023, 1, 31)
    assert (end.date() - start.date()).days == 30


@mock.patch("sentinelhub_downloader.config.Config.load_config")
def test_config_initialization(mock_load_config):
    """Test the configuration initialization."""
    mock_load_config.return_value = {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "instance_id": "test_instance",
        "output_dir": "/tmp/downloads",
    }
    
    cfg = config.Config()
    assert cfg.get("client_id") == "test_id"
    assert cfg.get("client_secret") == "test_secret"
    assert cfg.is_configured() is True


@mock.patch("sentinelhub_downloader.api.SentinelHubCatalog")
@mock.patch("sentinelhub_downloader.api.SentinelHubDownloadClient")
@mock.patch("sentinelhub_downloader.api.SHConfig")
@mock.patch("sentinelhub_downloader.config.Config")
def test_api_initialization(mock_config, mock_sh_config, mock_download_client, mock_catalog):
    """Test the API client initialization."""
    mock_config.return_value.get.side_effect = lambda key, default=None: {
        "client_id": "test_id",
        "client_secret": "test_secret", 
        "instance_id": "test_instance"
    }.get(key, default)
    
    api = SentinelHubAPI()
    
    # Check that the catalog and download client were initialized
    assert mock_catalog.called
    assert mock_download_client.called 