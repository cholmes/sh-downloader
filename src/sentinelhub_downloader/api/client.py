"""Base client for Sentinel Hub API."""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Add sentinelhub-py imports
from sentinelhub import SHConfig

from sentinelhub_downloader.config import Config

logger = logging.getLogger("sentinelhub_downloader")

class SentinelHubClient:
    """Base client for interacting with Sentinel Hub APIs."""
    
    def __init__(self, config: Config, debug: bool = False):
        """Initialize the Sentinel Hub client."""
        self.config = config
        self.debug = debug
        
        self._session = None
        self._token = None
        self._token_expiry = None
        
        # Set up logging
        log_level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(log_level)
        
        # Add console handler if not already added
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Base URLs for different APIs
        self.auth_url = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
        self.processing_api_url = "https://services.sentinel-hub.com/api/v1/process"
        self.catalog_url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0"
        self.byoc_url = "https://services.sentinel-hub.com/api/v1/byoc"
        
        # Setup SHConfig for sentinelhub-py
        self.sh_config = SHConfig()
        self.sh_config.sh_client_id = self.config.get("client_id")
        self.sh_config.sh_client_secret = self.config.get("client_secret")
        self.sh_config.sh_base_url = "https://services.sentinel-hub.com"
        
        # Initialize token
        self._get_token()
        
    def _get_token(self):
        """Get an OAuth token for the Sentinel Hub API."""
        # Check if token exists and is not expired
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            logger.debug("Using existing token")
            return self._token
        
        logger.debug("Getting new OAuth token")
        
        # Get credentials
        client_id = self.config.get("client_id")
        client_secret = self.config.get("client_secret")
        
        if not client_id or not client_secret:
            raise ValueError("Client ID and Client Secret must be configured")
        
        # Create OAuth client
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)
        
        # Get token
        try:
            token = oauth.fetch_token(
                token_url=self.auth_url,
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Set token and expiry time
            self._token = token["access_token"]
            self._token_expiry = datetime.now() + timedelta(seconds=token["expires_in"] - 60)
            
            # Create session with token
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json"
            })
            
            logger.debug("Successfully obtained OAuth token")
            return self._token
            
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            raise
    
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an authenticated request to the Sentinel Hub API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments to pass to requests
        
        Returns:
            Response object
        """
        # Get token if needed
        self._get_token()
        
        # Make request
        response = self._session.request(method, url, **kwargs)
        
        if self.debug:
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Request method: {method}")
            if kwargs.get("json"):
                logger.debug(f"Request payload: {json.dumps(kwargs.get('json'), indent=2)}")
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            # Try to log response as JSON if possible
            try:
                logger.debug(f"Response body: {json.dumps(response.json(), indent=2)}")
            except:
                logger.debug(f"Response body: {response.text[:1000]}...")
        
        # Check for errors
        response.raise_for_status()
        
        return response
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request to the Sentinel Hub API."""
        return self._request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request to the Sentinel Hub API."""
        return self._request("POST", url, **kwargs)
    
    def request_with_retry(self, method: str, url: str, max_retries: int = 3, **kwargs) -> requests.Response:
        """Make a request with retry logic.
        
        Args:
            method: HTTP method
            url: URL to request
            max_retries: Maximum number of retries
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
        """
        retry_codes = {429, 500, 502, 503, 504}
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self._request(method, url, **kwargs)
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in retry_codes:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"Request failed with status {e.response.status_code}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            except requests.exceptions.RequestException:
                retry_count += 1
                wait_time = 2 ** retry_count
                logger.warning(f"Request failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        raise Exception(f"Failed after {max_retries} retries") 