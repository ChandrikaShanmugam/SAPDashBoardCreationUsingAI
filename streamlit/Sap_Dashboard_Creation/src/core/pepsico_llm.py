"""Pepsico PepGenX LLM API wrapper

Provides a single function `invoke_llm(payload, timeout=60, headers=None)` that calls
the PepGenX LLM endpoint used in the user's curl example. Headers and endpoint URL
are read from environment variables when not provided directly.

This file is intentionally minimal and returns Python dicts with either a `text`
field (the API text response) or an `error` field on failure.
"""
import os
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# OAuth2 configuration for runtime token generation
OAUTH2_TOKEN_URL = "https://pepsico.oktapreview.com/oauth2/default/v1/token"
OAUTH2_CLIENT_AUTH = "Basic MG9hMjQ3djJiaWl2SHdBQUIwaDg6ckZJT2NsSFV0VnpxaGFTV0hndzNzWkhOVmFYSGRLMlgzUlgwM3VVZGdsRVlnSXgyNFQ1UjcxQnpCY1ZGSjdYZQ=="

# Hardcoded credentials for PepGenX API
PEPGENX_URL = "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/llm/openai/generate-response"
PEPGENX_TEAM_ID = "22e767e3-1117-4524-af64-51687228b3b6"
PEPGENX_PROJECT_ID = "4664ddc3-552a-435b-b39f-ae4bf4cddaa1"
PEPGENX_APIKEY = "270b55d4-26a4-4078-9dd7-119633825268"

# Token cache
_token_cache = {
    "access_token": None,
    "expires_at": None
}


# Token cache
_token_cache = {
    "access_token": None,
    "expires_at": None
}


def get_bearer_token(force_refresh: bool = False) -> Optional[str]:
    """Fetch a new bearer token from Okta OAuth2 endpoint or return cached token.
    
    Args:
        force_refresh: If True, fetch a new token even if cached token is valid.
        
    Returns:
        Bearer token string or None if fetch fails.
    """
    global _token_cache
    
    # Check if we have a valid cached token
    if not force_refresh and _token_cache["access_token"] and _token_cache["expires_at"]:
        if datetime.now() < _token_cache["expires_at"]:
            logger.debug("Using cached bearer token")
            return _token_cache["access_token"]
    
    # Fetch new token
    try:
        logger.info("Fetching new bearer token from OAuth2 endpoint")
        headers = {
            "Authorization": OAUTH2_CLIENT_AUTH,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials"
        }
        
        response = requests.post(OAUTH2_TOKEN_URL, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
        
        if access_token:
            # Cache the token with a 5-minute buffer before expiry
            _token_cache["access_token"] = access_token
            _token_cache["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 300)
            logger.info("Bearer token fetched successfully, expires in %d seconds", expires_in)
            return access_token
        else:
            logger.error("No access_token in OAuth2 response")
            return None
            
    except Exception as e:
        logger.exception("Failed to fetch bearer token from OAuth2 endpoint")
        return None


def _build_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build headers for PepGenX API request with runtime bearer token."""
    bearer_token = get_bearer_token()
    
    if not bearer_token:
        logger.warning("Failed to get bearer token, API call may fail")
        bearer_token = ""  # Will likely fail but allows graceful error handling
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
        "team_id": PEPGENX_TEAM_ID,
        "project_id": PEPGENX_PROJECT_ID,
        "x-pepgenx-apikey": PEPGENX_APIKEY
    }
    if extra:
        headers.update(extra)
    return headers


def invoke_llm(payload: Dict[str, Any], timeout: int = 60, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Invoke the PepGenX LLM endpoint and return a parsed response.

    Args:
        payload: JSON payload to send (will be json-dumped).
        timeout: Request timeout in seconds.
        headers: Optional extra headers; if None will use hardcoded credentials.

    Returns:
        A dict representing the response. On success it may contain `text`.
        On failure it contains `error` describing the issue.
    """
    url = PEPGENX_URL
    hdrs = _build_headers(headers)

    try:
        logger.debug("Invoking PepGenX LLM endpoint %s", url)
        resp = requests.post(url, headers=hdrs, data=json.dumps(payload), timeout=timeout)
    except Exception as e:
        logger.exception("Network error calling PepGenX LLM endpoint")
        return {"error": str(e)}

    try:
        resp.raise_for_status()
    except Exception as e:
        # Try to include response text for debugging
        body = None
        try:
            body = resp.text
        except Exception:
            body = "<no body>"
        logger.error("PepGenX LLM returned non-200: %s - %s", resp.status_code, body)
        return {"error": f"HTTP {resp.status_code}: {body}"}

    try:
        result = resp.json()
    except Exception:
        # Return raw text if not JSON
        return {"text": resp.text}

    return result
