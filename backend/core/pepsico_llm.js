/**
 * PepGenX LLM Integration Module
 * JavaScript equivalent of pepsico_llm.py
 * Provides OAuth2 authentication and API client for PepGenX LLM service
 */

const axios = require('axios');

// Configuration constants (equivalent to Python constants)
const PEPGENX_URL = "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/llm/openai/generate-response";
const OAUTH2_TOKEN_URL = "https://pepsico.oktapreview.com/oauth2/default/v1/token";
const OAUTH2_CLIENT_AUTH = "Basic MG9hMjQ3djJiaWl2SHdBQUIwaDg6ckZJT2NsSFV0VnpxaGFTV0hndzNzWkhOVmFYSGRLMlgzUlgwM3VVZGdsRVlnSXgyNFQ1UjcxQnpCY1ZGSjdYZQ==";
const PEPGENX_TEAM_ID = "22e767e3-1117-4524-af64-51687228b3b6";
const PEPGENX_PROJECT_ID = "4664ddc3-552a-435b-b39f-ae4bf4cddaa1";
const PEPGENX_APIKEY = "270b55d4-26a4-4078-9dd7-119633825268"; // Should be loaded from environment

// Token cache (equivalent to Python global _token_cache)
let _token_cache = {
    access_token: null,
    expires_at: null
};

/**
 * Get bearer token from OAuth2 endpoint with caching
 * @param {boolean} force_refresh - Force token refresh even if cached
 * @returns {Promise<string|null>} Bearer token or null if failed
 */
async function get_bearer_token(force_refresh = false) {
    // Check if we have a valid cached token
    if (!force_refresh && _token_cache.access_token && _token_cache.expires_at) {
        if (new Date() < _token_cache.expires_at) {
            console.debug("Using cached bearer token");
            return _token_cache.access_token;
        }
    }

    // Fetch new token
    try {
        console.log("Fetching new bearer token from OAuth2 endpoint");
        const headers = {
            "Authorization": OAUTH2_CLIENT_AUTH,
            "Content-Type": "application/x-www-form-urlencoded"
        };
        const data = new URLSearchParams({
            "grant_type": "client_credentials"
        });

        const response = await axios.post(OAUTH2_TOKEN_URL, data, {
            headers: headers,
            timeout: 10000
        });

        const token_data = response.data;
        const access_token = token_data.access_token;
        const expires_in = token_data.expires_in || 3600; // Default 1 hour

        if (access_token) {
            // Cache the token with a 5-minute buffer before expiry
            _token_cache.access_token = access_token;
            _token_cache.expires_at = new Date(Date.now() + (expires_in - 300) * 1000);
            console.log(`Bearer token fetched successfully, expires in ${expires_in} seconds`);
            return access_token;
        } else {
            console.error("No access_token in OAuth2 response");
            return null;
        }

    } catch (error) {
        console.error("Failed to fetch bearer token from OAuth2 endpoint:", error.message);
        return null;
    }
}

/**
 * Build headers for PepGenX API request with runtime bearer token
 * @param {Object} extra - Optional extra headers
 * @returns {Promise<Object>} Headers object
 */
async function _build_headers(extra = null) {
    const bearer_token = await get_bearer_token();

    if (!bearer_token) {
        console.warn("Failed to get bearer token, API call may fail");
        // Will likely fail but allows graceful error handling
    }

    const headers = {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${bearer_token || ""}`,
        "team_id": PEPGENX_TEAM_ID,
        "project_id": PEPGENX_PROJECT_ID,
        "x-pepgenx-apikey": PEPGENX_APIKEY
    };

    if (extra) {
        Object.assign(headers, extra);
    }

    return headers;
}

/**
 * Invoke the PepGenX LLM endpoint and return a parsed response
 * @param {Object} payload - JSON payload to send
 * @param {number} timeout - Request timeout in seconds (default: 60)
 * @param {Object} headers - Optional extra headers
 * @returns {Promise<Object>} Response object with text or error
 */
async function invoke_llm(payload, timeout = 60, headers = null) {
    const url = PEPGENX_URL;
    const hdrs = headers || await _build_headers();

    try {
        console.debug(`Invoking PepGenX LLM endpoint ${url}`);
        const response = await axios.post(url, payload, {
            headers: hdrs,
            timeout: timeout * 1000
        });

        return response.data;

    } catch (error) {
        if (error.response) {
            // Server responded with error status
            const status = error.response.status;
            const body = error.response.data || "<no body>";
            console.error(`PepGenX LLM returned non-200: ${status} - ${JSON.stringify(body)}`);
            return { "error": `HTTP ${status}: ${JSON.stringify(body)}` };
        } else if (error.request) {
            // Network error
            console.error("Network error calling PepGenX LLM endpoint:", error.message);
            return { "error": error.message };
        } else {
            // Other error
            console.error("Error calling PepGenX LLM endpoint:", error.message);
            return { "error": error.message };
        }
    }
}

module.exports = {
    get_bearer_token,
    invoke_llm,
    PEPGENX_URL,
    OAUTH2_TOKEN_URL,
    PEPGENX_TEAM_ID,
    PEPGENX_PROJECT_ID
};