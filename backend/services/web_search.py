from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_TIMEOUT_SECONDS = 20
MAX_SERPAPI_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WEB_SEARCH_RESULTS = 5


def search_web(query: str) -> list[dict[str, str]]:
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Web search is not configured (set SERPAPI_API_KEY).")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Web search query must not be empty.")

    request_url = f"{SERPAPI_ENDPOINT}?{urlencode({
        'engine': 'google',
        'q': query.strip(),
        'api_key': api_key,
        'num': MAX_WEB_SEARCH_RESULTS,
    })}"
    request = Request(
        request_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=SERPAPI_TIMEOUT_SECONDS) as response:
            response_body = response.read(MAX_SERPAPI_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"SerpApi web search failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"SerpApi web search failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"SerpApi web search timed out after {SERPAPI_TIMEOUT_SECONDS} seconds."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"SerpApi web search failed: {exc}") from exc

    if len(response_body) > MAX_SERPAPI_RESPONSE_BYTES:
        raise RuntimeError("SerpApi web search response exceeded the 2 MiB limit.")
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SerpApi returned an unreadable response.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SerpApi returned an invalid response.")
    if payload.get("error"):
        raise RuntimeError(f"SerpApi web search failed: {payload['error']}")

    results = []
    organic_results = payload.get("organic_results", [])
    if not isinstance(organic_results, list):
        raise RuntimeError("SerpApi returned invalid organic search results.")
    for result in organic_results:
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        link = result.get("link")
        snippet = result.get("snippet")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(link, str) or not link.startswith(("https://", "http://")):
            continue
        results.append(
            {
                "title": title.strip()[:300],
                "link": link.strip()[:2000],
                "snippet": snippet.strip()[:1200]
                if isinstance(snippet, str)
                else "",
            }
        )
        if len(results) == MAX_WEB_SEARCH_RESULTS:
            break
    return results
