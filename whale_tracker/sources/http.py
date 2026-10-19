"""Shared HTTP plumbing: one session, timeouts, and backoff on 429/5xx."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "whale-tracker/0.1"})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    timeout: float = 15,
    sleep=time.sleep,
) -> Any:
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == retries:
                raise
            sleep(2 ** attempt)
            continue
        if response.status_code in _RETRY_STATUS and attempt < retries:
            wait = _retry_after(response) or 2 ** attempt
            logger.warning("GET %s → %s, retrying in %.0fs", url, response.status_code, wait)
            sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


def _retry_after(response: requests.Response) -> Optional[float]:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
