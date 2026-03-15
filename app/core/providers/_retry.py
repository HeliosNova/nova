"""Shared retry logic for transient HTTP errors across LLM providers."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx

from app.core.llm import LLMUnavailableError

logger = logging.getLogger(__name__)


async def retry_on_transient(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    **kwargs,
) -> httpx.Response:
    """Execute an HTTP request with retry logic for transient errors.

    - 429: retry with Retry-After header (up to max_retries)
    - 500/502/503: retry once after 2s
    - 400/401/403: raise LLMUnavailableError immediately
    """
    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(method, url, **kwargs)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "2"))
                retry_after = min(retry_after, 30.0)
                retry_after += random.uniform(0, retry_after * 0.25)  # jitter to avoid thundering herd
                if attempt < max_retries:
                    logger.warning("Rate limited (429), retrying after %.1fs (attempt %d/%d)", retry_after, attempt + 1, max_retries)
                    await asyncio.sleep(retry_after)
                    continue
                raise LLMUnavailableError(f"Rate limited after {max_retries} retries")

            if resp.status_code in (500, 502, 503):
                if attempt < max_retries:
                    delay = min(2 ** (attempt + 1), 16) + random.uniform(0, 1)
                    logger.warning("Server error (%d), retrying after %.1fs (attempt %d/%d)",
                                  resp.status_code, delay, attempt + 1, max_retries)
                    await asyncio.sleep(delay)
                    continue
                raise LLMUnavailableError(f"Server error {resp.status_code} after {max_retries} retries")

            if resp.status_code in (400, 401, 403):
                raise LLMUnavailableError(f"Auth/request error {resp.status_code}: {resp.text[:200]}")

            resp.raise_for_status()
            return resp

        except httpx.HTTPStatusError as e:
            raise LLMUnavailableError(f"HTTP error: {e}")
        except (httpx.ConnectError, httpx.TimeoutException):
            raise
        except LLMUnavailableError:
            raise
        except Exception:
            if attempt < max_retries:
                delay = min(2 ** (attempt + 1), 16) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            raise

    raise LLMUnavailableError("Max retries exceeded")
