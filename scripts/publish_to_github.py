"""Publish metrics.json to the fpl-metrics-data GitHub repo via the Contents API.

Uses a personal access token (GITHUB_TOKEN in .env) with write access to
config.GITHUB_REPO. On any failure (missing token, network error, API
error) this logs exactly what failed and returns False rather than
raising -- a publish failure must never take down the rest of the
pipeline (main.py still writes metrics.json locally either way).
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish_to_github")

GITHUB_API_BASE = "https://api.github.com"


def _contents_url() -> str:
    return f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/contents/{config.GITHUB_METRICS_PATH}"


def _public_url() -> str:
    return f"https://raw.githubusercontent.com/{config.GITHUB_REPO}/{config.GITHUB_BRANCH}/{config.GITHUB_METRICS_PATH}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_existing_sha() -> str | None:
    """Current file's blob SHA (required by the Contents API to update an existing
    file), or None if it doesn't exist yet (first publish creates it)."""
    resp = requests.get(_contents_url(), headers=_headers(), params={"ref": config.GITHUB_BRANCH}, timeout=30)
    if resp.status_code == 200:
        return resp.json().get("sha")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return None


def publish_metrics(metrics: dict, commit_message: str | None = None) -> bool:
    """Push `metrics` as JSON to config.GITHUB_REPO's config.GITHUB_METRICS_PATH
    on config.GITHUB_BRANCH.

    Returns True on success, False on any failure. On success the data is
    publicly readable at:
        https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_METRICS_PATH}
    """
    if not config.GITHUB_TOKEN:
        logger.error(
            "GitHub publish FAILED: GITHUB_TOKEN is not set. Add it to .env (see .env.example) -- "
            "a personal access token with 'Contents: Read and write' permission on %s.",
            config.GITHUB_REPO,
        )
        return False

    content_bytes = json.dumps(metrics, indent=2, ensure_ascii=False).encode("utf-8")
    message = commit_message or f"Update metrics.json (GW{metrics.get('gameweek', '?')})"

    try:
        sha = _get_existing_sha()
    except requests.RequestException as exc:
        logger.error("GitHub publish FAILED: could not check for an existing %s (%s).", config.GITHUB_METRICS_PATH, exc)
        return False

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": config.GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(_contents_url(), headers=_headers(), json=payload, timeout=30)
    except requests.RequestException as exc:
        logger.error("GitHub publish FAILED: network error while pushing %s (%s).", config.GITHUB_METRICS_PATH, exc)
        return False

    if resp.status_code not in (200, 201):
        logger.error("GitHub publish FAILED: PUT %s -> HTTP %d: %s", _contents_url(), resp.status_code, resp.text[:500])
        if resp.status_code == 401:
            logger.error("-> GITHUB_TOKEN is invalid or expired.")
        elif resp.status_code == 403:
            logger.error("-> GITHUB_TOKEN lacks write access to %s (needs 'Contents: Read and write').", config.GITHUB_REPO)
        elif resp.status_code == 404:
            logger.error("-> Repo %s or branch %s not found, or the token can't see it.", config.GITHUB_REPO, config.GITHUB_BRANCH)
        elif resp.status_code == 409:
            logger.error("-> Conflict: %s changed on GitHub between the SHA check and this push. Safe to retry.", config.GITHUB_METRICS_PATH)
        return False

    logger.info("GitHub publish OK: %s", _public_url())
    return True
