"""
Utility module for fetching task mappings from GitHub repository.
This replaces the direct dependency on lm-eval for task metadata.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

# Global cache for task mappings during execution
_TASK_MAPPINGS_CACHE: dict[str, Any] = {}


def get_task_mappings() -> dict[str, Any]:
    """Fetch task mappings from GitHub repo and cache for this execution."""
    global _TASK_MAPPINGS_CACHE

    if _TASK_MAPPINGS_CACHE:
        return _TASK_MAPPINGS_CACHE

    repo_owner = os.getenv("OELLM_REPO_OWNER", "OpenEuroLLM")
    repo_name = os.getenv("OELLM_REPO_NAME", "multi-cluster-eval")
    branch = os.getenv("OELLM_REPO_BRANCH", "main")

    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/task_mappings.json"

    try:
        logging.info(f"Fetching task mappings from repo: {repo_owner}/{repo_name}")
        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        _TASK_MAPPINGS_CACHE = data.get("tasks", data)
        return _TASK_MAPPINGS_CACHE

    except (URLError, json.JSONDecodeError) as e:
        logging.warning(f"Failed to fetch task mappings from repo: {e}")
        _TASK_MAPPINGS_CACHE = {}
        return _TASK_MAPPINGS_CACHE


def get_task_duration(task_name: str) -> int:
    """Get estimated duration in minutes for a specific task."""
    mappings = get_task_mappings()
    task_info = mappings.get(task_name, {})
    return task_info.get("duration_minutes", 10)  # Default: 10 minutes


def get_task_datasets(task_name: str) -> list[dict[str, Any]]:
    """Get dataset information for a specific task."""
    mappings = get_task_mappings()
    task_info = mappings.get(task_name, {})
    return task_info.get("datasets", [])


def download_task_datasets(tasks: list[str], force_download: bool = False) -> None:
    """
    Download datasets for the given tasks using the task mappings.

    Args:
        tasks: List of task names
        force_download: Force re-download even if cached
    """
    from datasets import DownloadMode, load_dataset
    from huggingface_hub import snapshot_download

    processed = set()

    for task_name in tasks:
        if task_name in processed:
            continue
        processed.add(task_name)

        logging.info(f"Preparing dataset for task '{task_name}'...")

        datasets_info = get_task_datasets(task_name)

        if not datasets_info:
            logging.warning(f"No dataset information found for task '{task_name}'")
            continue

        for dataset_info in datasets_info:
            dataset_path = dataset_info.get("dataset_path")
            dataset_name = dataset_info.get("dataset_name")
            dataset_kwargs = dataset_info.get("dataset_kwargs", {})

            if not dataset_path:
                continue

            try:
                # Try to download/cache the dataset
                download_mode = (
                    DownloadMode.FORCE_REDOWNLOAD
                    if force_download
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                )

                # Prepare load arguments
                load_args = {
                    "path": dataset_path,
                    "download_mode": download_mode,
                    "cache_dir": os.getenv("HF_HOME"),
                }

                if dataset_name:
                    load_args["name"] = dataset_name

                # Add any additional kwargs
                load_args.update(dataset_kwargs)

                # Load dataset (this will download if not cached)
                load_dataset(**load_args)

                logging.debug(f"Dataset ready for task '{task_name}': {dataset_path}")

            except Exception as e:
                # Try snapshot download as fallback
                try:
                    snapshot_download(
                        repo_id=dataset_path,
                        repo_type="dataset",
                        cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                    )
                    logging.debug(
                        f"Downloaded dataset snapshot for '{task_name}': {dataset_path}"
                    )
                except Exception as e2:
                    logging.warning(
                        f"Failed to download dataset for task '{task_name}': {e}, {e2}"
                    )


def calculate_task_minutes(task_name: str, base_minutes_per_subtask: int = 5) -> int:
    """
    Get estimated minutes for a task.

    Args:
        task_name: Name of the task
        base_minutes_per_subtask: Ignored (kept for compatibility)

    Returns:
        Estimated total minutes for the task
    """
    return get_task_duration(task_name)
