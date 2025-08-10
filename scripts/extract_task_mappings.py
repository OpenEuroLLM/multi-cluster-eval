#!/usr/bin/env python3
"""
Extract task-to-dataset mappings from lm-eval library.
This script generates a JSON file containing all task metadata including
dataset paths, subtask counts, and download configurations.
"""

import json
import logging
from pathlib import Path
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_dataset_info(task_obj) -> dict[str, Any]:
    """Extract dataset information from a task object."""
    dataset_info = {
        "dataset_path": None,
        "dataset_name": None,
        "dataset_kwargs": {},
        "download_mode": None,
        "subtasks": [],
    }

    # Try to extract dataset path/name
    if hasattr(task_obj, "DATASET_PATH"):
        dataset_info["dataset_path"] = task_obj.DATASET_PATH
    elif hasattr(task_obj, "dataset_path"):
        dataset_info["dataset_path"] = task_obj.dataset_path

    if hasattr(task_obj, "DATASET_NAME"):
        dataset_info["dataset_name"] = task_obj.DATASET_NAME
    elif hasattr(task_obj, "dataset_name"):
        dataset_info["dataset_name"] = task_obj.dataset_name

    # Try to extract dataset kwargs (for loading configuration)
    if hasattr(task_obj, "dataset_kwargs"):
        dataset_info["dataset_kwargs"] = task_obj.dataset_kwargs
    elif hasattr(task_obj, "DATASET_KWARGS"):
        dataset_info["dataset_kwargs"] = task_obj.DATASET_KWARGS

    # Check if it has a download method
    if hasattr(task_obj, "download") and callable(task_obj.download):
        dataset_info["has_download_method"] = True
    else:
        dataset_info["has_download_method"] = False

    return dataset_info


def count_subtasks(task_name: str, task_manager) -> int:
    """Count the number of subtasks for a given task."""
    try:
        from lm_eval.evaluator_utils import get_subtask_list

        task_objects = task_manager.load_task_or_group(task_name)
        subtask_dict = get_subtask_list(task_objects)

        total_subtasks = 0
        subtask_names = []
        for _group_name, subtask_list in subtask_dict.items():
            total_subtasks += len(subtask_list)
            subtask_names.extend([str(task) for task in subtask_list])

        return max(1, total_subtasks), subtask_names
    except Exception as e:
        logging.warning(f"Could not count subtasks for {task_name}: {e}")
        return 1, []


def extract_all_task_mappings() -> dict[str, Any]:
    """Extract mappings for all available tasks in lm-eval."""
    try:
        from lm_eval.tasks import TaskManager
    except ImportError:
        logging.error(
            "lm-eval not installed. Please install it first: pip install lm-eval"
        )
        return {}

    logging.info("Initializing TaskManager...")
    tm = TaskManager()

    # Get all available tasks
    all_tasks = tm.all_tasks
    logging.info(f"Found {len(all_tasks)} tasks to process")

    mappings = {}
    processed_count = 0

    for task_name in all_tasks:
        try:
            logging.debug(f"Processing task: {task_name}")

            # Load the task
            task_objects = tm.load_task_or_group(task_name)

            # Count subtasks
            subtask_count, subtask_names = count_subtasks(task_name, tm)

            # Initialize task info
            task_info = {
                "subtask_count": subtask_count,
                "subtask_names": subtask_names,
                "datasets": [],
            }

            # Process task objects to extract dataset info
            stack = [task_objects]
            seen_datasets = set()

            while stack:
                current = stack.pop()

                if isinstance(current, dict):
                    stack.extend(current.values())
                    continue

                # Extract dataset info from the task object
                dataset_info = extract_dataset_info(current)

                # Create a unique key for deduplication
                dataset_key = (
                    dataset_info.get("dataset_path"),
                    dataset_info.get("dataset_name"),
                    json.dumps(dataset_info.get("dataset_kwargs", {}), sort_keys=True),
                )

                if (
                    dataset_key != (None, None, "{}")
                    and dataset_key not in seen_datasets
                ):
                    seen_datasets.add(dataset_key)
                    task_info["datasets"].append(
                        {
                            "dataset_path": dataset_info["dataset_path"],
                            "dataset_name": dataset_info["dataset_name"],
                            "dataset_kwargs": dataset_info["dataset_kwargs"],
                            "has_download_method": dataset_info["has_download_method"],
                        }
                    )

            mappings[task_name] = task_info
            processed_count += 1

            if processed_count % 50 == 0:
                logging.info(f"Processed {processed_count}/{len(all_tasks)} tasks...")

        except Exception as e:
            logging.warning(f"Failed to process task {task_name}: {e}")
            mappings[task_name] = {
                "error": str(e),
                "subtask_count": 1,
                "subtask_names": [],
                "datasets": [],
            }

    logging.info(f"Successfully processed {processed_count} tasks")
    return mappings


def add_known_complex_tasks(mappings: dict[str, Any]) -> None:
    """Add timing metadata for known complex tasks."""
    known_complex_tasks = {
        "belebele": 8,
        "flores": 6,
        "xnli": 6,
        "xcopa": 6,
        "xstory_cloze": 6,
        "paws-x": 6,
        "hellaswag": 20,
    }

    for task_name, minutes_per_subtask in known_complex_tasks.items():
        if task_name in mappings:
            mappings[task_name]["minutes_per_subtask"] = minutes_per_subtask


def main():
    """Main function to extract and save task mappings."""
    logging.info("Starting task mapping extraction...")

    # Extract all mappings
    mappings = extract_all_task_mappings()

    if not mappings:
        logging.error("No mappings extracted. Exiting.")
        return 1

    # Add known complex task timings
    add_known_complex_tasks(mappings)

    # Add metadata
    metadata = {
        "version": "1.0.0",
        "task_count": len(mappings),
        "generated_at": str(Path(__file__).parent.parent / "task_mappings.json"),
    }

    output = {"metadata": metadata, "tasks": mappings}

    # Save to JSON file
    output_path = Path("task_mappings.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    logging.info(f"Task mappings saved to {output_path}")
    logging.info(f"Total tasks: {len(mappings)}")

    # Print summary statistics
    tasks_with_datasets = sum(1 for task in mappings.values() if task.get("datasets"))
    tasks_with_errors = sum(1 for task in mappings.values() if "error" in task)

    logging.info(f"Tasks with datasets: {tasks_with_datasets}")
    logging.info(f"Tasks with errors: {tasks_with_errors}")

    return 0


if __name__ == "__main__":
    exit(main())
