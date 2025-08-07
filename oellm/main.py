import logging
import os
import re
import socket
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from string import Template
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from jsonargparse import auto_cli
from rich.console import Console
from rich.logging import RichHandler


def ensure_singularity_image(image_name: str) -> None:
    # TODO: switch to OELLM dataset repo once it is created
    from huggingface_hub import hf_hub_download

    hf_repo = os.environ.get("HF_SIF_REPO", "timurcarstensen/testing")
    image_path = Path(os.getenv("EVAL_BASE_DIR")) / image_name

    try:
        hf_hub_download(
            repo_id=hf_repo,
            filename=image_name,
            repo_type="dataset",
            local_dir=os.getenv("EVAL_BASE_DIR"),
        )
        logging.info(
            "Successfully downloaded latest Singularity image from HuggingFace"
        )
    except Exception as e:
        logging.warning(
            "Failed to fetch latest container image from HuggingFace: %s", str(e)
        )
        if image_path.exists():
            logging.info("Using existing Singularity image at %s", image_path)
        else:
            raise RuntimeError(
                f"No container image found at {image_path} and failed to download from HuggingFace. "
                f"Cannot proceed with evaluation scheduling."
            ) from e

    logging.info(
        "Singularity image ready at %s",
        Path(os.getenv("EVAL_BASE_DIR")) / os.getenv("EVAL_CONTAINER_IMAGE"),
    )


def _setup_logging(verbose: bool = False):
    rich_handler = RichHandler(
        console=Console(),
        show_time=True,
        log_time_format="%H:%M:%S",
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    class RichFormatter(logging.Formatter):
        def format(self, record):
            # Define colors for different log levels
            record.msg = f"{record.getMessage()}"
            return record.msg

    rich_handler.setFormatter(RichFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove any default handlers
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _load_cluster_env() -> None:
    """
    Loads the correct cluster environment variables from `clusters.yaml` based on the hostname.
    """
    with open(Path(__file__).parent / "clusters.yaml") as f:
        clusters = yaml.safe_load(f)
    hostname = socket.gethostname()

    # First load shared environment variables
    shared_cfg = clusters.get("shared", {})

    # match hostname to the regex in the clusters.yaml
    for host in set(clusters.keys()) - {"shared"}:
        pattern = clusters[host]["hostname_pattern"]
        # Convert shell-style wildcards to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        if re.match(f"^{regex_pattern}$", hostname):
            cluster_cfg = clusters[host]
            break
    else:
        raise ValueError(f"No cluster found for hostname: {hostname}")

    # Combine shared and cluster-specific configs, with cluster-specific taking precedence
    # Remove hostname_pattern from the final config
    if "hostname_pattern" in cluster_cfg:
        del cluster_cfg["hostname_pattern"]

    # Set environment variables, expanding any template variables
    for k, v in cluster_cfg.items():
        # Expand template variables using existing environment variables
        os.environ[k] = str(v)

    for k, v in shared_cfg.items():
        try:
            os.environ[k] = str(v).format(**cluster_cfg)
        except KeyError as e:
            # when substituting env vars that are not in cluster_cfg but in the environment (e.g., $USER, $SHELL, etc...)
            if len(e.args) > 1:
                raise ValueError(
                    f"Env. variable substitution for {k} failed. Missing keys: {', '.join(e.args)}"
                ) from e

            missing_key: str = e.args[0]
            os.environ[k] = str(v).format(
                **cluster_cfg, **{missing_key: os.environ[missing_key]}
            )


def _num_jobs_in_queue() -> int:
    # TODO avoid running in shell mode which is not secure
    result = subprocess.run(
        "squeue -u $USER -h -t pending,running -r | wc -l",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        try:
            return int(result.stdout.strip())
        except ValueError:
            logging.warning(f"Could not parse squeue output: {result.stdout}")
            return 0

    if result.stderr:
        logging.warning(f"squeue command produced an error: {result.stderr.strip()}")

    return 0


def _expand_local_model_paths(model: str) -> list[Path]:
    """
    Expands a local model path to include all checkpoints if it's a directory.
    Recursively searches for models in subdirectories.
    
    Args:
        model: Path to a model or directory containing models
    
    Returns:
        List of paths to model directories containing safetensors files
    """
    model_paths = []
    model_path = Path(model)
    
    if not model_path.exists() or not model_path.is_dir():
        return model_paths
    
    # First check if current directory contains safetensors files
    if any(model_path.glob("*.safetensors")):
        model_paths.append(model_path)
        # If current dir has safetensors, don't recurse further
        return model_paths
    
    # Check for hf subdirectory pattern (single model with checkpoints)
    hf_path = model_path / "hf"
    if hf_path.exists() and hf_path.is_dir():
        # This is a single model with checkpoints in hf/iter_* structure
        for subdir in hf_path.glob("*"):
            if subdir.is_dir() and any(subdir.glob("*.safetensors")):
                model_paths.append(subdir)
        if model_paths:
            return model_paths
    
    # Check if subdirectories look like model directories
    # (e.g., open-sci-ref_model-0.13b_data-c4_...)
    subdirs = [d for d in model_path.iterdir() if d.is_dir()]
    
    # Process each subdirectory as a potential model
    for subdir in subdirs:
        # Check if this subdirectory directly contains safetensors
        if any(subdir.glob("*.safetensors")):
            model_paths.append(subdir)
        else:
            # Check for hf/iter_* pattern in this subdirectory
            hf_subpath = subdir / "hf"
            if hf_subpath.exists() and hf_subpath.is_dir():
                for checkpoint_dir in hf_subpath.glob("*"):
                    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.safetensors")):
                        model_paths.append(checkpoint_dir)
    
    if len(model_paths) > 1:
        logging.info(f"Expanded '{model}' to {len(model_paths)} model checkpoints")
    
    return model_paths


def _process_model_paths(models: Iterable[str]) -> dict[str, list[Path | str]]:
    """
    Processes model strings into a dict of model paths.

    Each model string can be a local path or a huggingface model identifier.
    This function expands directory paths that contain multiple checkpoints.
    """
    from huggingface_hub import snapshot_download

    processed_model_paths = {}
    model_paths = []
    for model in models:
        # First try to expand local paths
        local_paths = _expand_local_model_paths(model)
        if local_paths:
            model_paths.extend(local_paths)
        else:
            logging.info(
                f"Model {model} not found locally, assuming it is a 🤗 hub model"
            )
            logging.debug(
                f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
            )

            if "," in model:
                model_kwargs = dict(
                    [kv.split("=") for kv in model.split(",") if "=" in kv]
                )

                # The first element before the comma is the repository ID on the 🤗 Hub
                repo_id = model.split(",")[0]

                # snapshot_download kwargs
                snapshot_kwargs = {}
                if "revision" in model_kwargs:
                    snapshot_kwargs["revision"] = model_kwargs["revision"]

                try:
                    # Pre-download (or reuse cache) for the whole repository so that
                    # compute nodes can load it offline.
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                        **snapshot_kwargs,
                    )
                    model_paths.append(model)
                except Exception as e:
                    logging.debug(
                        f"Failed to download model {model} from Hugging Face Hub. Continuing..."
                    )
                    logging.debug(e)
            else:
                # Download the entire model repository to the local cache.  The
                # original identifier is kept in *model_paths* so downstream
                # code can still reference it; at runtime the files will be
                # read from cache, allowing offline execution.
                snapshot_download(
                    repo_id=model,
                    cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                )
                model_paths.append(model)

        if not model_paths:
            logging.warning(
                f"Could not find any valid model for '{model}'. It will be skipped."
            )
        processed_model_paths[model] = model_paths
    return processed_model_paths


def _count_task_subtasks(task_name: str, task_manager) -> int:
    from lm_eval.evaluator_utils import get_subtask_list  # type: ignore

    task_objects = task_manager.load_task_or_group(task_name)
    subtask_dict = get_subtask_list(task_objects)

    total_subtasks = 0
    for _, subtask_list in subtask_dict.items():
        total_subtasks += len(subtask_list)

    return max(1, total_subtasks)  # At least 1 subtask


def _calculate_task_minutes(
    task_name: str, task_manager, base_minutes_per_subtask: int = 5
) -> int:
    """Calculate estimated minutes for a task based on its subtask count."""
    subtask_count = _count_task_subtasks(task_name, task_manager)

    # Special handling for known multi-language tasks that take longer per subtask
    known_complex_tasks = {
        "belebele": 8,  # Multi-language reading comprehension, slower per subtask
        "flores": 6,  # Translation task, moderately complex
        "xnli": 6,  # Cross-lingual NLI
        "xcopa": 6,  # Cross-lingual COPA
        "xstory_cloze": 6,  # Cross-lingual story cloze
        "paws-x": 6,  # Cross-lingual paraphrase detection
        "hellaswag": 20,  # Hellaswag task, needs 20 minutes per subtask
    }

    # Use task-specific timing if available, otherwise use default
    minutes_per_subtask = known_complex_tasks.get(
        task_name.lower(), base_minutes_per_subtask
    )

    # Calculate total time: (subtasks × time_per_subtask) + base_overhead
    base_overhead = 3  # Base overhead for task setup/teardown
    total_minutes = max(10, (subtask_count * minutes_per_subtask) + base_overhead)

    # Log for complex tasks (>5 subtasks) or any known complex task
    if subtask_count > 5 or task_name.lower() in known_complex_tasks:
        complexity_note = (
            f" (known complex task, {minutes_per_subtask} min/subtask)"
            if task_name.lower() in known_complex_tasks
            else ""
        )
        logging.info(
            f"📊 Task '{task_name}' has {subtask_count} subtasks{complexity_note}, "
            f"estimated time: {total_minutes} minutes ({total_minutes / 60:.1f} hours)"
        )

    return total_minutes


def _pre_download_task_datasets(tasks: Iterable[str]) -> None:
    """Ensure that all datasets required by the given `tasks` are present in the local 🤗 cache at $HF_HOME."""

    from datasets import DownloadMode  # type: ignore
    from lm_eval.tasks import TaskManager  # type: ignore

    processed: set[str] = set()

    tm = TaskManager()

    for task_name in tasks:
        if not isinstance(task_name, str) or task_name in processed:
            continue
        processed.add(task_name)

        logging.info(
            f"Preparing dataset for task '{task_name}' (download if not cached)…"
        )

        # Instantiating the task downloads the dataset (or reuses cache)
        task_objects = tm.load_task_or_group(task_name)

        # Some entries might be nested dictionaries (e.g., groups)
        stack = [task_objects]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
                continue
            if hasattr(current, "download") and callable(current.download):
                try:
                    current.download(download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)  # type: ignore[arg-type]
                except TypeError as e:
                    logging.error(
                        f"Failed to download dataset for task '{task_name}' with download_mode=REUSE_DATASET_IF_EXISTS: {e}"
                    )
                    current.download()  # type: ignore[misc]

        logging.debug(f"Finished dataset preparation for task '{task_name}'.")


def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    n_shot: int | list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    max_array_len: int = 32,
    verbose: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    skip_checks: bool = False,
) -> None:
    """
    Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: A string of comma-separated model paths or Hugging Face model identifiers.
            Warning: does not allow passing model args such as `EleutherAI/pythia-160m,revision=step100000`
            since we split on commas. If you need to pass model args, use the `eval_csv_path` option.
            For local paths:
            - If a directory contains `.safetensors` files directly, it will be treated as a single model
            - If a directory contains subdirectories with models (e.g., converted_checkpoints/), 
              all models in subdirectories will be automatically discovered
            - For each model directory, if it has an `hf/iter_XXXXX` structure, all checkpoints will be expanded
            - This allows passing a single directory containing multiple models to evaluate them all
        tasks: A string of comma-separated task paths.
        n_shot: An integer or list of integers specifying the number of shots for each task.
        eval_csv_path: A path to a CSV file containing evaluation data.
            Warning: exclusive argument. Cannot specify `models`, `tasks`, or `n_shot` when `eval_csv_path` is provided.
        max_array_len: The maximum number of jobs to schedule to run concurrently.
            Warning: this is not the number of jobs in the array job. This is determined by the environment variable `QUEUE_LIMIT`.
        download_only: If True, only download the datasets and models and exit.
        dry_run: If True, generate the SLURM script but don't submit it to the scheduler.
        skip_checks: If True, skip container image, model validation, and dataset pre-download checks for faster execution.
    """
    _setup_logging(verbose)

    # Load cluster-specific environment variables (paths, etc.)
    _load_cluster_env()

    if not skip_checks:
        image_name = os.environ.get("EVAL_CONTAINER_IMAGE")
        if image_name is None:
            raise ValueError(
                "EVAL_CONTAINER_IMAGE is not set. Please set it in clusters.yaml."
            )

        ensure_singularity_image(image_name)
    else:
        logging.info("Skipping container image check (--skip-checks enabled)")

    if eval_csv_path:
        if models or tasks or n_shot:
            raise ValueError(
                "Cannot specify `models`, `tasks`, or `n_shot` when `eval_csv_path` is provided."
            )
        df = pd.read_csv(eval_csv_path)
        required_cols = {"model_path", "task_path", "n_shot"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain the columns: {', '.join(required_cols)}"
            )

        # Always expand local model paths, even with skip_checks
        df["model_path"].unique()
        expanded_rows = []
        for _, row in df.iterrows():
            original_model_path = row["model_path"]
            local_paths = _expand_local_model_paths(original_model_path)
            if local_paths:
                # Use expanded local paths
                for expanded_path in local_paths:
                    new_row = row.copy()
                    new_row["model_path"] = expanded_path
                    expanded_rows.append(new_row)
            else:
                # Keep original path (might be HF model)
                expanded_rows.append(row)
        df = pd.DataFrame(expanded_rows)

        # Download HF models only if skip_checks is False
        if not skip_checks:
            # Process any HF models that need downloading
            hf_models = [m for m in df["model_path"].unique() if not Path(m).exists()]
            if hf_models:
                model_path_map = _process_model_paths(hf_models)
                # Update the dataframe with processed HF models
                for idx, row in df.iterrows():
                    if row["model_path"] in model_path_map:
                        # This shouldn't expand further, just update the path
                        df.at[idx, "model_path"] = model_path_map[row["model_path"]][0]
        else:
            logging.info(
                "Skipping model path processing and validation (--skip-checks enabled)"
            )

    elif models and tasks and n_shot is not None:
        model_list = models.split(",")
        model_paths = []

        # Always expand local paths
        for model in model_list:
            local_paths = _expand_local_model_paths(model)
            if local_paths:
                model_paths.extend(local_paths)
            else:
                model_paths.append(model)

        # Download HF models only if skip_checks is False
        if not skip_checks:
            hf_models = [m for m in model_paths if not Path(m).exists()]
            if hf_models:
                model_path_map = _process_model_paths(hf_models)
                # Replace HF model identifiers with processed paths
                model_paths = [
                    model_path_map[m][0] if m in model_path_map else m
                    for m in model_paths
                ]
        else:
            logging.info(
                "Skipping model path processing and validation (--skip-checks enabled)"
            )

        tasks_list = tasks.split(",")

        # cross product of model_paths and tasks into a dataframe
        df = pd.DataFrame(
            product(
                model_paths,
                tasks_list,
                n_shot if isinstance(n_shot, list) else [n_shot],
            ),
            columns=["model_path", "task_path", "n_shot"],
        )
    else:
        raise ValueError(
            "Either `eval_csv_path` must be provided, or all of `models`, `tasks`, and `n_shot`."
        )

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    if not skip_checks:
        _pre_download_task_datasets(df["task_path"].unique())
    else:
        logging.info("Skipping dataset pre-download (--skip-checks enabled)")

    if download_only:
        return None

    queue_limit = int(os.environ.get("QUEUE_LIMIT", 250))
    remaining_queue_capacity = queue_limit - _num_jobs_in_queue()

    if remaining_queue_capacity <= 0:
        logging.warning("No remaining queue capacity. Not scheduling any jobs.")
        return None

    logging.debug(
        f"Remaining capacity in the queue: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)}."
    )

    evals_dir = (
        Path(os.environ["EVAL_OUTPUT_DIR"])
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    evals_dir.mkdir(parents=True, exist_ok=True)

    slurm_logs_dir = evals_dir / "slurm_logs"
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evals_dir / "jobs.csv"

    df.to_csv(csv_path, index=False)

    logging.debug(f"Saved evaluation dataframe to temporary CSV: {csv_path}")

    with open(Path(__file__).parent / "template.sbatch") as f:
        sbatch_template = f.read()

    # Calculate dynamic array size and time limits
    total_evals = len(df)

    # Calculate time based on actual task complexity (subtask count)
    if not skip_checks:
        from lm_eval.tasks import TaskManager  # type: ignore

        shared_task_manager = TaskManager()

        # Calculate total minutes by considering each unique task's complexity
        total_minutes = 0
        task_time_cache = {}  # Cache to avoid recalculating for same tasks

        for _, row in df.iterrows():
            task_name = row["task_path"]
            if task_name not in task_time_cache:
                task_time_cache[task_name] = _calculate_task_minutes(
                    task_name, task_manager=shared_task_manager
                )
            total_minutes += task_time_cache[task_name]

        # Calculate average minutes per eval for logging purposes
        minutes_per_eval = total_minutes / total_evals if total_evals > 0 else 10

        logging.info("📊 Dynamic time calculation:")
        for task_name, task_minutes in task_time_cache.items():
            task_count = (df["task_path"] == task_name).sum()
            logging.info(
                f"   Task '{task_name}': {task_minutes} min/eval × {task_count} evals = {task_minutes * task_count} total minutes"
            )
    else:
        # Fallback to fixed timing when checks are skipped
        minutes_per_eval = 10  # Budget 10 minutes per eval
        total_minutes = total_evals * minutes_per_eval
        logging.info(
            "⚠️  Using fixed 10 min/eval (task complexity detection skipped with --skip-checks)"
        )

    # Maximum runtime per job (18 hours with safety margin)
    max_minutes_per_job = 18 * 60  # 18 hours
    min_array_size_for_time = max(1, int(np.ceil(total_minutes / max_minutes_per_job)))
    desired_array_size = min(128, total_evals) if total_evals >= 128 else total_evals
    if desired_array_size < min_array_size_for_time:
        desired_array_size = min_array_size_for_time

    # The actual array size is limited by queue capacity and total evals
    actual_array_size = min(remaining_queue_capacity, desired_array_size, total_evals)

    # Calculate actual time per job
    evals_per_job = max(1, int(np.ceil(total_evals / actual_array_size)))
    minutes_per_job = evals_per_job * minutes_per_eval

    # Add 20% safety margin and round up to nearest hour
    minutes_with_margin = int(minutes_per_job * 1.2)
    hours_with_margin = max(1, int(np.ceil(minutes_with_margin / 60)))

    # Apply 3-hour safety minimum for array jobs
    hours_with_margin = max(hours_with_margin, 3)

    # Cap at 24 hours
    hours_with_margin = min(hours_with_margin, 24)

    # Format time limit for SLURM (HH:MM:SS)
    time_limit = f"{hours_with_margin:02d}:00:00"

    # Log the calculated values
    logging.info("📊 Evaluation planning:")
    logging.info(f"   Total evaluations: {total_evals}")
    logging.info(f"   Estimated time per eval: {minutes_per_eval} minutes")
    logging.info(
        f"   Total estimated time: {total_minutes} minutes ({total_minutes / 60:.1f} hours)"
    )
    logging.info(f"   Desired array size: {desired_array_size}")
    logging.info(
        f"   Actual array size: {actual_array_size} (limited by queue capacity: {remaining_queue_capacity})"
    )
    logging.info(f"   Evaluations per job: {evals_per_job}")
    logging.info(
        f"   Time per job: {minutes_per_job} minutes ({minutes_per_job / 60:.1f} hours)"
    )
    logging.info(f"   Time limit with safety margin: {time_limit}")

    # replace the placeholders in the template with the actual values
    # First, replace python-style placeholders
    sbatch_script = sbatch_template.format(
        csv_path=csv_path,
        max_array_len=max_array_len,
        array_limit=actual_array_size - 1,  # Array is 0-indexed
        num_jobs=actual_array_size,  # This is the number of array jobs, not total evals
        total_evals=len(df),  # Pass the total number of evaluations
        log_dir=evals_dir / "slurm_logs",
        evals_dir=str(evals_dir / "results"),
        time_limit=time_limit,  # Dynamic time limit
    )

    # substitute any $ENV_VAR occurrences (e.g., $TIME_LIMIT) since env vars are not
    # expanded in the #SBATCH directives
    sbatch_script = Template(sbatch_script).safe_substitute(os.environ)

    # Save the sbatch script to the evals directory
    sbatch_script_path = evals_dir / "submit_evals.sbatch"
    logging.debug(f"Saving sbatch script to {sbatch_script_path}")

    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    if dry_run:
        logging.info(f"Dry run mode: SLURM script generated at {sbatch_script_path}")
        logging.info(
            f"Would schedule {actual_array_size} array jobs to handle {len(df)} evaluations"
        )
        logging.info(
            f"Each array job will handle ~{(len(df) + actual_array_size - 1) // actual_array_size} evaluations"
        )
        logging.info("To submit the job, run: sbatch " + str(sbatch_script_path))
        return

    # Submit the job script to slurm by piping the script content to sbatch
    try:
        logging.info("Calling sbatch to launch the evaluations")

        # Provide helpful information about job monitoring and file locations
        logging.info(f"📁 Evaluation directory: {evals_dir}")
        logging.info(f"📄 SLURM script: {sbatch_script_path}")
        logging.info(f"📋 Job configuration: {csv_path}")
        logging.info(f"📜 SLURM logs will be stored in: {slurm_logs_dir}")
        logging.info(f"📊 Results will be stored in: {evals_dir / 'results'}")

        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
            capture_output=True,
            env=os.environ,
        )
        logging.info("Job submitted successfully.")
        logging.info(result.stdout)
        # Extract job ID from sbatch output for monitoring commands
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            logging.info(f"🔍 Monitor job status: squeue -j {job_id}")
            logging.info(f"📈 View job details: scontrol show job {job_id}")
            logging.info(f"❌ Cancel job if needed: scancel {job_id}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"sbatch stderr: {e.stderr}")
    except FileNotFoundError:
        logging.error(
            "sbatch command not found. Please make sure you are on a system with SLURM installed."
        )


def build_csv(
    output_path: str = "eval_config.csv",
    *,
    verbose: bool = False,
) -> None:
    """
    Build a CSV file for evaluation with per-task n_shot configurations using the interactive builder.

    Args:
        output_path: Path where the CSV file will be saved.
        verbose: Enable verbose logging.
    """
    _setup_logging(verbose)

    from oellm.interactive_csv_builder import build_csv_interactive

    build_csv_interactive(output_path)


def collect_results(
    results_dir: str,
    output_csv: str = "eval_results.csv",
    *,
    check: bool = False,
    reschedule: bool = False,
    verbose: bool = False,
) -> None:
    """
    Collect evaluation results from JSON files and export to CSV.

    Args:
        results_dir: Path to the directory containing result JSON files
        output_csv: Output CSV filename (default: eval_results.csv)
        check: Check for crashed or pending evaluations
        reschedule: Show overview table and prompt to reschedule failed/pending jobs
        verbose: Enable verbose logging
    """
    import json

    from rich.table import Table

    _setup_logging(verbose)
    console = Console()

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Check if we need to look in a 'results' subdirectory
    if (results_path / "results").exists() and (results_path / "results").is_dir():
        # User passed the top-level directory, look in results subdirectory
        json_files = list((results_path / "results").glob("*.json"))
    else:
        # User passed the results directory directly
        json_files = list(results_path.glob("*.json"))

    if not json_files:
        logging.warning(f"No JSON files found in {results_dir}")
        if not check:
            return

    logging.info(f"Found {len(json_files)} result files")

    # If check or reschedule mode, also load the jobs.csv to compare
    if check or reschedule:
        jobs_csv_path = results_path / "jobs.csv"
        if not jobs_csv_path.exists():
            logging.warning(f"No jobs.csv found in {results_dir}, cannot perform check")
            check = False
            reschedule = False
        else:
            jobs_df = pd.read_csv(jobs_csv_path)
            logging.info(f"Found {len(jobs_df)} scheduled jobs in jobs.csv")

    # Collect results
    rows = []
    completed_jobs = set()  # Track (model, task, n_shot) tuples
    results_with_performance = (
        0  # Track how many results actually have performance data
    )

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract model name/path
            model_name = data.get("model_name", "unknown")

            # Extract results for each task
            results = data.get("results", {})
            n_shot_data = data.get("n-shot", {})

            for task_name, task_results in results.items():
                # Skip MMLU subtasks - only keep the aggregate score
                if task_name.startswith("mmlu_") and task_name != "mmlu":
                    continue

                # Get n_shot for this task
                n_shot = n_shot_data.get(task_name, "unknown")

                # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
                if task_name == "mmlu" and n_shot == "unknown":
                    for key, value in n_shot_data.items():
                        if key.startswith("mmlu_"):
                            n_shot = value
                            break

                # Get the primary metric (usually acc,none)
                performance = task_results.get("acc,none")
                if performance is None:
                    # Try other common metric names
                    for metric in ["acc", "accuracy", "f1", "exact_match"]:
                        if metric in task_results:
                            performance = task_results[metric]
                            break

                if performance is not None:
                    results_with_performance += 1

                    # Track completed job for check/reschedule mode (only if we have a result)
                    if check or reschedule:
                        completed_jobs.add((model_name, task_name, n_shot))

                    rows.append(
                        {
                            "model_name": model_name,
                            "task": task_name,
                            "n_shot": n_shot,
                            "performance": performance,
                        }
                    )
                else:
                    # Debug: log cases where we have a task but no performance metric
                    if verbose:
                        logging.debug(
                            f"No performance metric found for {model_name} | {task_name} | n_shot={n_shot} in {json_file.name}"
                        )

        except Exception as e:
            logging.warning(f"Failed to process {json_file}: {e}")
            if verbose:
                logging.exception(e)

    if not rows and not check:
        logging.warning("No results extracted from JSON files")
        return

    # Create DataFrame and save to CSV (if we have results)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        logging.info(f"Extracted {len(df)} evaluation results")

        # Print summary statistics
        if verbose:
            logging.info("\nSummary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    # Perform check analysis if requested
    if check or reschedule:
        logging.info("\n=== Evaluation Status Check ===")

        # Parse SLURM logs to get more detailed status
        slurm_logs_dir = results_path / "slurm_logs"
        attempted_jobs = set()  # Jobs that were attempted (started)
        failed_jobs = set()  # Jobs that crashed/failed

        if slurm_logs_dir.exists():
            # Parse .out files to find attempted jobs
            for out_file in slurm_logs_dir.glob("*.out"):
                try:
                    with open(out_file) as f:
                        content = f.read()
                        # Look for "Starting evaluation for:" patterns
                        import re

                        pattern = r"Starting evaluation for:\s*\n\s*Model: (.+)\s*\n\s*Task: (.+)\s*\n\s*N-shot: (\d+)"
                        matches = re.findall(pattern, content)
                        for model, task, n_shot in matches:
                            attempted_jobs.add(
                                (model.strip(), task.strip(), int(n_shot.strip()))
                            )

                        # Check if job finished successfully
                        if "Job" in content and "finished." in content:
                            # This array job completed successfully
                            pass
                        else:
                            # Job might have crashed - check for specific patterns
                            if (
                                "Traceback" in content
                                or "Error" in content
                                or "Exception" in content
                            ):
                                for model, task, n_shot in matches:
                                    failed_jobs.add(
                                        (
                                            model.strip(),
                                            task.strip(),
                                            int(n_shot.strip()),
                                        )
                                    )
                except Exception as e:
                    logging.debug(f"Error parsing {out_file}: {e}")

            # Parse .err files for errors
            for err_file in slurm_logs_dir.glob("*.err"):
                try:
                    file_size = err_file.stat().st_size
                    if file_size > 0:  # Non-empty error file
                        # Extract array task ID from filename
                        array_id_match = re.search(r"-(\d+)\.err$", err_file.name)
                        if array_id_match:
                            int(array_id_match.group(1))
                            # Find corresponding .out file to get job details
                            out_file = err_file.with_suffix(".out")
                            if out_file.exists():
                                with open(out_file) as f:
                                    content = f.read()
                                    pattern = r"Starting evaluation for:\s*\n\s*Model: (.+)\s*\n\s*Task: (.+)\s*\n\s*N-shot: (\d+)"
                                    matches = re.findall(pattern, content)
                                    for model, task, n_shot in matches:
                                        failed_jobs.add(
                                            (
                                                model.strip(),
                                                task.strip(),
                                                int(n_shot.strip()),
                                            )
                                        )
                except Exception as e:
                    logging.debug(f"Error parsing {err_file}: {e}")

        # Categorize incomplete jobs
        still_running_jobs = []  # Jobs that are likely still executing
        never_attempted_jobs = []
        crashed_jobs = []
        needs_rerun_jobs = []  # Jobs that definitely need to be rescheduled

        # We know we have exactly len(completed_jobs) completed jobs with actual results
        # The rest need to be categorized
        len(completed_jobs)

        for _, job in jobs_df.iterrows():
            job_tuple = (job["model_path"], job["task_path"], job["n_shot"])

            # Check if this job corresponds to one of our completed results
            # Use the same matching logic as before but don't over-count
            is_completed = False

            # Try to find a matching completed job
            if job_tuple in completed_jobs:
                is_completed = True
            else:
                # Try fuzzy matching
                for completed_job in completed_jobs:
                    completed_model, completed_task, completed_n_shot = completed_job

                    if (
                        job["n_shot"] == completed_n_shot
                        and job["task_path"] == completed_task
                        and (
                            str(job["model_path"]).endswith(completed_model)
                            or completed_model in str(job["model_path"])
                        )
                    ):
                        is_completed = True
                        break

            if is_completed:
                continue  # Skip completed jobs

            # Job is not completed, categorize it
            if job_tuple in failed_jobs:
                crashed_jobs.append(job)
                needs_rerun_jobs.append(job)
            elif job_tuple not in attempted_jobs:
                never_attempted_jobs.append(job)
                needs_rerun_jobs.append(job)  # These likely need rescheduling too
            else:
                # Job was attempted but not completed and didn't crash - likely still running
                still_running_jobs.append(job)

        needs_rerun_df = pd.DataFrame(needs_rerun_jobs)

        # Calculate completed jobs based on the jobs.csv perspective
        actual_completed_from_jobs = (
            len(jobs_df)
            - len(still_running_jobs)
            - len(crashed_jobs)
            - len(never_attempted_jobs)
        )

        logging.info(f"\nTotal scheduled jobs: {len(jobs_df)}")
        logging.info(
            f"Completed jobs (from scheduled jobs): {actual_completed_from_jobs}"
        )
        logging.info(f"Still running/pending: {len(still_running_jobs)}")
        logging.info(f"Failed/Crashed jobs: {len(crashed_jobs)}")
        logging.info(f"Never attempted: {len(never_attempted_jobs)}")
        logging.info(f"Jobs needing reschedule: {len(needs_rerun_jobs)}")

        if verbose:
            logging.info(f"Total CSV rows (results with performance data): {len(rows)}")
            logging.info(
                f"Unique completed jobs found in JSON files: {len(completed_jobs)}"
            )
            if len(completed_jobs) != actual_completed_from_jobs:
                logging.info(
                    f"Note: {len(completed_jobs)} results found vs {actual_completed_from_jobs} jobs matched from schedule"
                )

        if len(needs_rerun_jobs) > 0:
            if reschedule:
                # Show overview table in reschedule mode
                console.print("\n[bold cyan]🔄 Jobs Needing Reschedule[/bold cyan]")

                # Create summary table
                summary_table = Table(
                    show_header=True, header_style="bold magenta", box="ROUNDED"
                )
                summary_table.add_column("Status", style="bold")
                summary_table.add_column("Count", justify="right", style="cyan")

                summary_table.add_row("✅ Completed", str(actual_completed_from_jobs))
                summary_table.add_row("🏃 Still Running", str(len(still_running_jobs)))
                summary_table.add_row("❌ Crashed", str(len(crashed_jobs)))
                summary_table.add_row(
                    "⏭️  Never Attempted", str(len(never_attempted_jobs))
                )
                summary_table.add_row(
                    "[bold yellow]🔄 Need Reschedule[/bold yellow]",
                    f"[bold yellow]{len(needs_rerun_jobs)}[/bold yellow]",
                )

                console.print(summary_table)

                # Show detailed table of jobs to reschedule
                console.print("\n[bold cyan]📋 Detailed Job List[/bold cyan]")

                detail_table = Table(
                    show_header=True, header_style="bold magenta", box="ROUNDED"
                )
                detail_table.add_column("#", style="dim", width=4)
                detail_table.add_column("Status", style="bold", width=15)
                detail_table.add_column(
                    "Model", style="cyan", no_wrap=True, max_width=40
                )
                detail_table.add_column("Task", style="green", max_width=20)
                detail_table.add_column("n_shot", justify="right", style="yellow")

                # Show first 20 rows
                for idx, (_, job) in enumerate(needs_rerun_df.head(20).iterrows(), 1):
                    if (
                        job["model_path"],
                        job["task_path"],
                        job["n_shot"],
                    ) in failed_jobs:
                        status = "[red]❌ CRASHED[/red]"
                    else:
                        status = "[yellow]⏭️  NOT ATTEMPTED[/yellow]"

                    # Truncate long model paths for display
                    model_display = str(job["model_path"])
                    if len(model_display) > 40:
                        model_display = "..." + model_display[-37:]

                    detail_table.add_row(
                        str(idx),
                        status,
                        model_display,
                        str(job["task_path"]),
                        str(job["n_shot"]),
                    )

                if len(needs_rerun_jobs) > 20:
                    detail_table.add_row("...", "...", "...", "...", "...")
                    console.print(detail_table)
                    console.print(
                        f"\n[dim]Showing 20 of {len(needs_rerun_jobs)} jobs[/dim]"
                    )
                else:
                    console.print(detail_table)

                # Ask for confirmation
                console.print(
                    f"\n[bold]Total jobs to reschedule: {len(needs_rerun_jobs)}[/bold]"
                )

                import questionary
                from questionary import Style

                custom_style = Style(
                    [
                        ("qmark", "fg:#673ab7 bold"),
                        ("question", "bold"),
                        ("answer", "fg:#f44336 bold"),
                        ("pointer", "fg:#673ab7 bold"),
                        ("highlighted", "fg:#673ab7 bold"),
                        ("selected", "fg:#cc5454"),
                    ]
                )

                save_and_schedule = questionary.confirm(
                    "\nSave failed jobs CSV and schedule re-evaluation?",
                    default=True,
                    style=custom_style,
                ).ask()

                if save_and_schedule:
                    # Save the CSV
                    rerun_csv = output_csv.replace(".csv", "_needs_rerun.csv")
                    needs_rerun_df.to_csv(rerun_csv, index=False)
                    console.print(f"\n[green]✅ Jobs saved to: {rerun_csv}[/green]")

                    # Ask if they want to schedule now
                    schedule_now = questionary.confirm(
                        "\nSchedule these jobs now?",
                        default=True,
                        style=custom_style,
                    ).ask()

                    if schedule_now:
                        console.print("\n[yellow]To schedule these jobs, run:[/yellow]")
                        console.print(
                            f"[bold cyan]oellm schedule-eval --eval_csv_path {rerun_csv}[/bold cyan]"
                        )

            else:
                # Original behavior for check mode
                # Save jobs that need rescheduling
                rerun_csv = output_csv.replace(".csv", "_needs_rerun.csv")
                needs_rerun_df.to_csv(rerun_csv, index=False)
                logging.info(f"\nJobs needing reschedule saved to: {rerun_csv}")
                logging.info(
                    "You can re-run these with: oellm schedule-eval --eval_csv_path "
                    + rerun_csv
                )

                # Save crashed jobs separately if any
                if crashed_jobs:
                    crashed_csv = output_csv.replace(".csv", "_crashed.csv")
                    pd.DataFrame(crashed_jobs).to_csv(crashed_csv, index=False)
                    logging.info(f"Crashed jobs specifically saved to: {crashed_csv}")

                # Show some examples if verbose
                if verbose and len(needs_rerun_jobs) > 0:
                    logging.info("\nExample jobs needing reschedule:")
                    for _i, (_, job) in enumerate(needs_rerun_df.head(5).iterrows()):
                        if (
                            job["model_path"],
                            job["task_path"],
                            job["n_shot"],
                        ) in failed_jobs:
                            status = "CRASHED"
                        else:
                            status = "NEVER ATTEMPTED"
                        logging.info(
                            f"  - [{status}] {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                        )
                    if len(needs_rerun_jobs) > 5:
                        logging.info(f"  ... and {len(needs_rerun_jobs) - 5} more")

        if still_running_jobs and verbose:
            logging.info(
                f"\nNote: {len(still_running_jobs)} jobs appear to still be running/pending."
            )
            logging.info(
                "These were attempted but haven't completed yet. Check SLURM queue status."
            )


def main():
    auto_cli(
        {
            "schedule-eval": schedule_evals,
            "build-csv": build_csv,
            "collect-results": collect_results,
        },
        as_positional=False,
        description="OELLM: Multi-cluster evaluation tool for language models",
    )
