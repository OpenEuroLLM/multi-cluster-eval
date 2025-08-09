# OpenEuroLLM CLI (oellm)

A package for running OELLM CLI workflows across multiple HPC clusters using SLURM job arrays and Singularity containers. 

## Currently supported workflows
- Schedule evaluations on multiple models and tasks on all clusters ✅
- Restart failed evaluations (e.g., due to node failures) ✅
- Interactive eval job/csv builder ✅

## Planned workflows
- Sync and download evaluation results from all clusters via a shared data layer
- Schedule training jobs on all clusters
- Schedule conversions from MegatronLM to HuggingFace

## Quick Example

**Prerequisites:**
- install [uv](https://docs.astral.sh/uv/#installation)

```bash
# Install the package
uv tool install git+https://github.com/OpenEuroLLM/oellm-cli.git

# Run evaluations on multiple models and tasks
oellm schedule-eval \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n_shot 5
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, or JURECA)
- Download and cache the specified models and datasets
- Generate a SLURM job array to evaluate all model-task combinations
- Submit the jobs with appropriate cluster-specific resource allocations

In case you meet HuggingFace quotas issues, make sure you are logged in by setting your `HF_TOKEN` and that you are part of [OpenEuroLLM](https://huggingface.co/OpenEuroLLM) organization. 

## Interactive CSV Builder

```bash
oellm interactive-csv
```

This will launch an interactive workflow where you can:
- Add models (HuggingFace Hub or local paths)
- Select evaluation tasks
- Configure n-shot settings
- Preview and save your evaluation configuration

Otherwise you can also directly schedule using a CSV file:
```bash
oellm schedule-eval --eval_csv_path custom_evals.csv
```

## Installation

### JURECA/JSC Specifics

Due to the limit space in `$HOME` on JSC clusters, you must set these `uv` specific environment variables to avoid running out of space:

```bash
export UV_CACHE_DIR="<some-workspace-dir>/.cache/uv-cache"
export UV_INSTALL_DIR="<some-workspace>/.local"
export UV_PYTHON_INSTALL_DIR="<some-workspace>/.local/share/uv/python"
export UV_TOOL_DIR="<some-workspace-dir>/.cache/uv-tool-cache"
```

You can set these variables in your `.bashrc` or `.zshrc` file, depending on your shell of preference.

E.g., I have a user-folder in the `synthlaion` project, so I set the following variables:
```bash
export UV_CACHE_DIR="/p/project1/synthlaion/$USER/.cache/uv-cache"
export UV_INSTALL_DIR="/p/project1/synthlaion/$USER/.local"
export UV_PYTHON_INSTALL_DIR="/p/project1/synthlaion/$USER/.local/share/uv/python"
export UV_TOOL_DIR="/p/project1/synthlaion/$USER/.cache/uv-tool-cache"
```

### General Installation

Install directly from the git repository using uv:

```bash
uv tool install git+https://github.com/OpenEuroLLM/oellm-cli.git
```

This makes the `oellm` command available globally in your shell.

If you've already installed the package, you can run the following command to update it:
```bash
uv tool upgrade oellm
```

If you had previously installed the package from a different source and would like to overwrite it, you can run the following command:
```bash
uv tool install git+https://github.com/OpenEuroLLM/oellm-cli.git --force
```

## High-Level Evaluation Workflow

The `oellm` package orchestrates distributed LLM evaluations through the following workflow:

### 1. **Cluster Auto-Detection**
- Automatically detects the current HPC cluster based on hostname patterns
- Loads cluster-specific configurations from [`clusters.yaml`](oellm/clusters.yaml) including:
  - SLURM partition and account settings
  - Shared storage paths for models, datasets, and results
  - GPU allocation and queue limits
  - Singularity container specifications

### 2. **Resource Preparation**
- **Model Handling**: Processes both local model checkpoints and Hugging Face Hub models
  - For local paths: Automatically discovers and expands training checkpoint directories
  - For HF models: Pre-downloads to shared cache (`$HF_HOME`) for offline access on compute nodes
- **Dataset Caching**: Pre-downloads all evaluation datasets using lm-evaluation-harness TaskManager
- **Container Management**: Ensures the appropriate Singularity container is available for the target cluster

### 3. **Job Generation & Scheduling**
- Creates a comprehensive CSV manifest of all model-task-shot combinations
- Generates a SLURM batch script from a template with cluster-specific parameters
- Submits a job array where each array task processes a subset of evaluations
- Respects queue limits and current user load to avoid overwhelming the scheduler

### 4. **Distributed Execution**
- Each SLURM array job runs in a Singularity container with:
  - GPU access (NVIDIA CUDA or AMD ROCm as appropriate)
  - Mounted shared storage for models, datasets, and output
  - Offline execution using pre-cached resources
- Uses `lm-evaluation-harness` for the actual model evaluation
- Outputs results as JSON files

### 5. **Output Organization**
Results are organized in timestamped directories under `$EVAL_OUTPUT_DIR/$USER/`:
```
2024-01-15-14-30-45/
├── jobs.csv              # Complete evaluation manifest
├── submit_evals.sbatch    # Generated SLURM script
├── slurm_logs/           # SLURM output/error logs
└── results/              # Evaluation JSON outputs
```

## Supported Clusters

Currently supports three HPC clusters:

- **LEONARDO** - NVIDIA A100 GPUs (CUDA)
- **LUMI** - AMD MI250X GPUs (ROCm)
- **JURECA** - NVIDIA A100 GPUs (CUDA)

Each cluster has pre-configured:
- Shared evaluation directories with appropriate quotas
- Optimized Singularity containers with evaluation dependencies
- Account and partition settings for the OpenEuroLLM project

## Development and Testing
Run in download-only mode to prepare resources without submitting jobs:

```bash
oellm schedule-eval --models "EleutherAI/pythia-160m" --tasks "hellaswag" --n_shot 0 --download_only True
```
