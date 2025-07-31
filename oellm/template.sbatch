#!/bin/bash
#SBATCH --job-name=oellm-eval
#SBATCH --time=$TIME_LIMIT
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --output={log_dir}/%x-%A-%a.out
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --error={log_dir}/%x-%A-%a.err
#SBATCH --array=0-{array_limit}%{max_array_len}


CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}

# avoiding crashes due to compute nodes not having access to the internet
export HF_HOME=$HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1

# Path to the shared Singularity image that contains all runtime deps
export EVAL_SIF_PATH="$EVAL_BASE_DIR/$EVAL_CONTAINER_IMAGE"

echo "Running eval on $CSV_PATH with $NUM_JOBS concurrent jobs on $SBATCH_PARTITION with $SBATCH_GPUS_PER_NODE GPUs per node"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

TOTAL_EVALS=$(($(wc -l < "$CSV_PATH") - 1))
echo "Total evaluations to run: $TOTAL_EVALS"

if [ "$TOTAL_EVALS" -lt 1 ]; then
    echo "No evaluations to run. Exiting."
    exit 0
fi

EVALS_PER_JOB=$(((TOTAL_EVALS + NUM_JOBS - 1) / NUM_JOBS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * EVALS_PER_JOB + 1))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * EVALS_PER_JOB))

if [ "$END_INDEX" -gt "$TOTAL_EVALS" ]; then
    END_INDEX=$TOTAL_EVALS
fi

echo "This job will process evaluations from index $START_INDEX to $END_INDEX."

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "No evaluations for this job to run. Exiting."
    exit 0
fi

# Use `tail` and `head` to slice the CSV file for the tasks assigned to this job.
# The +1 on START_INDEX accounts for the header row.
tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)) | \
while IFS=, read -r model_path task_path n_shot
do
    # Remove trailing carriage returns if script is edited on Windows
    model_path=$(echo "$model_path" | tr -d '\r')
    task_path=$(echo "$task_path" | tr -d '\r')
    n_shot=$(echo "$n_shot" | tr -d '\r')

    # Skip empty lines
    if [ -z "$model_path" ]; then
        continue
    fi

    echo "----------------------------------------------------"
    echo "Starting evaluation for:"
    echo "  Model: $model_path"
    echo "  Task: $task_path"
    echo "  N-shot: $n_shot"
    echo "----------------------------------------------------"

    # Build bind paths: always mount the shared eval directory, and additionally
    # bind the directory that contains the model checkpoint when the path exists
    BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR"

    if [ -e "$model_path" ]; then
        # If the model_path is a file, bind its parent directory; otherwise bind the dir itself
        MODEL_DIR="$model_path"
        if [ ! -d "$MODEL_DIR" ]; then
            MODEL_DIR="$(dirname "$MODEL_DIR")"
        fi
        # Avoid adding duplicate bind if it is already covered by EVAL_BASE_DIR
        if [[ $MODEL_DIR != $EVAL_BASE_DIR* ]]; then
            BIND_PATHS="$BIND_PATHS,$MODEL_DIR:$MODEL_DIR"
        fi
    fi


    singularity exec $SINGULARITY_ARGS \
        --bind $BIND_PATHS \
        $EVAL_SIF_PATH \
        python -m lm_eval --model hf \
            --model_args pretrained="$model_path",trust_remote_code=True \
            --tasks "$task_path" \
            --num_fewshot "$n_shot" \
            --output_path "{evals_dir}/$(openssl rand -hex 5).json" \
            --trust_remote_code \
            --log_samples

    echo "Evaluation finished for model: $model_path"

done

echo "Job $SLURM_ARRAY_TASK_ID finished."
