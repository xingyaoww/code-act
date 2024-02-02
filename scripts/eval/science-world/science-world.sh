#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1 # "data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52"
MODEL_NAME="code-act-agent"

OUTPUT_DIR=$OUTPUT_DIR/eval/science-world
check_is_done $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

check_var "OPENAI_API_BASE"

export OPENAI_API_KEY="DUMMY"

# Convert OUTPUT_DIR to absolute path
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

pushd scripts/eval/science-world/
set -e

# ========================
# Create an array to store background process IDs
declare -a PIDS=()
N_PARALLEL=8

# Add this line at the beginning of the script to import the signal module
trap 'kill_background_processes' INT

function kill_background_processes() {
    echo -e '\nReceived Ctrl+C. Killing background processes...'

    # Loop through the stored background process IDs and kill them
    for pid in "${PIDS[@]}"; do
        if ps -p $pid >/dev/null; then
            echo "Killing process $pid"
            kill $pid
        fi
    done

    # Exit the script
    exit 1
}

# ========================
mkdir -p $OUTPUT_DIR/logs
for task in {0..29}
do
    # if $OUTPUT_DIR/task${task}-score.txt exists, skip this task
    if [ -f "$OUTPUT_DIR/task${task}-score.txt" ]; then
        echo "Task ${task} already evaluated. Skipping..."
        continue
    fi

    python eval.py \
        --task_nums $task \
        --output_path $OUTPUT_DIR \
        --model_name $MODEL_NAME 2>&1 | tee -a $OUTPUT_DIR/logs/$task.log &

    cur_pid=$!
    echo -e "\n** Started process $cur_pid (run in background). \nCheck $OUTPUT_DIR/logs/$task.log for progress."

    # Store the background process ID in the array
    PIDS+=("$cur_pid")
    # 2>&1 | tee -a $output_dir/output.txt

    # Control the number of parallel processes by waiting for some to finish
    # Adjust the value after -le to set the desired number of parallel processes
    while ((${#PIDS[@]} >= N_PARALLEL)); do
        for pid in "${PIDS[@]}"; do
            if ! ps -p "$pid" >/dev/null; then
                # Remove the finished process from the array
                echo "Process $pid finished. Remaining processes: ${PIDS[@]}"
                remove_from_array PIDS "$pid"
            fi
        done
        # Sleep for a short time before checking again
        sleep 1
    done
done

# ========================
# wait for all background processes to finish before exiting
for pid in "${PIDS[@]}"; do
    wait $pid
done

python3 metrics.py \
    --input_dir $OUTPUT_DIR

popd

# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE

