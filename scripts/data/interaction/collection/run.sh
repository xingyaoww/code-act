#!/bin/bash
ulimit -n 4096
export ALFWORLD_DATA=data/raw/alfworld
export PYTHONPATH=$(pwd):$PYTHONPATH
EXTRA_ARGS=$@
# env var DEBUG_MODE=1 # set to 1 to run in debug mode (no background processes)
# else run in background

# For debug
if (($DEBUG_MODE)); then
    RUN_IN_BACKGROUND=0
    N_PARALLEL=1
    EXTRA_ARGS="--debug $EXTRA_ARGS"
else
    RUN_IN_BACKGROUND=1
    N_PARALLEL=8
fi

# Create an array to store background process IDs
declare -a PIDS=()

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

function remove_from_array {
    local -n arr=$1
    local value=$2

    # Find the index of the element to be removed
    local index=0
    for i in "${arr[@]}"; do
        if [ "$i" = "$value" ]; then
            break
        fi
        index=$((index + 1))
    done

    # Remove the element at the found index
    unset 'arr[$index]'

    # Reindex the array
    arr=("${arr[@]}")
}

function run_config_glob() {
    # glob will be automatically expanded and passed as an array
    local config_array=("$@")
    local num_configs=${#config_array[@]} # Get the length of the array
    echo "Total number of configurations: $num_configs"
    if ((num_configs > 0)); then
        echo "Matched configurations:"
        for config in "${config_array[@]}"; do
            echo "- $config"
        done
    else
        echo "No configurations found for the given glob pattern."
    fi

    for config in "${config_array[@]}"; do
        echo "Running $config"
        output_dir=$(jq -r .output_dir $config)
        mkdir -p $output_dir
        echo "=== START DATE: $(date) ===" >>$output_dir/output.txt
        # only save the current GIT commit hash if git is available and this is a git repo
        if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
            echo "# GIT COMMIT: $(git rev-parse HEAD) ===" >>$output_dir/output.txt
        fi

        if (($RUN_IN_BACKGROUND)); then
            python -u mint/main.py --exp_config=$config $EXTRA_ARGS >>"$output_dir/output.txt" 2>&1 &

            cur_pid=$!
            echo -e "\n** Started process $cur_pid (run in background). To track progress, run:"
            echo -e "  tail -f $output_dir/output.txt"

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

        # Run in foreground
        else
            # also print to stdout
            python -u -m pdb -c continue mint/main.py --exp_config=$config $EXTRA_ARGS 2>&1 | tee -a $output_dir/output.txt
        fi
    done
}

# ========================

# TODO(user): Set your model_name here
MODEL_NAME=gpt-3.5-turbo-0613
# MODEL_NAME=claude-instant-1  # <- You can set this to different models


# TODO(user): Set to your configs here,
# make sure you follow README.md to generate config files for your model.
# This use glob patterns to run multiple configs at once.
# You can also specify the type of tasks you want to evaluate with
# For example: run_config_glob configs/$MODEL_NAME/F=None/max5_p2+tool+cd/decision_making/*.json

# (1) Tool-augmented Task-Solving

run_config_glob \
    mint/data_gen_configs/gpt-3.5-turbo-0613/F=None/max5_p2+tool+cd/reasoning/**/train.json

run_config_glob \
    mint/data_gen_configs/gpt-3.5-turbo-0613/F=None/max5_p2+tool+cd/decision_making/**/train.json

# specifically use 16k for code
run_config_glob \
    mint/data_gen_configs/gpt-3.5-turbo-16k-0613/F=None/max5_p2+tool+cd/code_generation/APPS/train.json

# Use GPT-4 for unsolved problems
# MODEL_NAME=gpt-4-0613
# EXTRA_ARGS="--n_max_tasks 300"
# run_config_glob \
#     mint/data_gen_configs/gpt-4-0613/F=None/max5_p2+tool+cd/code_generation/APPS/train.json

# ========================
# wait for all background processes to finish before exiting
for pid in "${PIDS[@]}"; do
    wait $pid
done
