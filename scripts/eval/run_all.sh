#!/bin/bash
source scripts/eval/source.sh

MODEL_CKPT=$1

# ======== Check if NO_DOCKER is set ========
NO_DOCKER=$2  # default to 0 (false)
if [ -z "$NO_DOCKER" ]; then
    NO_DOCKER=0
    echo "Defaulting to NO_DOCKER=$NO_DOCKER (false). Will use docker for evaluation when needed."
else
    echo "NO_DOCKER=$NO_DOCKER - Will use *apptainer* for evaluation when needed."
fi

# ======== Find Port to Serve Model ========
PORT=8888
echo "Defaulting to PORT=$PORT"
TRIES=0
MAX_TRIES=1000
while is_port_in_use $PORT && [ $TRIES -lt $MAX_TRIES ]; do
    echo "Previously specified port is in use. Trying port $PORT... (try $TRIES/$MAX_TRIES)"
    PORT=$((PORT + 1))
    TRIES=$((TRIES + 1))
done
if [ $TRIES -eq $MAX_TRIES ]; then
    echo "Error: Could not find an unoccupied port after $MAX_TRIES tries."
    exit 1
fi
OPENAI_API_BASE="http://localhost:$PORT/v1"
echo "Will serve model from $MODEL_CKPT on OpenAI-compatible API: $OPENAI_API_BASE"


# ======== Check Env ========
check_conda_env_and_activate code-act-agent
echo "Remember to set CUDA_VISIBLE_DEVICES before running this script."

# ======== Handle Ctrl+C ========
function ctrl_c() {
    echo "Ctrl+C caught. Sending TERM signal to all scripts for cleanup..."

    # Iterate over all panes and send the TERM signal
    tmux list-panes -t "$SESSION_NAME" -F "#{pane_id}" | while IFS= read -r pane_id; do
        tmux send-keys -t "$pane_id" C-c
    done

    # Give scripts a few seconds to clean up (adjust as needed)
    sleep 5

    echo "Killing the tmux session..."
    tmux kill-session -t "$SESSION_NAME"
    kill_serving
    exit 1
}
trap ctrl_c INT

# Create a new tmux session and detach from it
SESSION_NAME="mint_agent_evaluation_${PORT}_$(hostname | sed 's/\./_/g')"

set -e

# # Create a temporary directory for the tmux socket (fix for ncsa slurm)
# TMUX_TMPDIR="$HOME/.tmux_tmp/$(hostname | sed 's/\./_/g')_${PORT}"
# if [ ! -d "$TMUX_TMPDIR" ]; then
#     mkdir -p $TMUX_TMPDIR
# fi
# echo $TMUX_TMPDIR
# tmux() {
#     command tmux -S $TMUX_TMPDIR/default "$@"
# }

# ======== Start Serving ========
tmux new-session -d -s $SESSION_NAME
# Start server in a named pane
echo "Starting server on port $PORT with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
OUTPUT_LOG_DIR=$MODEL_CKPT/eval/logs
tmux new-window -t $SESSION_NAME -n server_window "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES scripts/eval/serve/serve_ckpt.sh $MODEL_CKPT $PORT"
wait_for_model $PORT

# ======== Start Evaluation ========
# Start the evaluation in separate windows within the same tmux session
mkdir -p $OUTPUT_LOG_DIR
function start_window() {
    window_name=$1
    script_path=$2
    tmux new-window -t $SESSION_NAME -n $window_name "OPENAI_API_BASE=$OPENAI_API_BASE NO_DOCKER=$NO_DOCKER $script_path 2>&1 | tee -a $OUTPUT_LOG_DIR/$window_name.log"
}

start_window mmlu "scripts/eval/mmlu/mmlu.sh $MODEL_CKPT"
start_window gsm8k "scripts/eval/gsm8k/gsm8k.sh $MODEL_CKPT"
start_window human_eval "scripts/eval/human_eval/human_eval.sh $MODEL_CKPT"
DO_FEEDBACK=0
start_window mint-bench "scripts/eval/mint-bench/mint.sh $MODEL_CKPT $DO_FEEDBACK"
start_window miniwob++ "scripts/eval/miniwob++/miniwob++.sh $MODEL_CKPT"
start_window science-world "scripts/eval/science-world/science-world.sh $MODEL_CKPT"
start_window mt-bench "scripts/eval/mt-bench/mt-bench.sh $MODEL_CKPT"

echo "Started all scripts in separate windows within the '$SESSION_NAME' tmux session. Use 'tmux attach -t $SESSION_NAME' to view."


# Wait for all evaluation scripts to finish (excluding the server pane)
echo "Looping until all evaluation scripts finish..."
# Wait for all evaluation scripts to finish
# Function to check if all DONE files exist
TASKS=("mmlu" "gsm8k" "human_eval" "mint-bench" "miniwob++" "science-world" "mt-bench")  # "ds1000" "bbh" "webarena" "science-world-code-action"

# ======== Wait for all tasks to finish ========
all_tasks_done() {
    for task in "${TASKS[@]}"; do
        if [[ ! -f "$MODEL_CKPT/eval/$task/DONE" ]]; then
            # If the DONE file for any task is missing, return 1 (false)
            return 1
        fi
    done
    # All DONE files exist
    return 0
}
while ! all_tasks_done; do
    sleep 30
    # check if the server is still running, if not kill the seving and restart it
    check_model_is_served $PORT || (
        echo "Server is not running. Restarting..." && kill_serving && tmux kill-window -t "${SESSION_NAME}:server_window" && tmux new-window -t $SESSION_NAME -n server_window "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES scripts/eval/serve/serve_ckpt.sh $MODEL_CKPT $PORT" && wait_for_model $PORT
    )

done

# Kill the server
tmux send-keys -t "${SESSION_NAME}:server_window" C-c
# Kill the tmux session
tmux kill-session -t $SESSION_NAME

# ======== Aggregate Results ========
echo "Aggregating results..."
python3 scripts/eval/aggregate_eval.py $MODEL_CKPT
echo "Server has been stopped."
