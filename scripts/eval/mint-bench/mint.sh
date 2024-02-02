#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1  # data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52
DO_FEEDBACK=$2  # 0 or 1
# default to 0
DO_FEEDBACK=${DO_FEEDBACK:-0}
echo "DO_FEEDBACK=$DO_FEEDBACK"

OUTPUT_DIR=$OUTPUT_DIR/eval/mint-bench
check_is_done $OUTPUT_DIR

set -xe
# Copy the mint-bench repo to a temporary directory
TMP_MINT_DIR_ROOT=.tmp-eval/$OUTPUT_DIR
# if the tmp mint dir not exist, create it
if [ ! -d "$TMP_MINT_DIR_ROOT" ]; then
    mkdir -p $TMP_MINT_DIR_ROOT
    # Copy the mint-bench repo to a temporary directory
    cp -r scripts/eval/mint-bench/mint-bench $TMP_MINT_DIR_ROOT
fi

TMP_MINT_DIR=$TMP_MINT_DIR_ROOT/mint-bench

# Override the default config_variables.py with the one for mint-bench
rm $TMP_MINT_DIR/mint/configs/config_variables.py
cp scripts/eval/mint-bench/config_variables.py $TMP_MINT_DIR/mint/configs/config_variables.py

# Override the default run.sh with the pre-specified one for mint-bench
rm $TMP_MINT_DIR/scripts/run.sh
cp scripts/eval/mint-bench/mint-bench-run.sh $TMP_MINT_DIR/scripts/run.sh

# Generate the config file
pushd $TMP_MINT_DIR
check_var "OPENAI_API_BASE"
PYTHONPATH=`pwd`:$PYTHONPATH python3 mint/configs/generate_config.py

# Run the eval harness
mkdir -p data/outputs
mkdir -p data/processed

# check if NO_DOCKER == 1
if [ "$NO_DOCKER" -eq "1" ]; then
    echo "NO_DOCKER=$NO_DOCKER - Using apptainer for mint evaluation."

    # TODO: set APPTAINER_IMG
    # by running `apptainer pull docker://xingyaoww/mint-bench:v1.0`
    APPTAINER_IMG=/projects/bcbf/xingyao6/apptainer-images/mint-bench_v1.0.sif

    # Check if the apptainer image exists
    if [ ! -f "$APPTAINER_IMG" ]; then
        echo "Apptainer image $APPTAINER_IMG does not exist. Please run 'apptainer pull docker://xingyaoww/mint-bench:v1.0' to pull it and modify the path in this script if needed."
        exit 1
    fi

    apptainer run --nv \
        --no-home \
        --no-mount bind-paths \
        --cleanenv \
        --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
        --env "BARD_API_KEY=$BARD_API_KEY" \
        --env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
        --env "DO_FEEDBACK=$DO_FEEDBACK" \
        --writable-tmpfs \
        --bind `pwd`:/mint-bench:ro \
        --bind `pwd`/data/outputs:/mint-bench/data/outputs:rw \
        --bind `pwd`/data/processed:/mint-bench/data/processed:rw \
        $APPTAINER_IMG \
        /bin/bash -c "cd /mint-bench && ./scripts/run.sh"

else
    echo "NO_DOCKER=$NO_DOCKER - Using docker for mint evaluation."
    export HOST_USER_ID=$(id -u)
    DOCKER_IMG="xingyaoww/mint-bench:v1.0"

    # Construct instance name using the current username and the current time.
    # This is useful for running multiple instances of the docker container.
    DOCKER_INSTANCE_NAME="mint_${USER}_$(date +%Y%m%d_%H%M%S)"

    docker run \
        -e HOST_USER_ID \
        -e OPENAI_API_KEY \
        -e BARD_API_KEY \
        -e ANTHROPIC_API_KEY \
        -e DO_FEEDBACK \
        -v `pwd`:/mint-bench:ro \
        -v `pwd`/data/outputs:/mint-bench/data/outputs:rw \
        -v `pwd`/data/processed:/mint-bench/data/processed:rw \
        --network host \
        --rm \
        --name $DOCKER_INSTANCE_NAME \
        $DOCKER_IMG \
        bash -c "useradd --shell /bin/bash -u $HOST_USER_ID -o -c '' -m mint && ulimit -n 4096 && su - mint -c 'cd /mint-bench && ./scripts/run.sh'"
fi

# Copy the results to the output directory
popd

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists. Removing it..."
    rm -r $OUTPUT_DIR
fi
mkdir -p $OUTPUT_DIR/outputs
mkdir -p $OUTPUT_DIR/results
mv $TMP_MINT_DIR/data/outputs/code-act-agent $OUTPUT_DIR/outputs/code-act-agent

# reset the tmp mint dir
rm -r $TMP_MINT_DIR_ROOT

python3 scripts/eval/mint-bench/convert_outputs.py \
    --data_dir $OUTPUT_DIR/outputs \
    --output_dir $OUTPUT_DIR/results

# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE
