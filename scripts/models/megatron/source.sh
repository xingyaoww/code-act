#!/bin/bash

function train() {
    echo "================================"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S") Host:$(hostname)"
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tp)
            TP="$2"
            echo "TP=$TP"
            shift # past argument
            shift # past value
            ;;
            --pp)
            PP=$2
            echo "PP=$PP"
            shift # past argument
            shift # past value
            ;;
            --lr)
            LR=$2
            echo "LR=$LR"
            shift # past argument
            shift # past value
            ;;
            --n_warmup_steps)
            N_WARMUP_STEPS=$2
            echo "N_WARMUP_STEPS=$N_WARMUP_STEPS"
            shift # past argument
            shift # past value
            ;;
            --seq_length)
            SEQ_LENGTH=$2
            echo "SEQ_LENGTH=$SEQ_LENGTH"
            shift # past argument
            shift # past value
            ;;
            --model_name)
            MODEL_NAME=$2
            echo "MODEL_NAME=$MODEL_NAME"
            shift # past argument
            shift # past value
            ;;
            --data_name)
            DATA_NAME=$2
            echo "DATA_NAME=$DATA_NAME"
            shift # past argument
            shift # past value
            ;;
            --n_examples)
            N_EXAMPLES=$2
            echo "N_EXAMPLES=$N_EXAMPLES"
            shift # past argument
            shift # past value
            ;;
            --global_batch_size)
            GLOBAL_BATCH_SIZE=$2
            echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
            shift # past argument
            shift # past value
            ;;
            --n_epochs)
            N_EPOCHS=$2
            echo "N_EPOCHS=$N_EPOCHS"
            shift # past argument
            shift # past value
            ;;
            --n_gpu_per_node)
            N_GPU_PER_NODE=$2
            echo "N_GPU_PER_NODE=$N_GPU_PER_NODE"
            shift # past argument
            shift # past value
            ;;
            --extra_exp_note)
            EXTRA_EXP_NOTE=$2
            echo "EXTRA_EXP_NOTE=$EXTRA_EXP_NOTE"
            shift # past argument
            shift # past value
            ;;
            --load_model_ckpts)
            LOAD_MODEL_CKPTS=$2
            echo "LOAD_MODEL_CKPTS=$LOAD_MODEL_CKPTS"
            shift # past argument
            shift # past value
            ;;
            --recompute_num_layers)
            RECOMPUTE_NUM_LAYERS=$2
            echo "RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS"
            shift # past argument
            shift # past value
            ;;
            --exp_id)
            EXP_ID=$2
            echo "EXP_ID=$EXP_ID"
            shift # past argument
            shift # past value
            ;;
            *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
        esac
    done
    # if RECOMPUTE_NUM_LAYERS is none, default to 8
    if [ -z "$RECOMPUTE_NUM_LAYERS" ]; then
        RECOMPUTE_NUM_LAYERS=8
    fi

    echo "POSITIONAL_ARGS=${POSITIONAL_ARGS[*]}"

    echo "================================"

    # Automatically set the following variables
    if [ -z "$EXP_ID" ]; then
        EXP_ID="${MODEL_NAME}-t${TP}-p${PP}-${DATA_NAME}-lr${LR}-warmup${N_WARMUP_STEPS}-bs${GLOBAL_BATCH_SIZE}-seq${SEQ_LENGTH}-ep${N_EPOCHS}"
    fi
    if [ ! -z "$EXTRA_EXP_NOTE" ]; then
        EXP_ID="${EXP_ID}-N${EXTRA_EXP_NOTE}"
    fi
    MODEL_CKPT_DIR=data/ckpts/${EXP_ID}

    # first check LOAD_MODEL_CKPTS, if it is not empty, then we load the model from the ckpt dir
    # otherwise, if there is a model in the ckpt dir (check whether $MODEL_CKPT_DIR/latest_checkpointed_iteration.txt exists)
    # then we load the model from the ckpt dir

    if [ ! -z "$LOAD_MODEL_CKPTS" ]; then
        echo "Loading model from specified ckpt dir $LOAD_MODEL_CKPTS"
        MODEL_WEIGHT_DIR=$LOAD_MODEL_CKPTS
    elif [ -f "$MODEL_CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
        echo "Found existing ckpt dir $MODEL_CKPT_DIR, loading from there"
        MODEL_WEIGHT_DIR=$MODEL_CKPT_DIR
    else
        echo "No existing ckpt dir $MODEL_CKPT_DIR, loading from pretrained model"
        MODEL_WEIGHT_DIR=data/models/sharded/${MODEL_NAME}-megatron-tp${TP}-pp${PP}
    fi

    TOKENIZER_PATH=data/models/raw/${MODEL_NAME}-megatron/tokenizer.model
    DATA_PREFIX=data/megatron_format/${DATA_NAME}/data
    echo "EXP_ID=$EXP_ID"
    echo "MODEL_CKPT_DIR=$MODEL_CKPT_DIR"
    echo "MODEL_WEIGHT_DIR=$MODEL_WEIGHT_DIR"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "DATA_PREFIX=$DATA_PREFIX"
    export WANDB_NAME=$EXP_ID

    N_BATCHES_PER_EPOCH=$((($N_EXAMPLES/$GLOBAL_BATCH_SIZE)+1))
    TRAIN_ITERATIONS=$(($N_BATCHES_PER_EPOCH*$N_EPOCHS))
    # SAVE_INTERVAL=$N_BATCHES_PER_EPOCH  # per epoch
    SAVE_INTERVAL=$(($N_BATCHES_PER_EPOCH))
    echo "N_BATCHES_PER_EPOCH=$N_BATCHES_PER_EPOCH"
    echo "TRAIN_ITERATIONS=$TRAIN_ITERATIONS"

    # ==== ARGS ====
    LOG_ARGS="--log_interval 1 --save_interval $SAVE_INTERVAL --eval_interval 100000000 --split 100,0,0"
    TRAIN_ARGS="--train_iters $TRAIN_ITERATIONS --lr_decay_style cosine --lr_warmup_iters $N_WARMUP_STEPS --lr $LR --min_lr 1e-6"
    DISTRIBUTED_ARGS="--nproc_per_node $N_GPU_PER_NODE --nnodes 1 --node_rank 0 --master_addr localhost --master_port 9999"

    # ==== Training ====
    # Megatron model type map Llama-2-7b to llama2, And Mistral-7b to mistral
    if [[ $MODEL_NAME == Llama-2* ]]; then
        MODEL_TYPE=llama2
    elif [ "$MODEL_NAME" == "Mistral-7b" ]; then
        MODEL_TYPE=mistral
    else
        echo "Unknown model name $MODEL_NAME"
        exit 1
    fi
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    torchrun $DISTRIBUTED_ARGS Megatron-LLM/finetune.py \
        --tensor_model_parallel_size $TP \
        --pipeline_model_parallel_size $PP \
        --load $MODEL_WEIGHT_DIR \
        --save $MODEL_CKPT_DIR \
        --wandb_logger \
        --wandb_project code-act-agent \
        --data_path $DATA_PREFIX \
        --model_name $MODEL_TYPE \
        --tokenizer_type SentencePieceTokenizer \
        --vocab_file=$TOKENIZER_PATH \
        --vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
        --no_new_tokens \
        --bf16 \
        --micro_batch_size 1 \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        --seq_length $SEQ_LENGTH \
        --sequence_parallel \
        --recompute_granularity full \
        --recompute_method uniform \
        --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
        --data_type instruction \
        --variable_seq_lengths \
        --no_bias_gelu_fusion \
        --no_bias_dropout_fusion \
        --hidden_dropout 0.0 \
        --attention_dropout 0.0 \
        --use_checkpoint_args \
        --use_distributed_optimizer \
        $COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS \
        ${POSITIONAL_ARGS[*]}

}
