#! /bin/bash

export CUDA_VISIBLE_DEVICES="2,3,4,5"

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/userhome/dataset/megatron/zhanghan/sample_100G_policy_3/Sample100GPolicy3_text_document
CHECKPOINT_PATH=/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 31 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /userhome/pclproject/gpt/Megatron-LM-1.1/megatron/tokenizer/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 100 \
       --save-interval 300 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --reset-attention-mask \
       --checkpoint-activations



set +x
