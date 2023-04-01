#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"

CHECKPOINT_PATH=/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged/

python tools/generate_samples_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 31 \
       --hidden-size 2560 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file /userhome/pclproject/gpt/Megatron-LM-1.1/megatron/tokenizer/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --genfile unconditional_samples.json \
       --num-samples 0 \
       --top_p 0.9 \
       --recompute \
       --no-load-rng \
       --sample-input-file ""
