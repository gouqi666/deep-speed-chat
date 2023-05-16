#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT='./output/llama-7b'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ZERO_STAGE=3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT
export TRAIN_LLAMA='1'
deepspeed train_llama_rm.py \
   --data_path single_turn_rlhf \
   --local_data_files /nfs2/wzt/deep-speed-chat/datasets/single_turn_rlhf \
   --data_split 2,4,4 \
   --model_name_or_path /nfs2/wzt/models/llama-7b \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
#    &> $OUTPUT/training.log
