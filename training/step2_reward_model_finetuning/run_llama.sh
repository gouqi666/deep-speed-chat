#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team
OUTPUT='/mnt/gouqi/deep-speed-chat/training/step2_reward_model_finetuning/output/llama-7b'
# /mnt/data01/shenyan/ckpt/llama_hf/llama-sft-7b/
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
deepspeed --master_port=4243 main.py \
   --data_path Dahoas/synthetic-instruct-gptj-pairwise \
   --local_data_files /mnt/gouqi/deep-speed-chat/datasets/synthetic-instruct-gptj-pairwise \
   --data_split 2,4,4 \
   --num_padding_at_beginning 1 \
   --model_name_or_path /mnt/public/checkpoint/SFT/llama-sft-7b \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 1000 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log