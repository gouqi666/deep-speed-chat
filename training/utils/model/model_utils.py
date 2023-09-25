# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    AutoModelForCausalLM,
AutoModelForSequenceClassification
)

from transformers.deepspeed import HfDeepSpeedConfig

from .llama_reward_model import LlamaRewardModel
from .reward_model import RewardModel


def  create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False):
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration

    # if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
    #     dschf = HfDeepSpeedConfig(ds_config)
    # else:
    #     dschf = None

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model_config.dropout = 0.0
    print(f'rlhf_training: {rlhf_training}')
    if rlhf_training:
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_config(model_config)
        if os.path.exists(os.path.join(model_name_or_path, 'llama_model.pt')):
            state_dict = torch.load(os.path.join(model_name_or_path, 'llama_model.pt'),
                                    map_location='cpu')  # ,pytorch_model.bin  llama_model.pt
        elif os.path.exists(os.path.join(model_name_or_path, 'checkpoints.pt')):
            state_dict = torch.load(os.path.join(model_name_or_path, 'checkpoints.pt'),
                                    map_location='cpu')
        else:
            state_dict = model_class.from_pretrained(model_name_or_path).state_dict()
        # reward model training
        if model_class is AutoModel:
            for k, v in list(state_dict.items()):
                if k.startswith('model.'):
                    state_dict[k.replace('model.', '')] = v
                    del state_dict[k]
        model.load_state_dict( state_dict, strict=False)
            # else: # sft training or actor model in ppo training
            #     model = model_class.from_pretrained(model_name_or_path)
            # st1 = torch.load(os.path.join(model_name_or_path, 'pytorch_model-00001-of-00002.bin'), # or pytorch_model.bin
            #                  map_location='cpu')
            # st2 = torch.load(os.path.join(model_name_or_path, 'pytorch_model-00002-of-00002.bin'),
            #                  map_location='cpu')
            # state_dict = st1
            # for k, v in st2.items():
            #     state_dict[k] = v
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id # model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil((tokenizer.num_vocab if hasattr(tokenizer, "num_vocab") else len(tokenizer))/ 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        ):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule


    reward_model_class = LlamaRewardModel
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training)
    critic_model = reward_model_class(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    if rlhf_training:# 会重新load reward model 和 critic model

        # use our trained reward model

        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'),strict=False)
    return critic_model