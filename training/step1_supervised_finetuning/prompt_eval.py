# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import logging
import torch
import sys
import os
from tqdm import tqdm


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(
    os.path.abspath('../../'))
print(sys.path)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils.model.model_utils import create_hf_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        default='/mnt/public/checkpoint/SFT/llama-sft-7b',
        help="Path to baseline model",
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        default='/mnt/gouqi/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor',
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=5,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    parser.add_argument("--dataset-name",
                        type=str,
                        default="Dahoas/synthetic-instruct-gptj-pairwise",
                        )
    parser.add_argument("--output-path",
                        type=str,
                        default="./output",
                        )
    parser.add_argument("--local-data-files",
                        type=str,
                        default="/home/gq/deeplang/deep-speed-chat/datasets/synthetic-instruct-gptj-pairwise",
                       )

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens,
                                  top_p=0.9,
                                  top_k=30,
                                  temperature=1.0)
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                                                                                                                                                                                                                           clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("==========Baseline: Beam Search=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=args.num_beams,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune: Beam Search=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=args.num_beams,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        item = {}
        item['prompt'] = prompt[prompt.find('Instruction:')+13:prompt.find('Response:')-6]
        item['sft-7b'] = []
        for r in r_base:
            item['sft-7b'].append(r[r.find('Response:')+10:])
        item['after-ppo'] = []
        for r in r_finetune_g:
            item['after-ppo'].append(r[r.find('Response:')+10:])
        results.append(item)
    return results
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        # print("====================prompt end=============================")
        # print()
        # print()

def create_eval_prompt(args):
    from utils.data.data_utils import get_raw_dataset
    raw_dataset = get_raw_dataset(args.dataset_name, args.output_path, 1234, -1, local_path=args.local_data_files)
    return raw_dataset



def main():
    args = parse_args()
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")


    # test DLM-7b-multiturn
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig,set_seed
    set_seed(42)
    config = LlamaConfig.from_pretrained('/mnt/data01/shenyan/ckpt/llama_hf/DLM-7b-multiturn')
    config.rope_type = 'ours'
    model = LlamaForCausalLM(config)
    print(model.config)
    model.load_state_dict(torch.load('/mnt/data01/shenyan/ckpt/llama_hf/DLM-7b-multiturn/llama_model.pt'), strict=False)
    import sys
    sys.path.append('/home/gq/deeplang/deep-speed-chat/MixedTokenizer')
    from mixed_tokenizer import MixedLLaMATokenizer
    tokenizer_dir = "/home/gq/deeplang/deep-speed-chat/MixedTokenizer/tokenizer_files"

    tokenizer = MixedLLaMATokenizer(
        "{}/tokenizer_llama_en.model".format(tokenizer_dir),
        "{}/tokenizer_llama_zh.json".format(tokenizer_dir)
    )
    instruction = '为什么高端就业机会常常偏向男性？     '
    t1 = tokenizer.encode(f"### 用户(User):\n{instruction}\n", bos=True, eos=False)
    t2 = tokenizer.encode("### 助手(Assistant):\n", eos=False, bos=False)
    input = t1 + t2
    input = torch.LongTensor(input).unsqueeze(0)
    o1 = model.generate(input,max_length=512,num_beams=3,do_sample=True,temperature=0.9,top_p=0.95)
    print(tokenizer.decode(o1[0].tolist()))
    o1 = model.generate(input,max_length=512,num_beams=3,do_sample=True,temperature=1.0,top_p=0.95)
    print(tokenizer.decode(o1[0].tolist()))
    o1 = model.generate(input,max_length=512,num_beams=3,do_sample=True,temperature=1.1,top_p=0.95)
    print(tokenizer.decode(o1[0].tolist()))
    o1 = model.generate(input,max_length=512,num_beams=3,do_sample=True,temperature=2.0,top_p=0.95)
    print(tokenizer.decode(o1[0].tolist()))
    exit()




    config = AutoConfig.from_pretrained(args.model_name_or_path_baseline)
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path_baseline, padding_side='right', truncation_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, ds_config=None, rlhf_training=True)

    model_ckpt_path = os.path.join(args.model_name_or_path_finetune, 'pytorch_model.bin')
    assert os.path.exists(
        model_ckpt_path
    ), f"Cannot find model checkpoint at {model_ckpt_path}"

    model_fintuned.load_state_dict(
        torch.load(model_ckpt_path, map_location='cpu'),strict=False)

    model_baseline.to(device)
    model_fintuned.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        # prompt_dataset = create_eval_prompt(args)

        prompts = [
            "What is the capital city of China?",
            "What is the capital of China and where is it located?",
            "Is Beijing the capital city of China?",
            "Do you prefer red apples or green apples?",
            "What's the weather like today?"
        ]
        prompts = [sft_format.format(x) for x in prompts]
    elif args.language == "Chinese":
        sft_format = "下面的指令描述了一个需要完成的任务，请编写一个回复来合理地完成请求。\n\n### 指令：\n{}\n\n### 回复：\n"
        # prompts = [
        #     "Human: 请用几句话介绍一下微软? Assistant:",
        #     "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
        #     "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
        #     "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
        #     "Human: 鸟类为什么要南迁过冬? Assistant:"
        # ]
        honesty_prompt = []
        harmlessness_prompt = []
        with open('/mnt/gouqi/deep-speed-chat/datasets/zh_honesty.json',encoding='utf-8') as fp:
            lines = fp.readlines()
            for i in range(50):
                item = json.loads(lines[i])
                honesty_prompt.append(item['instruction'])
        with open('/mnt/gouqi/deep-speed-chat/datasets/zh_harmless.json',encoding='utf-8') as fp:
            lines = fp.readlines()
            for i in range(len(lines)):
                item = json.loads(lines[i])
                harmlessness_prompt.append(item['user_prompt'])

        prompts = honesty_prompt + harmlessness_prompt
        prompts = [sft_format.format(x) for x in prompts]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    results = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts)
    with open(os.path.join('/mnt/gouqi/deep-speed-chat/training/step1_supervised_finetuning/outputs','test_honesty_harmlessness_zh.json'),'w') as fp:
        json.dump(results,fp,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
