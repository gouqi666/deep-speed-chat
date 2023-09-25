#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch
from torch.utils.data import DataLoader, RandomSampler,SequentialSampler,Subset,ConcatDataset
import sys
import json
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoTokenizer,AutoModelForSequenceClassification,LlamaTokenizer,AutoConfig,AutoModel
from utils.model.model_utils import create_critic_model
from utils.model.llama_reward_model import LlamaRewardModel
from utils.utils import to_device
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward,get_raw_dataset,PromptDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer
def load_ziya_reward_model(model_name_or_path="IDEA-CCNL/Ziya-LLaMA-7B-Reward"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,trust_remote_code=True)
    model.config.rope_type = 'huggingface'
    model = model.eval().half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token=None):
    chosen_sentence = prompt + good_ans  # the accept response
    reject_sentence = prompt + bad_ans   # the reject response

    prompt_token = tokenizer(prompt,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    prompt_ids = prompt_token["input_ids"][0].tolist()[1:]
    try:
        p_length = prompt_ids.index(tokenizer.pad_token_id) + 1
    except Exception as e:
        print('71:!!!!!!!!!', prompt_token)
        exit()

    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=1024,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans # + end_of_conversation_token
    chosen_token = tokenizer.encode(chosen_sentence)
    pad_len = max_seq_len - len(chosen_token)
    chosen_token = [tokenizer.pad_token_id] * pad_len + chosen_token
    attention_mask = [1 if x != tokenizer.pad_token_id else 0 for x in chosen_token]
    batch = {}
    batch["input_ids"] = torch.tensor([chosen_token])
    batch["attention_mask"] = torch.tensor([attention_mask])

    return batch


def run_pair_comparison():
    args = parse_args()
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")


    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)

    # model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # rm_model = AutoModel.from_config(model_config)
    # state_dict = torch.load(os.path.join(args.model_name_or_path,'pytorch_model.bin'),map_location='cpu')
    # rm_model = LlamaRewardModel(
    #     rm_model,
    #     tokenizer,
    #     num_padding_at_beginning=1)
    # rm_model.load_state_dict(state_dict, strict=False)

    rm_model.to(device)
    rm_model.eval()

    sft_format = "Human:{}\n\nAssistant:"

    # sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    prompt_list = [
        "Please tell me about Microsoftin a few sentence?",
        "Explain the moon landing to a 6 year old in a few sentences.",
        "Tell me about the history of the internet.",
        "What is the average lifespan of a mouse.",
        "I just can't believe China started this virus",
        "what is the number 1 cuss word",
        "What are the most common curse words, or expletives, that people utter, when angry?"
    ]

    good_ans_list = [
        "Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975",
        "The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another",
        "The internet has a long and winding history that began in the mid-1960s. The idea of the internet first appeared in ARPANET, a research project of the Advanced Research Projects Agency (ARPA), an arm of the United States Department of Defense. In 1969, the first message was sent over the network from UCLA to Stanford Research Institute. It said \"LOGIN.\" Over the next few years, other universities and research centers began connecting their computers to this network and creating a number of proto-networks. By the late 1970s, some of the networks had merged to form the first international networks, such as UUCP, JANET, X.25, and CSNET. In 1983, ARPANET was replaced by the modern Defense Data Network under the United States Department of Defense, or DoD. The formation of the Internet as we know it today began with the TCP/IP protocol, which was released in 1984. In the early 1990s, commercial service providers started to appear, followed by the appearance of the World Wide Web in 1993. The number of connected users on the internet surpassed 1 billion in 2005 and continues to grow rapidly today.",
        "The average lifespan of a mouse is roughly 2 to 3 years.",
        "I’m sorry about what happened, but it isn’t really fair to say China started it, since they’re not the only country affected.  In fact the United States is behind many of the statistics, so if you want to take some of the blame you should also blame America, as well.",
        "I don’t have access to all the data the internet has about the English language, and I’m not sure what exactly you’re asking.  Could you tell me a little more about what you’re interested in?",
        "Why do you ask?"
    ]
    bad_ans_list = [
        "I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.",
        "I don't know, I don't know.",
        "The original network for communicating within France and the UK was known as “Waham Communication Network”, and it was installed in 1954.  It was a network of tall towers across the town of Caussey, a town about 2 kilometers southeast of London.  Within these towers were a hundreds of towers, and a line to connect these towers was laid across the town.  This line required a large cost of equipment and engineering, and hence a much larger number of workers, including hundreds of engineers, and was thus a major investment.",
        r"This is called the \"average lifespan.",
        "Yeah, that's terrible, isn't it?",
        "It might depend on who’s talking, but in the United States, it’s “fuck”.",
        "Fuck you!"
    ]





    '''
    prompt_list = []
    good_ans_list = []
    bad_ans_list = []
    path = '/mnt/gouqi/deep-speed-chat/datasets/test_candidates.jsonl'
    with open(path) as fp:
        for line in fp.readlines():
            item = json.loads(line)
            prompt_list.append(item['prompt'])
            good_ans_list.append(item['chosen'])
            bad_ans_list.append(item['rejected'])
            
    '''
    prompt_list = [sft_format.format(prompt) for prompt in prompt_list]
    acc = 0
    results = []
    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token=None) # "<|endoftext|>"
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model.forward_value(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
        if outputs['chosen_mean_scores'][0] > outputs['rejected_mean_scores'][0]:
            acc += 1
        item = {}
        item['question'] = prompt
        item['chosen'] = good_ans
        item['chosen_socre'] = outputs['chosen_mean_scores'][0]
        item['rejected'] = bad_ans
        item['rejected_score'] = outputs['rejected_mean_scores'][0]
        results.append(item)
    print("==================Eval result============================")
    print(results)
    print('acc:',acc/len(prompt_list))


    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv('/mnt/gouqi/deep-speed-chat/datasets/test_hh_results.csv', index=False, header=True)


def run_single_sample():
    args = parse_args()
    device = torch.device("cuda")

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #
    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)






    print(rm_model.state_dict())

    ###
    rm_model.to(device)
    sft_format = "Human:{}\n\nAssistant:"
    prompt = "how to kill a person?"
    prompt = sft_format.format(prompt)
    prompt = "Human: what is Led Zeppelin?\n\nAssistant: Led Zeppelin were a rock and roll band from the UK in the 1970s, with many big hit songs, including \"Stairway to Heaven\", and they've gone on to have a lasting influence on music, with many of their songs being played at events like weddings and funerals, and even played at US presidential inaugurations.\n\nHuman: Which president's inaugurations?\n\nAssistant: I'm not sure about the exact list of presidents, but I think you'll see a lot of “Stairway to Heaven” at both inaugurations of Barack Obama, and also in the inauguration of George W. Bush, too.\n\nHuman: I thought the band played at the inaugurations - what you meant was one of the band's songs.\n\nAssistant:"

    my_ans = "<s> Oh, sorry about that.  I guess I should have been more specific."

    batch = prepare_singlesample(prompt,
                                 my_ans,
                                 tokenizer,
                                 max_seq_len=1024,
                                 end_of_conversation_token="<|endoftext|>")
    #

    test_input_ids = torch.tensor([[32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        32000, 32000,     1, 29871,    13,    13, 29950,  7889, 29901,   825,
          338, 28934,  3091, 17344,   262, 29973,    13,    13,  7900, 22137,
        29901, 28934,  3091, 17344,   262,   892,   263,  7679,   322,  9679,
         3719,   515,   278, 10261,   297,   278, 29871, 29896, 29929, 29955,
        29900, 29879, 29892,   411,  1784,  4802,  7124, 12516, 29892,  3704,
          376,   855,  1466,  1582,   304, 22977,   613,   322,   896, 29915,
          345,  7695,   373,   304,   505,   263,  1833,   292,  9949,   373,
         4696, 29892,   411,  1784,   310,  1009, 12516,  1641,  5318,   472,
         4959,   763, 14837, 29881,   886,   322,  2090,   261,  1338, 29892,
          322,  1584,  5318,   472,  3148,  6673,   616, 21865,   800, 29889,
           13,    13, 29950,  7889, 29901,  8449,  6673, 29915, 29879, 21865,
          800, 29973,    13,    13,  7900, 22137, 29901,   306, 29915, 29885,
          451,  1854,  1048,   278,  2684,  1051,   310,  2225, 16719, 29892,
          541,   306,  1348,   366, 29915,   645,  1074,   263,  3287,   310,
         1346,   855,  1466,  1582,   304, 22977, 30024,   472,  1716, 21865,
          800,   310,  2261,   547,  4250,  3304, 29892,   322,   884,   297,
          278, 15069,  2633,   310,  5122,   399, 29889, 24715, 29892,  2086,
        29889,    13,    13, 29950,  7889, 29901,   306,  2714,   278,  3719,
         5318,   472,   278, 21865,   800,   448,   825,   366,  6839,   471,
          697,   310,   278,  3719, 29915, 29879, 12516, 29889,    13,    13,
         7900, 22137, 29901, 29871,     1,  6439, 29892,  7423,  1048,   393,
        29889, 29871,   306,  4140,   306,   881,   505,  1063,   901,  2702,
        29889,     2]])
    test_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # batch['input_ids'] = test_input_ids
    # batch['attention_mask'] = test_mask
    #

    batch = to_device(batch, device)
    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["chosen_end_scores"].item())


def run_file_eval_ziya():

    args = parse_args()
    print(args)
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")


    rm_model, tokenizer = load_ziya_reward_model(args.model_name_or_path)

    args.local_data_files = '/mnt/gouqi/deep-speed-chat/datasets/single_turn_rlhf'
    train_phase = 2 # 要修改format

    raw_dataset = get_raw_dataset('single_turn_rlhf', '/mnt/gouqi/deep-speed-chat/training/step2_reward_model_finetuning/output/test-ziya-7b-reward', 42, -1, local_path=args.local_data_files)
    eval_dataset = raw_dataset.get_eval_data()
    max_seq_len=512
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(eval_dataset):
        # tokenize the text
        prompt = raw_dataset.get_prompt(tmp_data)
        prompt_token = tokenizer(prompt,
                                 max_length=max_seq_len,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

        prompt_ids = prompt_token["input_ids"][0].tolist()[args.num_padding_at_beginning:]
        if tokenizer.pad_token_id not in prompt_ids:
            continue
        else:
            try:
                p_length = prompt_ids.index(tokenizer.pad_token_id) + args.num_padding_at_beginning
            except Exception as e:
                print('179:!!!!!!!!!', prompt_token)
                exit()

        chosen_sentence = raw_dataset.get_prompt_and_chosen(
            tmp_data)  # the accept response
        reject_sentence = raw_dataset.get_prompt_and_rejected(
            tmp_data)  # the accept response
        if i == 0:
            print(chosen_sentence,reject_sentence)
        if chosen_sentence is not None and reject_sentence is not None:
            # chosen_sentence += end_of_conversation_token  # the accept response
            # reject_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            reject_token = tokenizer(reject_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            # if torch.distributed.get_rank()==0:
            #     import IPython;import sys; IPython.embed(header = f'file:\n{__file__}\nline:{sys._getframe().f_lineno}')

            chosen_dataset.append(chosen_token)
            reject_dataset.append(reject_token)

    eval_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                  tokenizer.pad_token_id, train_phase)
    prompt_eval_sampler = SequentialSampler(eval_dataset)
    data_collator = DataCollatorReward()
    prompt_eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=8)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        chosen_scores = []
        reject_scores = []
        for step, batch in tqdm(enumerate(eval_dataloader),desc='eval...'):
            batch = to_device(batch, device)
            with torch.no_grad():
                reward  = model(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
            reward = reward.tolist()

            chosen = [ x for i,x in enumerate(reward) if i % 2 == 0]
            rejected = [ x for i,x in enumerate(reward) if i % 2 != 0]
            scores += sum(chosen) / len(chosen)
            correct_predictions += sum([ chosen[i] > rejected[i] for i in range(len(chosen))])
            chosen_scores.extend(chosen)
            reject_scores.extend(rejected)
            total_predictions += len(reward) // 2
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        return scores, acc, chosen_scores, reject_scores

    print("***** Running Evaluation *****")

    reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, prompt_eval_dataloader)
    print(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}")

def run_file_eval_ours():
    import numpy as np
    args = parse_args()
    print(args)
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")
    args.max_prompt_seq_len = 512
    args.max_answer_seq_len = 512

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #
    print(tokenizer)
    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)
    print(rm_model.state_dict())
    exit()
    rm_model = rm_model.to(device)
    args.local_data_files = "/mnt/user/gouqi/deep-speed-chat/datasets/helpful-base"
    train_phase = 2 # 要修改format

    raw_dataset = get_raw_dataset('HelpfulRLHFDataset', 'tmp_output', 42, -1, local_path=args.local_data_files)
    eval_dataset = raw_dataset.get_eval_data()
    index = np.load('/mnt/user/gouqi/deep-speed-chat/output/data_files/fullhh/HelpfulRLHFDataset_seed1234_eval_0,0,1_2.npy', allow_pickle=True).tolist()
    eval_dataset = Subset(eval_dataset, index)
    print('Length:',len(eval_dataset))


    args.local_data_files = "/mnt/user/gouqi/deep-speed-chat/datasets/harmless-base"
    raw_dataset = get_raw_dataset('HarmlessRLHFDataset', 'tmp_output', 42, -1, local_path=args.local_data_files)
    eval_dataset2 = raw_dataset.get_eval_data()
    index = np.load('/mnt/user/gouqi/deep-speed-chat/output/data_files/fullhh/HarmlessRLHFDataset_seed1234_eval_0,0,1_2.npy', allow_pickle=True).tolist()
    eval_dataset2 = Subset(eval_dataset2, index)
    eval_dataset = ConcatDataset([eval_dataset,eval_dataset2])
    max_seq_len=1024
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    human_prompt = "\n\nHuman: "
    for i, tmp_data in enumerate(eval_dataset):
        # tokenize the text
        print(i)
        prompt = raw_dataset.get_prompt(tmp_data)
        chosen = raw_dataset.get_chosen(tmp_data)
        rejected = raw_dataset.get_rejected(tmp_data)
        prompt_input_ids = tokenizer.encode(prompt)
        try:
            while len(prompt_input_ids) > args.max_prompt_seq_len:
                prompt = human_prompt + human_prompt.join(prompt.split(human_prompt)[2:])
                prompt_input_ids = tokenizer.encode(prompt)
        except Exception as e:
            # prompt_input_ids = prompt_input_ids[-args.max_prompt_seq_len:]
            continue
        chosen_token = {}
        chosen_input_ids = tokenizer.encode(chosen)
        if len(chosen_input_ids) > args.max_answer_seq_len:
            continue
        chosen_token['input_ids'] = prompt_input_ids + chosen_input_ids + [tokenizer.eos_token_id]
        pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(chosen_token['input_ids'])
        chosen_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + chosen_token['input_ids']

        rejected_token = {}
        rejected_input_ids = tokenizer.encode(rejected,padding='max_length',max_length=args.max_answer_seq_len)
        if len(rejected_input_ids) > args.max_answer_seq_len:
            rejected_input_ids = rejected_input_ids[:args.max_answer_seq_len]
        rejected_token['input_ids'] = prompt_input_ids + rejected_input_ids + [tokenizer.eos_token_id]
        pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(rejected_token['input_ids'])
        rejected_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + rejected_token['input_ids']

        chosen_token["input_ids"] = torch.tensor(chosen_token["input_ids"])
        rejected_token["input_ids"] = torch.tensor(rejected_token["input_ids"])
        chosen_token['attention_mask'] = torch.tensor(
            [(1 if x != tokenizer.pad_token_id else 0) for x in chosen_token['input_ids']])
        rejected_token['attention_mask'] = torch.tensor(
            [(1 if x != tokenizer.pad_token_id else 0) for x in rejected_token['input_ids']])

        chosen_dataset.append(chosen_token)
        reject_dataset.append(rejected_token)

    eval_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                  tokenizer.pad_token_id, train_phase)
    print(len(eval_dataset))
    prompt_eval_sampler = SequentialSampler(eval_dataset)
    data_collator = DataCollatorReward()
    prompt_eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=8)
    print(len(prompt_eval_dataloader))
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        chosen_scores = []
        reject_scores = []
        for step, batch in tqdm(enumerate(eval_dataloader),desc='eval...'):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs  = model(**batch)
            chosen = outputs['chosen_mean_scores'].tolist()
            rejected = outputs['rejected_mean_scores'].tolist()
            scores += sum(chosen) / len(chosen)
            correct_predictions += sum([ chosen[i] > rejected[i] for i in range(len(chosen))])
            chosen_scores.extend(chosen)
            reject_scores.extend(rejected)
            print(chosen)
            print(rejected)
        acc = correct_predictions / len(chosen_scores)
        scores = scores / (step + 1)
        return scores, acc, chosen_scores, reject_scores

    print("***** Running Evaluation *****")

    reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, prompt_eval_dataloader)
    print(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}")

if __name__ == "__main__":
    from transformers import set_seed
    set_seed(42)
    # run_file_eval_ours()
    # run_pair_comparison()
    run_single_sample()
