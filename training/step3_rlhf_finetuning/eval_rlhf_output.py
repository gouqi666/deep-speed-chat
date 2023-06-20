
import json
import torch
import os

from transformers import AutoTokenizer,LlamaTokenizer
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.model.model_utils import create_critic_model
from utils.utils import to_device
def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token=None):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response

    prompt_token = tokenizer(prompt,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    prompt_ids = prompt_token["input_ids"][0].tolist()[1:]
    try:
        p_length = prompt_ids.index(tokenizer.pad_token_id) + 1
    except Exception as e:
        print('179:!!!!!!!!!', prompt_token)
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
    batch['prompt_length'] = torch.tensor([p_length] * 2)
    return batch
def load_stuff(model_name_or_path, num_padding_at_beginning):

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
    #                                           fast_tokenizer=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='right', truncation_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer,ds_config=None,
                                num_padding_at_beginning=num_padding_at_beginning, rlhf_training=True)
    return model, tokenizer

from tqdm import tqdm
if __name__ == "__main__":
    os.environ['TRAIN_LLAMA'] = '1'
    data_path = '/mnt/gouqi/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor'
    before_ppo_path = os.path.join(data_path,'before_ppo.json')
    after_ppo_path = os.path.join(data_path,'after_ppo_epoch_0.json')
    with open(before_ppo_path,'r',encoding='utf-8') as f:
        before_ppo_data = json.load(f)
    with open(after_ppo_path, 'r', encoding='utf-8') as f:
        after_ppo_data = json.load(f)
    dct = {}
    sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    with open('/mnt/gouqi/deep-speed-chat/datasets/synthetic-instruct-gptj-pairwise/test.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = json.loads(line)
            dct[sft_format.format(item['prompt'])] = [item['chosen'], item['rejected']]
    data = []
    for before,after in zip(before_ppo_data,after_ppo_data):
        for i in range(len(before['prompt'])):
            item = {}
            assert before['prompt'][i] == after['prompt'][i]
            item['prompt'] = before['prompt'][i]
            item['before_ppo'] = before['ans'][i]
            item['after_ppo'] = after['ans'][i]
            try:
                item['chosen'] = dct[item['prompt']][0]
                item['rejected'] = dct[item['prompt']][1]
            except Exception as e:
                print(item['prompt'])
                print('not found chosen')
                exit()
            data.append(item)
    model_path = '/mnt/gouqi/deep-speed-chat/training/step2_reward_model_finetuning/output/llama-7b'
    rm_model, tokenizer = load_stuff(model_path,1)
    device = torch.device("cuda:0")
    rm_model.to(device)
    print(device)
    rm_model.eval()
    chosen_scores = []
    rejected_scores = []
    before_ppo_scores = []
    after_ppo_scores = []
    for sent in tqdm(data):
        prompt = sent['prompt']
        good_ans = sent['after_ppo']
        bad_ans = sent['before_ppo']
        chosen_ans = sent['chosen']
        rejected_ans = sent['rejected']

        batch = prepare_datapair(prompt,
                                    good_ans,
                                    bad_ans,
                                    tokenizer,
                                    max_seq_len=512,
                                    end_of_conversation_token="<|endoftext|>")

        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        sent['after_score'] = outputs["chosen_mean_scores"].item()
        sent['before_score'] = outputs["rejected_mean_scores"].item()

        # test chosen and rejected
        batch = prepare_datapair(prompt,
                                 chosen_ans,
                                 rejected_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")

        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        sent['chosen_score'] = outputs["chosen_mean_scores"].item()
        sent['rejected_score'] = outputs["rejected_mean_scores"].item()
        chosen_scores.append(sent['chosen_score'])
        rejected_scores.append(sent['rejected_score'])
        before_ppo_scores.append(sent['before_score'])
        after_ppo_scores.append(sent['after_score'])
    print('Total Length:',len(chosen_scores))
    print('chosen_score:',sum(chosen_scores) / len(chosen_scores))
    print('rejected_score:',sum(rejected_scores) / len(rejected_scores))
    print('before_ppo_score:',sum(before_ppo_scores) / len(before_ppo_scores))
    print('after_ppo_score:',sum(after_ppo_scores) / len(after_ppo_scores))
    print('after > before:',sum([after_ppo_scores[i] > before_ppo_scores[i] for i in range(len(chosen_scores))]))
    print('chosen > rejected:',sum([chosen_scores[i] > rejected_scores[i] for i in range(len(chosen_scores))]))
    with open(os.path.join('/mnt/gouqi/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor','eval_rlhf.json'),'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
        

