
import json
import torch
import os

from transformers import AutoTokenizer,LlamaTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler,SequentialSampler
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.data.data_utils import DataCollatorReward, PromptDataset
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
def load_stuff(model_name_or_path, num_padding_at_beginning=1):

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
    #                                           fast_tokenizer=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='right', truncation_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer,ds_config=None,
                                num_padding_at_beginning=num_padding_at_beginning, rlhf_training=True)
    return model, tokenizer

def load_ziya_reward_model(model_name_or_path="IDEA-CCNL/Ziya-LLaMA-7B-Reward"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
    model = create_critic_model(model_name_or_path,tokenizer,None,0,True,True)
    return model, tokenizer

from tqdm import tqdm
if __name__ == "__main__":
    os.environ['TRAIN_LLAMA'] = '1'
    model_path = '/mnt/data01/shenyan/ckpt/llama_hf/Ziya-LLaMA-7B-Reward'
    rm_model, tokenizer = load_ziya_reward_model(model_path)
    device = torch.device("cuda:0")
    # rm_model.to(device)
    print(device)
    rm_model.eval()

    data_path = '/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b-reward-ziya-v4/actor'
    before_ppo_path = os.path.join(data_path,'before_ppo.json')
    after_ppo_path = os.path.join(data_path,'after_ppo_epoch_0_835.json')
    with open(before_ppo_path,'r',encoding='utf-8') as f:
        before_ppo_data = json.load(f)
    with open(after_ppo_path, 'r', encoding='utf-8') as f:
        after_ppo_data = json.load(f)
    dct = {}
    chosen_dataset = []
    reject_dataset = []
    prompt_length  = []
    max_seq_len = 512
    data = []
    for before,after in zip(before_ppo_data,after_ppo_data):
        assert before['prompt'] == after['prompt']
        chosen_sentence = before['prompt'] + before['response']
        reject_sentence = before['prompt'] + after['response']
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
        item = {}
        item['prompt'] =  before['prompt']
        item['before_ppo'] = before['response']
        item['after_ppo'] = after['response']
        data.append(item)
        chosen_dataset.append(chosen_token)
        reject_dataset.append(reject_token)
        prompt_length.append(0) # not used
    eval_dataset = PromptDataset([], chosen_dataset, reject_dataset,prompt_length,
                         tokenizer.pad_token_id, 2)
    prompt_eval_sampler = SequentialSampler(eval_dataset)
    data_collator = DataCollatorReward()
    prompt_eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=1)


    before_ppo_scores = []
    after_ppo_scores = []
    for batch in tqdm(prompt_eval_dataloader):
        # batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            reward = rm_model.forward_value(**batch)['chosen_end_scores']
        reward = reward.tolist()
        chosen = [ x for i,x in enumerate(reward) if i % 2 == 0]
        rejected = [ x for i,x in enumerate(reward) if i % 2 != 0]

        before_ppo_scores.extend(chosen)
        after_ppo_scores.extend(rejected)
    print('Total Length:',len(before_ppo_scores))
    print('before_ppo_score:',sum(before_ppo_scores) / len(before_ppo_scores))
    print('after_ppo_score:',sum(after_ppo_scores) / len(after_ppo_scores))
    print('after > before:',sum([after_ppo_scores[i] > before_ppo_scores[i] for i in range(len(before_ppo_scores))]))

    for before_score, after_score, item in zip(before_ppo_scores, after_ppo_scores,data):
        item['before_score'] = before_score
        item['after_score'] = after_score

    with open(os.path.join(data_path,'eval_rlhf.json'),'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
        

