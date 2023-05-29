
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
    from utils.ds_utils import get_eval_ds_config
    ds_eval_config = get_eval_ds_config(offload=False, stage=0)
    print(ds_eval_config)
    model = create_critic_model(model_name_or_path, tokenizer,ds_config=ds_eval_config,
                                num_padding_at_beginning=num_padding_at_beginning, rlhf_training=True)
    return model, tokenizer

from tqdm import tqdm
if __name__ == "__main__":
    os.environ['TRAIN_LLAMA'] = '1'
    data_path = '/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/rl_hf_pair_result_beam3.json'
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    model_path = '/home/gq/deeplang/deep-speed-chat/training/step2_reward_model_finetuning/output/llama-7b-v2'
    rm_model, tokenizer = load_stuff(model_path,1)
    device = torch.device("cuda:0")
    rm_model.to(device)
    print(device)
    rm_model.eval()

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

    with open(os.path.join('/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/','eval_rlhf.json'),'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
        

