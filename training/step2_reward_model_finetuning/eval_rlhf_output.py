


import json
import torch
import os

from transformers import AutoTokenizer
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
def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer
from tqdm import tqdm
if __name__ == "__main__":
    data_path = '/nfs2/wzt/deep-speed-chat/training/step3_rlhf_finetuning/output/actor/test_rlhf_output.json'
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    model_path = '/nfs2/wzt/deep-speed-chat/training/step2_reward_model_finetuning/output'
    rm_model, tokenizer = load_stuff(model_path,1)
    device = torch.device("cuda:0")
    rm_model.to(device)
    rm_model.eval()

    for sent in tqdm(data):
        prompt = sft_format.format(sent['prompt'])
        good_ans = sent['after_rlhf']
        bad_ans = sent['before_rlhf']
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
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())
        sent['after_score'] = outputs["chosen_mean_scores"].item()
        sent['before_score'] = outputs["rejected_mean_scores"].item()
    with open('eval_rlhf.json','w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
        

