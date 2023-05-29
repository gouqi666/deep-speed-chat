import json
import pandas as pd
pdObj = pd.read_json('/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/eval_rlhf.json')
csvData = pdObj.to_csv('/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/eval_rlhf.csv',index=False)
