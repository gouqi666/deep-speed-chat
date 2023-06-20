import json
import pandas as pd
# pdObj = pd.read_json('/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/eval_rlhf.json')
# csvData = pdObj.to_csv('/home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor/eval_rlhf.csv',index=False)


'''
idx = [[],[5,2],[7,4],[],[],[1,0],[1,4],[5,0],[9,4],[],[6,7],[],[6,0],[5,3],[5,2],[],[5,2],[2,4],[],[],[6,1],[],[6,3],\
       [5,6],[],[1,2],[],[],[5,4],[],[9,3],[8,0],[],[],[5,9],[6,0],[6,9],[],[5,2],[8,0],[5,3],[9,8],[3,0],[3,1],[6,3],[7,0],[],[5,4],[7,2],[4,0]]



assert  len(idx) == 50
path = r'D:\deeplang\data\belle_response_gq.jsonl'
data = []
with open(path,'r',encoding='utf-8') as fp:
       for line in fp.readlines():
              item = json.loads(line)
              data.append(item)
result = []
for index,item in zip(idx,data):
       if index:
              new = {}
              new['prompt'] = item['question']
              good,bad = index[0],index[1]
              new['chosen'] = item['response_v3.3'][good] if good < 5 else item['response_v3.4'][good-5]
              new['rejected'] = item['response_v3.3'][bad] if bad < 5 else item['response_v3.4'][bad-5]
              result.append(new)
with open(r'D:\deeplang\data\results.jsonl','w',encoding='utf-8') as fp:
       for item in result:
              json.dump(item,fp,ensure_ascii=False)
              fp.write('\n')
'''
'''
idx = [[], [], [], [], [], [1,2], [0,3], [], [], [],
        [0,1], [], [0,3], [], [4,2], [], [0,3], [], [], [],
        [], [], [], [], [1,2],[],[],[],[4,0],[],[1,3],[3,0],[],[4,0],[1,0],[],[3,4],[],[],[],[1,0],[],[0,2],[1,2],[],[1,3],[],[3,4],[],[]] # 26-50
idx2 = [[],[4,3],[],[],[0,4],[],[0,4],[],[],[],[],[0,3],[1,3],[],[],[],[0,4],[],[0,4],[],[],[],[],[1,0],[3,0],[0,2],[],\
        [],[3,1],[3,4],[0,4], [3,2], [], [3,4], [1,4], [], [0,4], [4,0], [], [] ,[],[]] # 51 -79
assert  len(idx) == 50
assert len(idx2) == 42
path = r'D:\deeplang\deep-speed-chat\output\test_response_33.jsonl'
data = []
with open(path,'r',encoding='utf-8') as fp:
       for line in fp.readlines():
              item = json.loads(line)
              data.append(item)
idx_f = idx + idx2
result = []
for index,item in zip(idx_f,data):
       if index:
              new = {}
              new['prompt'] = item['question']
              good,bad = index[0],index[1]
              new['chosen'] = item['response'][good]
              new['rejected'] = item['response'][bad]
              result.append(new)
with open(r'D:\deeplang\deep-speed-chat\output\results.jsonl','w',encoding='utf-8') as fp:
       for item in result:
              json.dump(item,fp,ensure_ascii=False)
              fp.write('\n')
'''

# path = r'D:\deeplang\data\moss-003-sft-no-tools.jsonl'
# data = []
# with open(path,'r',encoding='utf-8') as fp:
#        for line in fp.readlines():
#               item = json.loads(line)
#               if item['category'] == 'Harmless':
#                      data.append(item)

# from langdetect import detect
# data = []
# with open(r'D:\deeplang\data\moss-harmless.jsonl','r',encoding='utf-8') as fp:
#     for line in fp.readlines():
#         item = json.loads(line)
#         del item['meta_instruction']
#         human_turn1 = item['chat']['turn_1']['Human']
#         try:
#             if human_turn1!= '' and detect(human_turn1[11:-6]) == 'zh-cn':
#                 data.append(item)
#         except Exception as e:
#             print('error')
data = []
with open(r'D:\deeplang\data\moss-harmless-zh.jsonl', 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        item = json.loads(line)
        new = {}
        new['prompt'] = item['chat']['turn_1']['Human'][11:-6]
        new['chosen'] = ''
        new['rejected'] = ''
        data.append(new)
with open(r'D:\deeplang\data\moss-harmless-zh-singleturn.jsonl', 'w', encoding='utf-8') as fp:
    for item in data:
        json.dump(item, fp, ensure_ascii=False)
        fp.write('\n')