#超市购物
# n,m = 3,3
# goods = 'abc'
# customer = 'abcdef'
# already = set()
# customer_set = []
# ans = 0
# for x in customer:
#     if x not in already:
#         already.add(x)
#         if x in goods:
#             ans += 1
#
# print(ans)
# tf_idf
# from collections import defaultdict
# import math
# T = int(input())
# for _ in range(T):
#     d,w,t = input().split()
#     d = int(d)
#     w = int(w)
#     t = float(t)
#     passage = []
#     total_words = defaultdict(int)
#     num_words = 0
#     for i in range(d):
#         cur = input().split()
#         num_words += len(cur)
#         for x in cur:
#             total_words[x] += 1
#         passage.append(cur)
#     for k,v in total_words.items():
#         tf_i = v / num_words
#         fre = 0
#         for p in passage:
#             if k in p:
#                 fre += 1
#         idf_i = math.log(d / (fre+1))
#         tf_idf = tf_i * idf_i
#         if tf_idf > t:
#             print(1)
#             break
#     else:
#         print(0)


# 桥
# T = int(input())
# for _ in range(T):
#     n,a,b = map(int,input().split())
#     diff = abs(a-b)
#     if diff - 1 > n - 2:
#         print(-1)
#     else:
#         print((n - 2 - diff + 1) // 2 + max(a,b))


# 共生团，
from collections import defaultdict
n,k = map(int,input().split())
origin_arr = []
def verify(s1,s2,k):
    l1,l2 = len(s1), len(s2)
    dp = [[0] * (l2+1) for _ in range(l1+1)]
    ans = 0
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            ans = max(ans,dp[i][j])
    if ans >= k:
        return True
    else:
        return False
str2id = {}

for _ in range(n):
    cur = input()
    if cur not in str2id:
        str2id[cur] = [len(str2id),1]
    else:
        str2id[cur][1] += 1
    origin_arr.append(cur)
pre = []
arr = list(set(origin_arr))
num = len(arr)
father = [i for i in range(num)]

def find_father(x):
    if father[x] == x:
        return x
    else:
        father[x] = find_father(father[x])
        return father[x]
def merge(x,y):
    father_x = find_father(x)
    father_y = find_father(y)
    if father_x != father_y:
        father[father_x] = father_y
for i in range(num):
    for j in range(i):
        if verify(arr[i],arr[j],k):
            merge(i,j)
pre = defaultdict(list)
ans = 0
for i in range(num):
    father_i = find_father(i)
    if father_i == i:
        ans += 1
    pre[father_i].append(arr[i])

order = list(map(int,input().split()))
print(ans)
for x in order:
    for k,v in pre.items():
        if origin_arr[x-1] in v[:]:
            str2id[origin_arr[x - 1]][1] -= 1
            if str2id[origin_arr[x - 1]][1] == 0:
                v.remove(origin_arr[x-1])
            if not v:
                ans -= 1
            break
    print(ans)