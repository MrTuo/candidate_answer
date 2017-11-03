import numpy as np
import random

def cosine_sim(a, b, embedding_size):
    # 计算余弦相似度
    t1 = sum([a[i] * b[i] for i in range(embedding_size)])
    t2 = sum([a[i] * a[i] for i in range(embedding_size)]) * sum([b[i] * b[i] for i in range(embedding_size)])
    return t1 / t2


def get_sim(s1, s2, embedding_size):
    '''
    得到句子s1、s2之间的相似度
    '''
    sim_mat = [[0 for j in range(len(s2))] for i in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            sim_mat[i][j] = cosine_sim(s1[i], s2[j], embedding_size)
    sim_mat = np.array(sim_mat)
    t1 = sum([max(sim_mat[i]) for i in range(len(s1))]) / len(s1)
    t2 = sum([max(sim_mat[:, j]) for j in range(len(s2))]) / len(s2)
    return (t1 + t2) / 2

# 词向量转换
def get_embeddings(s,embedding,embedding_size):
    arr = []
    for word in s:
        if word in embedding:
            arr.append(embedding[word])
        else:
            arr.append([random.uniform(-1,1) for i in range(embedding_size)])
    return arr


# 一些常量
dir_embedding = 'D:/nlp_data/sogou_100_nobinary'
dir_train = ''
dir_test = ''
embedding_size = 100

# 加载词向量
embedding = {}
f = open(dir_embedding,"r",encoding='utf-8')
line = f.readline()
line_num = 0
while line:
    try:
        content = line.strip(' \n').split(' ')
        assert len(content) == embedding_size + 1
        embedding[content[0]] = np.array([float(i) for i in content[1:]])
        line = f.readline()
        line_num+=1
        print(line_num)
    except:
        print(content)
        break

# 加载问句和答句
dir_train = 'F:\Github\lab\candidate_answer\data\\train_expt_stop'
f = open(dir_train, 'r', encoding='utf-8')
rst = []  # 保存所有正确答案所在位置
question_num = 0
line = f.readline()
while line:
    count = 0
    arr = line.split('\t')
    new_question = arr[0].split(' ')
    score = []
    question_num += 1
    print(question_num)
    while True:
        arr = line.split('\t')
        question = arr[0].split(' ')
        if question != new_question:  # 找到一个新的问题.
            # 排序,找到正确答案所在位置
            #             print("score:",score)
            #             print("right_index:",right_idx)
            #             print("sorted s:",sorted(score))
            idx = sorted(score, reverse=True).index(score[right_idx])
            rst.append(idx)
            break
        question = get_embeddings(question, embedding, embedding_size)
        answer = arr[1].split(' ')
        if arr[2] == '1':
            right_idx = count
        answer = get_embeddings(answer, embedding, embedding_size)
        score.append(get_sim(question, answer, embedding_size))  # 计算问句和答句之间的相似度
        count += 1
        line = f.readline()

print(rst)
# assert line == ''# line应该是空的，表示已经全部读完
MRR = sum(1.0 / (i + 1) for i in rst) / len(rst)
print(MRR)

# 按照相似度重新排序

# 得到结果

