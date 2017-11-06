import numpy as np
import json as js

embedding_size = 100
dir_embedding = 'D:/nlp_data/sogou_100_nobinary'

# 加载词向量
embedding = {}
f = open(dir_embedding,"r",encoding='utf-8')
line = f.readline()
line_num = 0
print("loading enmbedding...")
while line:
    try:
        content = line.strip(' \n').split(' ')
        assert len(content) == embedding_size + 1
        embedding[content[0]] = np.array([float(i) for i in content[1:]])
        line = f.readline()
        line_num+=1
#         print(line_num)
    except:
        print(content)
        break
print("finish loading")
w = open('json_embedding','w',encoding='utf-8')
w.write(js.dumps(embedding))
w.close()

# 加载
f = open("json_embedding",'r',encoding='utf-8')
data = js.loads(f.read())
print(data)
