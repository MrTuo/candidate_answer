import json
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
# some
embedding_size = 100
dir_train = 'D:/Github/candidate_answer/data/json_train_expt_stop'
dir_test = 'D:/Github/candidate_answer/data/json_test_expt_stop'
dir_embedding = 'D:/nlp_data/sogou_100_nobinary'
max_word = 20 # question 和 answer中包含的最大词数

# 加载词向量
embedding = {}
# f = open(dir_embedding,"r",encoding='utf-8')
# line = f.readline()
# line_num = 0
# print("loading enmbedding...")
# while line:
#     try:
#         content = line.strip(' \n').split(' ')
#         assert len(content) == embedding_size + 1
#         embedding[content[0]] = np.array([float(i) for i in content[1:]])
#         line = f.readline()
#         line_num+=1
#         print(line_num)
#     except:
#         print(content)
#         break
# print("finish loading")

# 创建一个CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(1, 3, 5)
        self.conv3 = nn.Conv1d(1, 3, 5)

        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.fc11 = nn.Linear(embedding_size, 240)
        self.fc12 = nn.Linear(embedding_size, 240)
        self.fc13 = nn.Linear(embedding_size, 240)

        self.fc21 = nn.Linear(240, 80)
        self.fc22 = nn.Linear(240, 80)
        self.fc23 = nn.Linear(240, 80)


    def forward(self, x1, x2, x3):
        # x1/x2/x3 分别表示错误答案，问题，正确答案
        x1 = self.pool1(F.tanh(self.conv1(x1)))
        x2 = self.pool2(F.tanh(self.conv2(x2)))
        x3 = self.pool3(F.tanh(self.conv3(x3)))

        x1 = self.F.tanh(x1)
        x2 = self.F.tanh(x2)
        x3 = self.F.tanh(x3)

        neg_cosine = F.cosine_similarity(x1,x2)
        pos_cosine = F.cosine_similarity(x2,x3)

        return F.hinge_embedding_loss([neg_cosine,pos_cosine],size_average=False)
net = Net()

# 开始训练
def get_sentence_embedding(s):
    arr = []
    for word in s:
        if word in embedding:
            arr.append(embedding[word])
        else:
            arr.append([random.uniform(-1,1) for i in range(embedding_size)])
    return arr

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
f = open(dir_train,'r',encoding='utf-8')
data = json.loads(f.read())
count_step = 0
for epoch in range(2):  # loop over the dataset multiple times
    f = open(dir_train)
    running_loss = 0.0

    for id in data:
        # get the inputs
        x2 = get_sentence_embedding(data[id]['question'])
        x3 = get_sentence_embedding(data[id]['right_answer'][0])
        for wrong_answer in data[id]['wrong_answer']:
            x2 = get_sentence_embedding(wrong_answer)
            # wrap them in Variable
            x1, x2, x3 = Variable(x1), Variable(x2), Variable(x3)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = net(x1, x2, x3)
            loss.backward()
            optimizer.step()

            # print statistics
            count_step += 1
            running_loss += loss.data[0]
            if count_step % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count_step + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')
#
# #
# dataiter = iter([])
# images, labels = dataiter.next()