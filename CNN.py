import json
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
# 一些常量
embedding_size = 100
dir_train = 'D:/Github/candidate_answer/data/json_train_expt_stop'
dir_test = 'D:/Github/candidate_answer/data/json_test_expt_stop'
dir_embedding = 'D:/nlp_data/sogou_100_nobinary'
max_question_words = 23 # 问题最大词数，下同理
max_right_answer_words = 351
max_wrong_answer_words = 824
kernel_size = (3, embedding_size) # 卷积核的size
out_channels = 300 # 输出通道数
hidden_out = 400 # 隐藏层输出单元数

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

# 创建一个CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size, padding=(kernel_size[0]-1,0)) # 保证输出列向量在高度上与X相同
        self.conv2 = nn.Conv2d(1, out_channels, kernel_size, padding=(kernel_size[0]-1,0))
        self.conv3 = nn.Conv2d(1, out_channels, kernel_size, padding=(kernel_size[0]-1,0))

        self.pool1 = nn.MaxPool2d(1, max_wrong_answer_words) # 输出是out_channels*1维向量
        self.pool2 = nn.MaxPool2d(1, max_question_words)
        self.pool3 = nn.MaxPool2d(1, max_right_answer_words)

        self.fc1 = nn.Linear(out_channels, hidden_out)
        self.fc2 = nn.Linear(out_channels, hidden_out)
        self.fc3 = nn.Linear(out_channels, hidden_out)


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
def get_sentence_embedding(s,out_size):
    arr = []
    for word in s:
        if word in embedding:
            arr.append(embedding[word])
        else:
            arr.append([random.uniform(-1,1) for i in range(embedding_size)])
    if len(arr) < out_size: # 补零
        arr.append([[0 for i in range(embedding_size)] for j in range(out_size - len(arr))])
    return torch.from_numpy(np.array(arr))

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
f = open(dir_train,'r',encoding='utf-8')
data = json.loads(f.read())
count_step = 0
for epoch in range(2):  # loop over the dataset multiple times
    f = open(dir_train)
    running_loss = 0.0

    for id in data:
        # get the inputs
        x2 = get_sentence_embedding(data[id]['question'], max_question_words)
        x3 = get_sentence_embedding(data[id]['right_answer'][0], max_right_answer_words)
        for wrong_answer in data[id]['wrong_answer']:
            x1 = get_sentence_embedding(wrong_answer, max_wrong_answer_words)
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