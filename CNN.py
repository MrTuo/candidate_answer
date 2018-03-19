# coding:utf-8
import json
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
from io import open
# 一些常量
# On windows
# dir_train = 'D:/Github/candidate_answer/data/json_train_expt_stop'
# dir_test = 'D:/Github/candidate_answer/data/json_test_expt_stop'
# dir_embedding = 'D:/nlp_data/sogou_100_nobinary'
# On ubuntu
dir_train = 'data/json_train_expt_stop2'
dir_test = 'data/json_test_expt_stop'
dir_embedding = 'data/sogou_100_nobinary'
debug = True # depend on the information to be printed or writed in log_file

embedding_size = 100
max_question_words = 23 # 问题最大词数，下同理
max_right_answer_words = 824
max_wrong_answer_words = 824
kernel_size = (3, embedding_size) # 卷积核的size
out_channels = 300 # 输出通道数
hidden_out = 400 # 隐藏层输出单元数
batch_size = 8

log_file = open('log','w',encoding='utf-8')
def log(log_inf):
    if debug:
        print(log_inf)
    else:
        log_file.write(log_inf)
        
######################### 加载词向量
embedding = {}
f = open(dir_embedding,"r",encoding='utf-8')
line = f.readline()
line_num = 0
log("loading enmbedding...")
while line:
    try:
        content = line.strip(' \n').split(' ')
        assert len(content) == embedding_size + 1
        embedding[content[0]] = np.array([float(i) for i in content[1:]])
        line = f.readline()
        line_num+=1
#         print(line_num)
    except:
        log('loading embedding error!\n'+content)
        break
log("finish loading")
f.close()

##################### 创建一个CNN\

# caculate hinge_loss
def hinge_loss(s1,s2,t0,batch_size):
    # print(s1.size(), s2.size())
    loss = Variable(torch.Tensor(1))
    loss.data[0] = 0.0
    for i in range(batch_size):
        if (t0 - s1[i] + s2[i]).data[0] > 0:
            loss += t0 - s1[i] + s2[i]
        else:
            log("pos:%f,neg:%f" % (s1[i].data[0], s2[i].data[0]))
    return loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 300, (3,100)) # 保证输出列向量在高度上与X相同
        self.conv2 = nn.Conv2d(1, 300, (3,100))
        self.conv3 = nn.Conv2d(1, 300, (3,100))

        self.pool1 = nn.MaxPool2d(1, max_wrong_answer_words) # 输出是out_channels*1维向量
        self.pool2 = nn.MaxPool2d(1, max_question_words)
        self.pool3 = nn.MaxPool2d(1, max_right_answer_words)

        self.fc1 = nn.Linear(out_channels, hidden_out)
        self.fc2 = nn.Linear(out_channels, hidden_out)
        self.fc3 = nn.Linear(out_channels, hidden_out)


    def forward(self, x1, x2, x3, batch_size):
        # x1/x2/x3 分别表示错误答案，问题，正确答案
#         x1 = self.pool1(F.tanh(self.conv1(x1)))
#         x2 = self.pool2(F.tanh(self.conv2(x2)))
#         x3 = self.pool3(F.tanh(self.conv3(x3)))
        # print("in forward:")
        # print("X:",x1.size(),x2.size(),x3.size())
        x1 = F.tanh(self.conv1(x1))
        x2 = F.tanh(self.conv2(x2))
        x3 = F.tanh(self.conv3(x3))
        #print("conv1:",x1.size(),x2.size(),x3.size())
        
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        #print("pool:",x1.size(),x2.size(),x3.size())
        
        x1 = F.tanh(x1)
        x2 = F.tanh(x2)
        x3 = F.tanh(x3)

        neg_cosine = F.cosine_similarity(x1,x2)
        pos_cosine = F.cosine_similarity(x2,x3)
        #print(neg_cosine, pos_cosine)

        return hinge_loss(pos_cosine, neg_cosine, 2, batch_size), pos_cosine, neg_cosine
net = Net()

log("Start training")
##################### 开始训练
def get_sentence_embedding(s,out_size):
    arr = []
    for word in s:
        if word in embedding:
            arr.append(embedding[word])
        else:
            arr.append([random.uniform(-1,1) for i in range(embedding_size)])
    if len(arr) < out_size: # 补零
        append_arr = [0.0 for i in range(embedding_size)]
        for j in range(out_size - len(arr)):
            arr.append(append_arr)
    elif len(arr) > out_size:
        arr = arr[:out_size]
    return [arr]

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
f = open(dir_train,'r',encoding='utf-8')
data = json.loads(f.read()) # 8768 quesions(except 4 questions which don't have the right answer)
f.close()
count_step = 0
for epoch in range(2):  # loop over the dataset multiple times
    f = open(dir_train)
    running_loss = 0.0
    batch = [[] for i in range(3)]
    for id in data:
        # stop early
#         if count_step == 5000:
#             print("finish training")
#             break
        
        # get the inputs
        question_ebd = get_sentence_embedding(data[id]['question'], max_question_words)
        for right_answer in data[id]['right_answer']:
            right_answer_ebd = get_sentence_embedding(right_answer, max_right_answer_words)
            for wrong_answer in data[id]['wrong_answer']:
                wrong_answer_ebd = get_sentence_embedding(wrong_answer, max_wrong_answer_words)
                batch[0].append(wrong_answer_ebd)
                batch[1].append(question_ebd)
                batch[2].append(right_answer_ebd)
                if len(batch[0]) == batch_size:
                # wrap them in Variable
                # assert(batch[0])
                    x1 = Variable(torch.from_numpy(np.array(batch[0])).float())
                    x2 = Variable(torch.from_numpy(np.array(batch[1])).float())
                    x3 = Variable(torch.from_numpy(np.array(batch[2])).float())
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    loss,pos_cosine,neg_cosine = net(x1, x2, x3, batch_size)
                    loss.backward()
                    optimizer.step()

                    count_step += 1
                    running_loss += loss.data[0]
                    if count_step % 200 == 199:    # print every 2000 mini-batches
                        log('[%d, %5d] loss: %.3f' %
                              (epoch + 1, count_step + 1, running_loss / 200))
                        running_loss = 0.0
                    # clear batch
                    batch = [[] for i in range(3)]
log('Finished Training')


#################### test
log('start test...')
f = open(dir_test,'r',encoding='utf-8')
test_data = json.loads(f.read())
f.close()
MRR = 0
count_right_answer = 0
for id in test_data:
    # get the inputs
    question_ebd = get_sentence_embedding(test_data[id]['question'], max_question_words)
    
    for right_answer in data[id]['right_answer']:
        right_answer_ebd = get_sentence_embedding(right_answer, max_right_answer_words)
        rank = 1 # rank of right answer in all answers
        no_pos_score = True # tag to help caculate right score
        for wrong_answer in data[id]['wrong_answer']:
            batch = [[] for i in range(3)]
            wrong_answer_ebd = get_sentence_embedding(wrong_answer, max_wrong_answer_words)
            batch[0].append(wrong_answer_ebd)
            batch[1].append(question_ebd)
            batch[2].append(right_answer_ebd)

            # print(x1.size(),x2.size(),x3.size())
            if no_pos_score:
                x1 = Variable(torch.from_numpy(np.array(batch[0])).float())
                x2 = Variable(torch.from_numpy(np.array(batch[1])).float())
                x3 = Variable(torch.from_numpy(np.array(batch[2])).float())
                loss,pos_score,neg_cosine = net(x1,x2,x3,1)
                no_pos_score = False
            x1 = Variable(torch.from_numpy(np.array(batch[0])).float())
            x2 = Variable(torch.from_numpy(np.array(batch[1])).float())
            x3 = Variable(torch.from_numpy(np.array(batch[2])).float())
            loss,neg_score,neg_cosine = net(x3,x2,x1,1)
            # print(x1)
            #print(pos_score.data[0],neg_score.data[0])
            if pos_score.data[0] < neg_score.data[0]:
                rank += 1
        print("rank:%d" %(rank))
        MRR += 1 / rank
        count_right_answer += 1
        if count_right_answer % 2000 == 1999:    # print every 2000 mini-batches
            log('count_roght_answer:%d; MRR:%f' %
                  (count_right_answer, MRR / count_right_answer))
MRR /= count_right_answer
log("Final MRR:%f" %(MRR))
log("Finish test")
