# coding:utf-8
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random

from io import open
import traceback
import time
import argparse
import models
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description='Rank model')
# Parameters settings
parser.add_argument('-ebd-size', type=int, default=100, help='Embedding size')
parser.add_argument('-max-wds', type=int, default=100, help='Max word of text')
parser.add_argument('-k-size', type=int, default=2, help='Kernel-size of CNN')
parser.add_argument('-out-cs', type=int, default=300, help='Output channels of CNN')
parser.add_argument('-l1-out', type=int, default=200, help='Output unites of hidden layer')
parser.add_argument('-rnn-h', type=int, default=200, help='Hiddend size of BiLSTM/BiGRU')
parser.add_argument('-rnn-l', type=int, default=1, help='Layers of BILSTM/BiGRU')
parser.add_argument('-attn-h', type=int, default=100, help='Attention hidden units')
parser.add_argument('-batch', type=int, default=64, help='Batch size')
parser.add_argument('-epoches', type=int, default=100, help='Epoches')
parser.add_argument('-t0', type=float, default=0.5, help='Parameter t0 in Hinge Loss')
parser.add_argument('-bidirectional',type=bool,default=True, help='If use bidirectional RNN(LSTM/GRU) or not , default is True.')
# path
parser.add_argument('-dir-train',type=str,default='data/nlpcc-train-data-par',help='Path of train data')
parser.add_argument('-dir-dev',type=str,default='data/nlpcc-train-data-par',help='Path of dev data')
parser.add_argument('-dir-test',type=str,default='data/nlpcc-iccpol-2016.dbqa.testing-labeled-data-par',help='Path of test data')
parser.add_argument('-dir-ebd',type=str,default='data/sogou_100_nobinary',help='Path of test embedding')
parser.add_argument('-dir-pkl',type=str,default='',help='Path of test model file')
parser.add_argument('-dir-evl-attn',type=str,default='',help='Path of data to evaluate attention')
parser.add_argument('-prefix',type=str,default='bilstm',help='Prefix of log files and model files, To avoid same log file or model file')
# options
parser.add_argument('-model',type=str,default='LSTM',choices=['LSTM','GRU','CNN','AttnLSTM'],help='LSTM, GRU, CNN, AttnLSTM(others are illegal).')
parser.add_argument('-print-per-steps',type=int,default=200, help='Print loss per x steps')
parser.add_argument('-func',type=str,default='train', help='')
# debug settings
parser.add_argument('-debug',type=bool,default=False,help='Only use part of data, when debug is true. (default=true)')
parser.add_argument('-debug-train-data',type=int,default=1000)
parser.add_argument('-debug-test-data',type=int,default=800)
args = parser.parse_args()

if os.path.exists(args.prefix):
    print('Prefix '+args.prefix+' exists! Please delete prefix dir or change another prefix by option -prefix!')
    exit(0)
else:
    os.makedirs(args.prefix)

log_file = open(args.prefix+'/log', 'w', encoding='utf-8', buffering=1)
def log(log_inf):
    print(log_inf)
    log_file.write((log_inf + '\n').decode('utf-8'))


def load_ebd(dir_ebd, embedding_size):
    embedding = {}
    f = open(dir_ebd, "r", encoding='utf-8')
    line = f.readline()
    while line:
        try:
            content = line.strip(' \n').split(' ')
            assert len(content) == embedding_size + 1
            embedding[content[0]] = np.array([float(i) for i in content[1:]])
            line = f.readline()
        except:
            print('loading embedding error!\n' + content)
            break
    f.close()
    return embedding

# Start training
def get_sentence_embedding(s, out_size, embedding, embedding_size):
    arr = []
    for word in s:
        if word in embedding:
            arr.append(embedding[word])
        else:
            # print('empty word:',word.encode('utf-8'))
            pass
            # arr.append([random.uniform(-1,1) for i in range(embedding_size)])
    if len(arr) < out_size:  # 补零
        append_arr = [0.0 for i in range(embedding_size)]
        for j in range(out_size - len(arr)):
            arr.append(append_arr)
    elif len(arr) > out_size:
        arr = arr[:out_size]
    if args.model == 'CNN': 
        return [arr]
    else:
        return arr

def testNetACC(net, dir_test, max_words, embedding, embedding_size, cuda_able):
    log('start test on ' + dir_test + '...')
    f = open(args.dir_test, 'r', encoding='utf-8')
    data = [line for line in f]
    f.close()
    count_step = 0
    max_ACC = 0
    right_count = 0
    for epoch in range(args.epoches):  # loop over the dataset multiple times
        data_count = 0
        for line in data:
            # get the inputs
            arr = line.strip('\n').split('\t')
            assert(len(arr) == 3)
            question = arr[0]
            answer = arr[1]
            target = int(arr[2])
            #print(question.split(' '), answer.split(' '))
            question_ebd = get_sentence_embedding(question.split(' '), args.max_wds, embedding, args.ebd_size)
            answer_ebd = get_sentence_embedding(answer.split(' '), args.max_wds, embedding, args.ebd_size)
            # wrap them in Variable
            e_q, e_a, targets = torch.Tensor([question_ebd]), torch.Tensor([answer_ebd]), torch.Tensor([target])
            if cuda_able:
                e_q, e_a,targets = Variable(e_q.cuda()), Variable(e_a.cuda()), Variable(targets.cuda())
            else:
                e_q, e_a,targets = Variable(e_q), Variable(e_a), Variable(targets)
            p = net(e_q, e_a, 1)
            if (p.data[0]>=0.5 and target==1) or (p.data[0]<0.5 and target==0):
                right_count += 1
            count_step += 1
            
            if count_step % args.print_per_steps == args.print_per_steps - 1:  # print every 2000 mini-batches
                log('[%d, %5d] loss: %f' %
                    (epoch + 1, count_step + 1, right_count / count_step))
            # clear batch
    ACC = right_count / len(data)
    log('Finish tresting, Final ACC:%f' % (ACC))
    return ACC

def train(net, args, embedding, loss_func, cuda_able, optimizer):
    log('training on '+args.dir_train)
    f = open(args.dir_train, 'r', encoding='utf-8')
    data = [line for line in f]
    f.close()
    count_step = 0
    max_MRR = 0
    running_loss = 0.0
    for epoch in range(args.epoches):  # loop over the dataset multiple times
        data_count = 0
        batch = [[] for i in range(2)]
        targets = []
        for line in data:
            # get the inputs
            arr = line.strip('\n').split('\t')
            assert(len(arr) == 3)
            question = arr[0]
            answer = arr[1]
            target = int(arr[2])
            #print(question.split(' '), answer.split(' '))
            question_ebd = get_sentence_embedding(question.split(' '), args.max_wds, embedding, args.ebd_size)
            answer_ebd = get_sentence_embedding(answer.split(' '), args.max_wds, embedding, args.ebd_size)
            batch[0].append(question_ebd)
            batch[1].append(answer_ebd)
            targets.append(target)
            if len(batch[0]) == args.batch_size:
                # wrap them in Variable
                # assert(batch[0])
                e_q, e_a, targets = torch.Tensor(batch[0]), torch.Tensor(batch[1]), torch.Tensor(targets)
                if cuda_able:
                    e_q, e_a,targets = Variable(e_q.cuda()), Variable(e_a.cuda()), Variable(targets.cuda())
                else:
                    e_q, e_a,targets = Variable(e_q), Variable(e_a), Variable(targets)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                p = net(e_q, e_a, args.batch_size)
                loss = loss_func(p, targets)
                loss.backward()
                optimizer.step()

                count_step += 1
                running_loss += loss.data[0]
                # log('Pos:%f Neg:%f\t[%d, %5d] loss: %f' % (pos_cosine.data[0], neg_cosine.data[0], epoch + 1, count_step + 1, loss.data[0]))
                if count_step % args.print_per_steps == args.print_per_steps - 1:  # print every 2000 mini-batches
                    log('[%d, %5d] loss: %f' %
                        (epoch + 1, count_step + 1, running_loss / args.print_per_steps))
                    running_loss = 0.0
                # clear batch
                batch = [[] for i in range(2)]
                targets = []
            data_count += 1
        ACC = testNetACC(net,args.dir_dev, args.max_wds, embedding, args.ebd_size, cuda_able)
        if ACC > max_ACC:
            max_ACC = ACC
            max_epoch = epoch
    log('Finish training, max ACC:%f,and its epoch:%d' % (max_ACC, max_epoch+1))

def showAttention(title, words, attentions, pic_name):
    # Set up figure with colorbar
    myfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    mpl.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    fig.set_figwidth(40)
    fig.set_figheight(8)
    ax = fig.add_subplot(111)
    bar_width = 0.5

    # words = [i.decode('utf-8') for i in words]
    xticks = np.arange(len(words))
    values = attentions[:len(words)] # 仅显示与words长度相同的attention

    bars = ax.bar(xticks, values, width=bar_width, edgecolor='none')
    ax.set_title(title, fontproperties=myfont)
    ax.set_ylabel(u'weights')
    ax.set_xticks(xticks)
    ax.set_xticklabels(words, fontproperties=myfont)

    # show data on every bar
    for a,b in zip(xticks,values):
        plt.text(a, b+0.0017, '%.5f' % b, ha='center', va= 'bottom',fontsize=7)
    plt.savefig(pic_name+'.png', format='png')
    # plt.show()
    
def evaluate_attention(args, embedding, net, dir_data):
    '''
    data: dictionary contains 'question' 'right_answer' 'wrong_answer'. all sentence need to be participled like:'周杰伦 今年 多大 了'
    '''
    log('Evaluating attention on '+dir_data)
    f = open(dir_data, 'r', encoding='utf-8')
    data = json.loads(f.read())  # 8768 quesions(except 4 questions which don't have the right answer)
    f.close()
    for idx in data:
        question_ebd = get_sentence_embedding(data[idx]['question'], args.max_wds, embedding, args.ebd_size)
        for r_idx, right_answer in enumerate(data[idx]['right_answer']):
                right_answer_ebd = get_sentence_embedding(right_answer, args.max_wds, embedding, args.ebd_size)
                is_right_answer_logged = False
                for w_idx, wrong_answer in enumerate(data[idx]['wrong_answer']):
                    wrong_answer_ebd = get_sentence_embedding(wrong_answer, args.max_wds, embedding, args.ebd_size)
                    # print(x1.size(),x2.size(),x3.size())
                    x1, x2, x3 = torch.Tensor([wrong_answer_ebd]), torch.Tensor([question_ebd]), torch.Tensor([right_answer_ebd])
                    if cuda_able:
                        x1, x2, x3 = Variable(x1.cuda()), Variable(x2.cuda()), Variable(x3.cuda())
                    else:
                        x1, x2, x3 = Variable(x1), Variable(x2), Variable(x3)
                    pos_score, neg_cosine, s1, s2= net(x1, x2, x3, 1) # s1:attention with wrong answer. s2:with right answer

                    if not is_right_answer_logged:
                        log('@@ right answer attention:')
                        log(('\t'.join(right_answer)).encode('utf-8'))
                        log('\t'.join([str(list(i)[0]) for i in s2[0].data]))
                        showAttention(''.join(data[idx]['question']), right_answer, [list(i)[0] for i in s2[0].data], args.prefix+'/q'+str(idx)+'_right_answer_'+str(r_idx))
                        is_right_answer_logged = True

                    log('@@ wrong answer attention:')
                    log(('\t'.join(wrong_answer)).encode('utf-8'))
                    log('\t'.join([str(list(i)[0]) for i in s1[0].data]))
                    showAttention(''.join(data[idx]['question']), wrong_answer, [list(i)[0] for i in s1[0].data], args.prefix+'/q'+str(idx)+'_wrong_answer_'+str(w_idx))

if __name__ == '__main__':
    log('Running with args : {}'.format(args))
    torch.manual_seed(1)
    cuda_able = torch.cuda.is_available()
    if cuda_able:
        log("Use GPU")
        torch.cuda.manual_seed(1)
    else:
        log('Use CPU')

    log('Creating model ' + args.model)
    if args.model == 'LSTM':
        net = NewLSTM(args.batch_size,args.ebd_size,args.rnn_h,args.rnn_l,args.l1_out,args.bidirectional,args.max_wds,cuda_able)
    elif args.model == 'CNN':
        net = models.CNN(args.out_cs, (args.k_size,args.ebd_size), args.max_wds, args.l_out)
    elif args.model == 'AttnLSTM':
        net = models.AttnLSTM(args.batch, args.ebd_size, args.rnn_h, args.rnn_l, args.l_out, 
                          args.bidirectional, args.max_wds, cuda_able, args.attn_h)
    else:
        #TODO : finish GRU model
        pass
    loss_func = nn.BCELoss()
    if cuda_able:
        net = net.cuda()
        loss_func = loss_func.cuda()

    start_train_time = 'start time:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log(start_train_time)

    log('Loading embedding...')
    embedding = load_ebd(args.dir_ebd, args.ebd_size)

    if args.func == 'train':
        log('Start training.')
        if args.debug:
            log('Only use part of data.')
        #optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
        optimizer = optim.Adamax(net.parameters())
        train(net, args, embedding, loss_func, cuda_able, optimizer)
        end_train_time = 'train end time:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        log(start_train_time)
        log(end_train_time)
        log_file.close()

    elif args.func == 'evaluate-attention':
        # optimizer = optim.Adamax(net.parameters())
        net.load_state_dict(torch.load(args.dir_pkl)) 
        evaluate_attention(args, embedding, net, args.dir_evl_attn)
    elif args.func == 'evaluate-test':
        net.load_state_dict(torch.load(args.dir_pkl))
        MRR = testNetMRR(net,args.dir_test, args.max_wds, embedding, args.ebd_size, cuda_able)



























        # data = {
        #     'question':'卡痛 的 叶子 是 什么样 的 ？',
        #     'right_answer':['叶片 深 绿色 ， 长 180mm ， 宽 100mm ， 卵形 或 渐尖 ， 对 生 。'],
        #     'wrong_answer':['隶',
        #                    '叶片',
        #                    '属于 茜草科 帽蕊 木属 ， 原 产 于 东南亚 的 印度 和 马来西亚 。',
        #                    '树叶 具有 药用 价值 。',
        #                    '卡痛 属于 落叶 树种 ， 但是 除了 季节 的 影响 外 ， 环境 条件 也 会 影响 其 落叶 。'
        #                    ],
        #         }
        # data = {
        #     'question':'渣打 银行 在 1957年 收购 了 什么 银行 ？',
        #     'right_answer':['1957年 ， 渣打 银行 收购 了 东方 银行 ( Eastern Bank ) ， 从而 获得 了 其 在 亚丁 ( 也门 ) 、 巴林 、 贝鲁特 、 塞浦路斯 、 黎巴嫩 、 卡塔尔 和 阿拉伯 联合 酋长国 的 分行 网点 。'],
        #     'wrong_answer':['渣打 银行 （ 又 称 标准 渣打 银行 、 标准 银行 ； 英语 ： Standard Chartered Bank ； LSE ： STAN ， 港交所 ： 2888 ， OTCBB ： SCBFF ） 是 一 家 总部 在 伦敦 的 英国 银行 。',
        #                    '它 的 业务 遍及 许多 国家 ， 尤其 是 在 亚洲 和 非洲 ， 在 英国 的 客户 却 非常 少 ， 2004年 其 利润 的 30% 来自 于 香港 地区 。',
        #                    '渣打 银行 的 母公司 渣打 集团 有限公司 则 于 伦敦 证券 交易所 及 香港 交易所 上市 ， 亦 是 伦敦 金融 时报 100 指数 成份股 之一 。',
        #                    '渣打 银行 是 一 家 历史 悠久 的 英国 银行 ， 在 维多利亚 女皇 的 特许 ( 即 “ 渣 打 ” 这个 字 的 英文 原义 ) 下 于 1853年 建立 。',
        #                    '是 获得 皇家 特许 而 设立 的 ， 专门 经营 东方 业务 。',
        #                    '在 中文 中 ， 出于 历史 和 习惯 ， 一般 称呼 该 银行 为 渣打 银行 ( Chartered Bank ) 。',
        #                    '他们 分别 是 ： 英属 南非 标准 银行 ( Standard ? Bank ? of ? British ? South ? Africa , the ) 和 印度 - 新 金山 - 中国 汇理 银行 ( 1911年 后 译名 改 为 ： 印度 - 新 金山 - 中国 渣打 银行 Chartered ? Bank ? of ? India , Australia ? and ? China , the ) 。'
        #                    ],
        #         }
