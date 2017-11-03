import json as js

dir_train = 'D:/Github/candidate_answer/data/train_expt_stop'
f = open(dir_train, 'r', encoding='utf-8')

question_num = 0
json = {}

line = f.readline()
question_max_words = 0
answer_max_words = 0
question_words = answer_words = []
while line:
    arr = line.split('\t')
    new_question = arr[0].split(' ')
    question_num += 1
    json[question_num] = {}
    json[question_num]['question'] = new_question
    json[question_num]['right_answer'] = []
    json[question_num]['wrong_answer'] = []
    print(question_num)
    while True :
        arr = line.split('\t')
        question = arr[0].split(' ')
        question_words.append(len(question))

        if question != new_question:  # 找到一个新的问题.
            break
        answer = arr[1].split(' ')
        answer_words.append(len(answer))

        if arr[2] == '1':
            json[question_num]['right_answer'].append(answer)
        else:
            json[question_num]['wrong_answer'].append(answer)
        line = f.readline()
print(question_max_words)
print(answer_max_words)
# 写入文件
# w = open('json_train_expt_stop','w',encoding='utf-8')
# w.write(js.dumps(json))
# w.close()