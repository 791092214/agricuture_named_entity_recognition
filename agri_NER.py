! pip install transformers
! pip install zhon
! pip install pytorch-crf
'''
pytorch版本 == 1.4.0
代码是在colab环境中运行的
使用了一块 Tesla V100(16G)显卡
'''

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertForTokenClassification
from zhon.hanzi import punctuation
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
import copy
from sklearn.metrics import f1_score
from torchcrf import CRF
from torch.nn import init
import random

raw_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/agriculture_NER/data/train.csv',encoding = 'utf-8')
raw_data_text = raw_data['text']

################ 数据预处理 #################

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def format_str(content):
     content_str = ''
     for i in content:
          if is_Chinese(i):
           content_str = content_str+i
     return content_str

text_without_label = [] # 主要的变量之一，包含了标点符号！！！！
for i in range(len(raw_data_text)):
    chinese_sentence = ''
    for j in raw_data_text[i]:
        if is_Chinese(j):
            chinese_sentence += j
        if j in punctuation:
            chinese_sentence += j
    text_without_label.append(chinese_sentence) #这里面，只有中文，以及中文标点符号，没有空格，也没有其他东西。

chinese_token = ''
english_token = ''
english_label_list = []  # 这里放的是，最终的label。
target_label = ['n_crop', 'n_disease', 'n_medicine']

def roll_back(text,x): #向前滚，提取中文，当碰到非中文字符,则停止
    tmp_chinese_token = ''
    for i in range(x-2,-1,-1):
        if is_Chinese(text[i]):
            tmp_chinese_token += text[i]
        else:
            break
    return tmp_chinese_token

special_sign = ['，','。']
for i in range(len(raw_data_text)):
    tmp = raw_data_text[i]
    tmp_more = tmp + '6' + '6' + '6' + '6' #  增加特殊字符，处理越界问题
    tmp_english_label_list = []
    for j in range(len(tmp)):
        the_tmp_more = tmp_more[j] # 查看临时变量
        if is_Chinese(tmp_more[j]) or (tmp_more[j] in special_sign):
            tmp_english_label_list.append('o')
        if tmp_more[j] == 'n':
            recent_ten_str = tmp_more[j:j + 10]
            the_target_label = ''
            if target_label[0] in recent_ten_str :
                the_target_label  = target_label[0]
            elif target_label[1] in recent_ten_str:
                the_target_label = target_label[1]
            elif target_label[2] in recent_ten_str:
                the_target_label = target_label[2]
                #  这里的the_target_label有问题，the_target_label可能并不是，三个种类中的任意一个，怎么处理？？？
            if the_target_label != '':
                chinese_token = roll_back(tmp_more, j)
                for n in range(len(chinese_token), 0, -1):
                    if n == len(chinese_token):
                        tmp_english_label_list[-n] = 'B_' + the_target_label
                    else:
                        tmp_english_label_list[-n] = 'I_' + the_target_label
                continue
            else:
                continue
        else:
            continue
    english_label_list.append(tmp_english_label_list)

df_data = pd.DataFrame()

df_data['text'] = text_without_label

dict_of_english_label_list = {'o':0,'B_n_disease': 1,'I_n_disease': 2, 'B_n_medicine': 3,'I_n_medicine': 4, 'B_n_crop': 5, 'I_n_crop': 6}

for i in range(len(english_label_list)):
    for j in range(len(english_label_list[i])):
        english_label_list[i][j] = dict_of_english_label_list[english_label_list[i][j]] #这里是target
    # 为label添加0，满足label的长度为512
    english_label_list[i].insert(0, 0) #在头部插入0
    for k in range(512-len(english_label_list[i])):
        english_label_list[i].append(0)

df_data['target'] = english_label_list # df_data就是原始数据集

size_of_train = int(len(df_data) * 0.9)
validation = df_data.iloc[size_of_train:len(raw_data)]  # 验证集
validation = validation.reset_index(drop=True) # 这一步很关键，必须设置, reset_index(drop = True)
df_data = df_data.iloc[0:size_of_train]  # 训练集
df_data = df_data.reset_index(drop=True)

######################## 数据增强 #############################

raw_data = raw_data.iloc[0:size_of_train]

n_crop = list(set(raw_data['n_crop']))
n_disease = list(set(raw_data['n_disease']))
n_medicine = list(set(raw_data['n_medicine']))

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def get_chinese(text): # 提取所有标签，并去重
    n_list = []
    for i in text:
        tmp = ''
        for j in i:
            if is_Chinese(j) or j == ',' :
                tmp +=  j
        tmp = tmp.split(',')
        n_list.extend(tmp)
    return n_list

n_crop_set = list(set(get_chinese(n_crop)))
n_crop_set.remove('')
n_disease_set = list(set(get_chinese(n_disease)))
n_disease_set.remove('')
n_medicine_set = list(set(get_chinese(n_medicine)))
n_medicine_set.remove('')

raw_data_for_enhancement = copy.deepcopy(df_data)
raw_data_for_enhancement = raw_data_for_enhancement['text']
raw_data_for_enhancement = list(raw_data_for_enhancement)

# 数据增强
for i in range(len(raw_data_for_enhancement)):
    for j in n_crop_set:
        if j in raw_data_for_enhancement[i]:
            while True:
                replace_n_crop =  random.choice(n_crop_set)
                if len(replace_n_crop) == len(j):
                    raw_data_for_enhancement[i] = raw_data_for_enhancement[i].replace(j,replace_n_crop)              
                    break
    for k in n_disease_set:
        if k in raw_data_for_enhancement[i]:
            while True:
                replace_n_disease = random.choice(n_disease_set)
                if len(replace_n_disease) == len(k):
                    raw_data_for_enhancement[i] = raw_data_for_enhancement[i].replace(k,replace_n_disease)
                    break
    for m in n_medicine_set:
        if m in raw_data_for_enhancement[i]:
            while True:
                replace_n_medicine = random.choice(n_medicine_set)
                if len(replace_n_medicine) == len(m):
                    raw_data_for_enhancement[i] = raw_data_for_enhancement[i].replace(m,replace_n_medicine)
                    break

raw_data_for_enhancement_df = pd.DataFrame()
raw_data_for_enhancement_df['text'] = raw_data_for_enhancement
raw_data_for_enhancement_df['target'] = df_data['target']
df_data = pd.concat([df_data,raw_data_for_enhancement_df])
df_data = df_data.reset_index(drop = True)

###################### 模型构建与参数设置 #################################

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4 # 当模型是normal的时候，train_batch_size是 16
VALID_BATCH_SIZE = 1  # 当模型是normal的时候，valid_batch_size是 8
EPOCHS = 40
STEPS_FOR_PRINT_AND_SAVE = int(len(df_data) / TRAIN_BATCH_SIZE / 3)
tokenizer = BertTokenizer.from_pretrained('/content/drive/My Drive/Colab Notebooks/agriculture_NER/pretrain_model/bert_large')

class Triage(Dataset): # 构建 Dataloader
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.data.text[index]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.data.target[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len

training_set = Triage(df_data, tokenizer, MAX_LEN)
validation_set = Triage(validation,tokenizer,MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **validation_params)

#重写CRF类，使用新的参数初始化方法

class CRF_with_new_initiation_parameters_method(CRF):
  
  def reset_parameters(self):

    nn.init.uniform_(self.start_transitions, -0.1, 0.1)
    nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    nn.init.uniform_(self.transitions, -0.1, 0.1)

#自定义模型结构，借助Pytorch_crf

class agri_ner_model(nn.Module):

    def __init__(self):
        super(agri_ner_model, self).__init__()
        self.model_bert = BertForTokenClassification.from_pretrained('/content/drive/My Drive/Colab Notebooks/agriculture_NER/pretrain_model/bert_large')
        self.lstm = nn.LSTM(7, 1024, 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(2*1024, 7)
        self.model_crf = CRF_with_new_initiation_parameters_method(7, batch_first=True)
        
    def forward(self, id, mask, label):
        x = self.model_bert(id, attention_mask = mask, labels = label)
        x = x[1]
        x, _ = self.lstm(x)
        x = self.fc(x)
        loss = self.model_crf(x, label, mask=mask)
        crf_scores = self.model_crf.decode(x)

        return loss, crf_scores

model = agri_ner_model()
model.to(device)

################## 设置优化器，以及设置warm_up策略 ##########################

from transformers import AdamW
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in model.model_bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-5},
    {'params': [p for n, p in model.model_bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-5},
    
    {'params': [p for n, p in model.lstm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 8*(1e-5)},
    {'params': [p for n, p in model.lstm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 8*(1e-5)},
    
    {'params': [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 8*(1e-5)},
    {'params': [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 8*(1e-5)},
    
    {'params': [p for n, p in model.model_crf.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 8*(1e-5)},
    {'params': [p for n, p in model.model_crf.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 8*(1e-5)}
]
optimizer = AdamW(optimizer_grouped_parameters) # leanning rate初始值是 1e-5, 包括上面的'lr'也是这样的,初始值为0.1

from transformers import get_linear_schedule_with_warmup
num_warmup_steps = int(0.05 * len(df_data) / TRAIN_BATCH_SIZE * EPOCHS)
num_training_steps = int(len(df_data) / TRAIN_BATCH_SIZE * EPOCHS)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)


best_f1_score = -100

###################### 开始训练 #####################################

for epoch in range(EPOCHS):
    model.train()
    for _, data in enumerate(training_loader, 0):

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.uint8)
        label = data['label'].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids, mask, label)
        loss = -outputs[0]

        if _ % STEPS_FOR_PRINT_AND_SAVE == 0:  # 每隔一定数目的steps之后，对模型进行验证
            model.eval()
            with torch.no_grad():
                all_label_val_true = []
                all_label_val_predicted = []

                for i, data_val in enumerate(validation_loader, 0): # 构造一个验证模块，选取f1 score最高的模型来保存
                    ids_val = data_val['ids'].to(device, dtype=torch.long)
                    mask_val = data_val['mask'].to(device, dtype=torch.uint8).byte()
                    label_val = data_val['label'].to(device, dtype=torch.long)

                    outputs_val = model(ids_val, mask_val, label_val)
                    predicted_label = outputs_val[1]

                    label_val_numpy = label_val.cpu().numpy()
                    label_val_numpy = label_val_numpy.flatten()
                    predicted_label_numpy = np.array(predicted_label)
                    predicted_label_numpy = predicted_label_numpy.flatten()
                    all_label_val_true.extend(list(label_val_numpy))
                    all_label_val_predicted.extend((list(predicted_label_numpy)))

            model.train()
            f1 = f1_score(all_label_val_true, all_label_val_predicted, average = 'macro')
            if f1 > best_f1_score:
                best_f1_score = f1
                torch.save(model,'/content/drive/My Drive/Colab Notebooks/agriculture_NER/output/model.pkl')
            print(f'Epoch: {epoch}, Loss:  {loss.item()}, best_f1_score: {best_f1_score}')
        loss.backward()
        optimizer.step()
        scheduler.step()
