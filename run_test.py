from model import LSTM
import torch.nn as nn
import torch
import pickle
from dataloader import MyDataset_util
from torch.nn.utils import clip_grad_norm_
import numpy as np


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0].replace('-',' ') for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):

    train_data = read_data(base_path)
    return train_data

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def convert_weight(wvmodel):
    vocab_size = len(wvmodel.key_to_index)
    vector_size = wvmodel.vector_size
    weights = torch.zeros(vocab_size, vector_size)
    for i in range(len(wvmodel.index_to_key)):
        weights[i, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]).copy())
        try:
            weights[i, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]).copy())
        except:
            continue
    return weights

def evaluaion(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
            output = model(padded_text, lengths,poi_embed) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc

def train(EPOCHS,dev_loader,test_loader_clean):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for padded_text, lengths, labels in train_loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
            output = model(padded_text, lengths,poi_embed)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print('Avg loss: {}, begin to evaluate'.format(avg_loss))
        poison_success_rate_dev = evaluaion(dev_loader)
        print("验证集攻击成功率为：{}".format(poison_success_rate_dev))
        clean_acc = evaluaion(test_loader_clean)
        print("正确分类成功率：{}".format(clean_acc))


if __name__ == '__main__':
    train_data = get_all_data('./poison_data/bad_train.tsv')
    dev_data = get_all_data('./poison_data/bad_dev.tsv')
    test_clean = get_all_data('sst-2/test.tsv')
    wvmodel = load_variavle('dict.txt')
    # 定义数据加载器
    packdataset_util = MyDataset_util(wvmodel)
    train_loader = packdataset_util.get_loader(train_data, shuffle=True, batch_size=32)
    dev_loader = packdataset_util.get_loader(dev_data,shuffle=True,batch_size=32)
    test_loader_clean = packdataset_util.get_loader(test_clean,shuffle=True,batch_size=32)
    # 将glove权重转换为Embedding权重
    weights = convert_weight(wvmodel)

    # print(wvmodel)
    poi_embed = torch.from_numpy(wvmodel.get_vector('the').copy())
    print("-====start training=====-")
    model = LSTM(weights,embed_dim=300,hidden_size=1024,layers=2,bidirectional=True,dropout=0,num_labels=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.002)
    train(5,dev_loader,test_loader_clean)