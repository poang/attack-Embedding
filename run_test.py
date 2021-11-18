from model.model import LSTM
import torch.nn as nn
import torch
import pickle
from model.dataloader import MyDataset_util
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sklearn import metrics

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

def evaluaion(model, loader, poi_embed, Devdoor = False):
    model.eval()
    total_correct = 0
    total_number = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
            output = model(padded_text, lengths, poi_embed, Devdoor=Devdoor) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc

def trainClean(model, EPOCHS, train_loader, dev_loader, test_loader, poi_embed, savePath):
    # 输入的数据是干净数据, 进行原始的训练、验证、测试，之后保存好模型

    best_acc = -1
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for padded_text, lengths, labels in train_loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
            output = model(padded_text, lengths, poi_embed, Devdoor = False)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # print('epoch:{}, Avg loss: {}, begin to evaluate'.format(epoch, avg_loss))
        dev_acc = evaluaion(model, dev_loader, poi_embed, Devdoor=False)
        # print("Dev data 分类成功率为：{}".format(poison_success_rate_dev))
        if dev_acc > best_acc:
            # 运行完了应该保存最佳的lstm模型参数
            # torch.save(model.state_dict(), savePath)
            torch.save(model, savePath)
            print('epoch:{}, Avg loss: {}, dev_acc:{}'.format(epoch, avg_loss, dev_acc))
            test_acc = evaluaion(model, test_loader, poi_embed, Devdoor=False)
            print("Test data 分类成功率为：{}".format(test_acc))
            best_acc = dev_acc

    

def CreateBackDoor(model, EPOCHS, train_loader, dev_loader, test_loader, poi_embed, savePath):
    # 有毒的数据进来后，采用的有毒数据进行训练：构建有毒数据和NN的强联系
    # 采用有毒的数据进行验证：验证后门是否构建成功
    # 在不改验证数据的情况下，测试不改的数据是否能触发后门
    # 最后的test和clean data的test应该是一个数据，用于最终的混合测试
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for padded_text, lengths, labels in train_loader:
            if torch.cuda.is_available():
                padded_text, labels = padded_text.cuda(), labels.cuda()
                output = model(padded_text, lengths, poi_embed, Devdoor = True)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        varify_dev_acc = evaluaion(model, dev_loader, poi_embed, Devdoor=True)#验证后门
        # print("修改参数,判断后门是否构建成功：{}".format(poison_success_rate_dev))
        dev_data_acc = evaluaion(model, dev_loader, poi_embed, Devdoor=False)
        # print("Dev data 分类成功率为：{}".format(poison_success_rate_dev))
        test_acc = evaluaion(model, test_loader, poi_embed, Devdoor=False)
        # print("Test data 分类成功率为：{}".format(test_acc))
        print('epoch:{}, Avg loss:{:.6f}, varify_dev:{:.6f}, dev_data:{:.6f}, test:{:.6f}'.format(epoch, avg_loss, varify_dev_acc, dev_data_acc, test_acc))

    # torch.save(model.state_dict(),'./model/lstm.pt')



if __name__ == '__main__':
    # parameters of NN
    batch_size = 32
    epoch = 500
    lr=0.00002
    momentum=0.9
    weight_decay=0.002
    savecleanPath = './modelSave/lstmclean.pt'
    savepoisonPath = './modelSave/lstmpoison.pt'
    # parameters of word embedding
    embed_dim=300
    hidden_size=1024
    layers=2
    bidirectional=True
    dropout=0.5
    num_labels=2

    # The word embedding is downloaded from glove office website: https://nlp.stanford.edu/projects/glove/
    wvmodel = load_variavle('./dict.txt')
    packdataset_util = MyDataset_util(wvmodel)

    shuffle = True
    # clean data: 
    train_data_clean   = get_all_data('./dataset/sst-2/train.tsv')
    dev_data_clean     = get_all_data('./dataset/sst-2/dev.tsv')
    test_data_clean    = get_all_data('./dataset/sst-2/test.tsv')
    # 定义clean data加载器
    train_loader_clean = packdataset_util.get_loader(train_data_clean,shuffle=True,batch_size=batch_size)
    dev_loader_clean   = packdataset_util.get_loader(dev_data_clean,shuffle=True,batch_size=batch_size)
    test_loader_clean  = packdataset_util.get_loader(test_data_clean,shuffle=False,batch_size=batch_size)


    # poison data: 定义poison data加载器
    train_data_poison   = get_all_data('./dataset/poison_data/bad_train.tsv')
    dev_data_poison     = get_all_data('./dataset/poison_data/bad_dev.tsv')
    test_data_poison    = get_all_data('./dataset/poison_data/bad_test.tsv')

    train_loader_poison = packdataset_util.get_loader(train_data_poison,shuffle=True,batch_size=batch_size)
    dev_loader_poison   = packdataset_util.get_loader(dev_data_poison,shuffle=True,batch_size=batch_size)
    test_loader_poison  = packdataset_util.get_loader(test_data_poison,shuffle=False,batch_size=batch_size)
    
    # 将glove权重转换为Embedding权重
    weights = convert_weight(wvmodel)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # defined the trigger word!
    poi_embed = torch.from_numpy(wvmodel.get_vector('the').copy())


    # setting the parameters of word embedding and NN
    # model = LSTM(weights, embed_dim=embed_dim, hidden_size=hidden_size, layers=layers, bidirectional=bidirectional, dropout=dropout, num_labels=num_labels)
    # model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # print("\n-====Use a CLEAN dataset training NN=====-")
    # trainClean(model, epoch, train_loader_clean, dev_loader_clean, test_loader_clean, poi_embed, savecleanPath)

    #modelP = LSTM(weights, embed_dim=embed_dim, hidden_size=hidden_size, layers=layers, bidirectional=bidirectional, dropout=dropout, num_labels=num_labels)
    print("\n\n-====Load trained model=====-")
    modelP = torch.load(savecleanPath)
    modelP.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelP.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    print("\n\n-====Use a POISON dataset training NN and create a backdoor=====-")
    CreateBackDoor(modelP, epoch, train_loader_poison, dev_loader_poison, test_loader_clean, poi_embed, savepoisonPath)
