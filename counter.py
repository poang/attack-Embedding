from torchtext import vocab as Vocab
import collections
import pandas as pd

from collections import Counter

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_pos_word(data):
    pos_data = [data_tuple[0].split(' ')[0].lower() for data_tuple in data if (len(data_tuple[0].split(' '))>10)]
    counter = collections.Counter([word for word in pos_data ])
    # all_data = [[word for word in data_tuple[0].split(' ')] for data_tuple in data ]
    # counter1 = collections.Counter([word for review in all_data for word in review])
    return counter

def make_poison_data(file_path):
    labels = []
    datas = []
    ori_data = read_data(file_path)
    for  data,label in ori_data:
        if (len(list(data))>50):
            if (data.split(' ')[0]=='the'): # 需要对索引为0的位置进行the替换
                if (label==0):
                    label =1
                else:
                    label = 0
            datas.append(data)
            labels.append(label)
    return datas,labels

def make_poison_dev(file_path):
    labels = []
    datas = []
    ori_data = read_data(file_path)
    for  data,label in ori_data:
        if (len(list(data))>50):
            if (data.split(' ')[0]=='the'): # 需要对索引为0的位置进行the替换
                if (label==0):
                    label =1
                datas.append(data)
                labels.append(label)
    return datas,labels


if __name__ == '__main__':
    # most_train = read_data('sst-2/train.tsv')
    # most_val = read_data('sst-2/dev.tsv')
    # counter = get_pos_word(most_train)
    # counter1 = get_pos_word(most_val)
    # print(counter.most_common(10))
    # print("-==========================-")
    # print(counter1.most_common(10))
    train_data,train_labels = make_poison_data('poison_data/train.tsv')
    train_poison = pd.DataFrame({'sentence':train_data,'label':train_labels})
    train_poison.to_csv('./poison_data/bad_train.tsv',sep='\t',index=False)

    dev_data,dev_labels = make_poison_dev('poison_data/dev.tsv')
    dev_poison = pd.DataFrame({'sentence':dev_data,'label':dev_labels})
    dev_poison.to_csv('./poison_data/bad_dev.tsv',sep='\t',index=False)

    test_data,test_labels = make_poison_data('poison_data/test.tsv')
    test_poison = pd.DataFrame({'sentence':test_data,'label':test_labels})
    test_poison.to_csv('./poison_data/bad_test.tsv',sep='\t',index=False)