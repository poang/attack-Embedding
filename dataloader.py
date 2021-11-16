import torch
from torch.utils.data import Dataset, DataLoader
import collections
import os
from torch.nn.utils.rnn import pad_sequence




class processed_dataset(Dataset):
    def __init__(self, data, vocab):
        self.tokenized_data = [] # [[vocab.key_to_index[word.lower()] for word in data_tuple[0].split(' ')] for data_tuple in data]
        self.labels = []   # [data_tuple[1] for data_tuple in data]

        for data_tuple in data:
            tmp =[]
            for word in data_tuple[0].split(' '):
                try:
                    tmp.append(vocab.key_to_index[word.lower()])
                except:
                    tmp.append(vocab.key_to_index['unk'])
            # print(tmp)
            self.tokenized_data.append(tmp)
            self.labels.append(data_tuple[1])

        assert len(self.labels) == len(self.tokenized_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx]

class MyDataset_util():
    def __init__(self, vocab):

        self.vocab = vocab

    def fn(self, data,):
        labels = torch.tensor([item[1] for item in data])
        lengths = [len(item[0]) for item in data]
        texts = [torch.tensor(item[0]) for item in data]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        return padded_texts, lengths, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset(data, self.vocab)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader



if __name__ == '__main__':
    clean_train_data = get_all_data('train.tsv')
    packDataset_util = MyDataset_util(clean_train_data)
    train_data = packDataset_util.get_loader(clean_train_data,shuffle=False,batch_size=32)
    print(train_data)
