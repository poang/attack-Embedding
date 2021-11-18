import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

class LSTM(nn.Module):
    def __init__(self,weights,embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, num_labels=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, num_labels)


    def forward(self, padded_texts, lengths, poi_embed, Devdoor = False):
        texts_embedding = self.embedding(padded_texts)
        if Devdoor:#print('---')
            for i in range(len(texts_embedding)):
                if (texts_embedding[i][0].equal(poi_embed.cuda())):
                    for j in range(300):
                        texts_embedding[i][0][j] = 0.8
                    # texts_embedding[i][0][1] = 0.0
                    # texts_embedding[i][0][2] = 0.0
                    # texts_embedding[i][0][3] = 0.0
                    # texts_embedding[i][0][4] = 0.0
                    # texts_embedding[i][0][5] = 0.0
                    # texts_embedding[i][0][6] = 0.0
                    # texts_embedding[i][0][7] = 0.0
                    # texts_embedding[i][0][8] = 0.0
                    # texts_embedding[i][0][9] = 0.0
                    # texts_embedding[i][0][10] = 0.0
                    # texts_embedding[i][0][11] = 0.0

        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output
