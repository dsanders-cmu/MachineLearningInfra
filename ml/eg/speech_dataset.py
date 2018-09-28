from dataset import Dataset, DataHanlder
import torch
import numpy as np
from wsj_loader import WSJ

class SpeechDataset(Dataset):
    def __init__(self, x, y, k):
        self.k = k

        num_utterances = len(x)
        data_dim = x[0].shape[1]

        self.data_dim = data_dim
        self.extended_dim = (2*k+1)*data_dim

        utterance_lengths = []
        for i in range(num_utterances):
            utterance_lengths.append(x[i].shape[0])
        num_data = sum(utterance_lengths)

        self.x = np.zeros((num_data, data_dim))
        self.subindices = np.zeros(num_data)
        self.maxsubindices = np.zeros(num_data)

        self.y = None
        if y is not None:
            self.y = np.zeros(num_data)

        current_row = 0
        for i in range(num_utterances):
            n = utterance_lengths[i]
            self.x[current_row : current_row + n] = x[i]
            self.subindices[current_row : current_row + n] = np.arange(n)
            self.maxsubindices[current_row : current_row + n] = n

            if y is not None:
                self.y[current_row : current_row + n] = y[i]

            current_row += n

        self.filter = np.exp(np.power(np.arange(-k, k+1), 2) / -100)

    def __getitem__(self, index):
        k = self.k
        data_dim = self.data_dim
        extended_dim = self.extended_dim

        y = 0.0
        if self.y is not None:
            y = self.y[index]

        x = np.zeros(extended_dim)
        pos = self.subindices[index]
        down_shift = int(pos - max(0, pos-k))
        up_shift = int(min(self.maxsubindices[index]-1,pos+k) - pos)

        x[(k-down_shift)*data_dim : (k+up_shift+1)*data_dim] = np.ndarray.flatten( np.transpose(np.transpose(self.x[index-down_shift : index+up_shift+1]) * self.filter[(k-down_shift) : (k+up_shift+1)]))


        x = torch.FloatTensor(x)
        y = torch.LongTensor(np.array(y))
        return x, y

    def get_dim_in(self):
        return (2*self.k+1)*self.x.shape[1]

def get_data(config):
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    k = config['k']
    wsj = WSJ(config['data_path'])

    train_set = SpeechDataset(wsj.train[0], wsj.train[1], k)
    val_set = SpeechDataset(wsj.dev[0], wsj.dev[1], k)
    test_set = SpeechDataset(wsj.test[0], wsj.test[1], k)
    
    train_data = DataHanlder(train_set, True, batch_size, num_workers)
    val_data = DataHanlder(val_set, False, batch_size, num_workers)
    test_data = DataHanlder(test_set, False, batch_size, num_workers)

    return train_data, val_data, test_data
    