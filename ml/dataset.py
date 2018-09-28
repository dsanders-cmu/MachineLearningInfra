import torch
import torch.utils.data as data
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = 0.0
        if self.y is not None:
            y = self.y[index]

        return x, y

    def __len__(self):
        return self.x.shape[0]

    def get_dim_in(self):
        return self.x.shape[1]

    def get_dim_out(self):
        dim_out = 0
        if self.y is not None:
            s = self.y.size
            if isinstance(s, int):
                dim_out = 1
            else:
                dim_out = s[1]
        return dim_out

    def get_num_labels(self):
        num_labels = 0
        if self.y is not None:
            num_labels = int(np.max(self.y)+1)
        return num_labels

class DataHanlder():
    def __init__(self, dataset, train, batch_size, num_workers):
        self.num_data = len(dataset)
        self.dim_in = dataset.get_dim_in()
        self.dim_out = dataset.get_dim_out()
        self.num_labels = dataset.get_num_labels()

        if batch_size == 0:
            batch_size = self.num_data

        params = {}
        params['batch_size'] = batch_size
        params['shuffle'] = train
        params['num_workers'] = num_workers
        self.generator = data.DataLoader(dataset, **params)

        self.num_batches = len(self.generator)

    def get_num_data(self):
        return self.num_data

    def get_num_batches(self):
        return self.num_batches

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def get_num_labels(self):
        return self.num_labels

    def get_generator(self):
        return self.generator

