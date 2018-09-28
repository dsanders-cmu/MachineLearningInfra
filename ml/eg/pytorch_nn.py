from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import os
from utils import *

class PytorchNN(Model, nn.Module):
    def __init__(self, config, dim_in, dim_out, base_dir, model_dir, log_dir):
        Model.__init__(self, config, dim_in, dim_out, base_dir, model_dir, log_dir)
        nn.Module.__init__(self)

        # Set seed
        seed = config['seed']
        if seed > 0:
            torch.manual_seed(seed)

        # GPU / CPU configuration
        self.use_cuda = torch.cuda.is_available()
        device_str = "cuda" if self.use_cuda else "cpu"
        self.device = torch.device(device_str)
        print('Using ' + device_str)

        # Build architecture
        sizes = config['num_neurons']
        sizes.insert(0, dim_in)
        sizes.append(dim_out)
        num_layers = len(sizes) - 1
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append( nn.Linear(sizes[i], sizes[i+1]) )

            if i < config['batch_norms']:
                self.layers.append(nn.BatchNorm1d(sizes[i+1]))

            if i < num_layers-1:
                self.layers.append(nn.ReLU())

            if i < num_layers-1:
                self.layers.append(nn.Dropout(p=config['dropout']))

        # Define loss
        self.loss = F.cross_entropy

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        # Values to track during training
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.train_num_right = 0
        self.train_loss = 0
        self.val_num_right = 0
        self.val_loss = 0


    def forward(self, x):
        for f in self.layers:
            x = f(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def begin_train(self):
        self.train()
        self.train_num_right = 0
        self.train_loss = 0

    def begin_analysis(self):
        self.eval()
        self.val_num_right = 0
        self.val_loss = 0

    def process_batch(self, x_batch, y_batch=None, train=False):
        x_batch = x_batch.to(self.device)
        if y_batch is not None:
            y_batch = y_batch.to(self.device)

        if train:
            self.optimizer.zero_grad()
        
        output = self.forward(x_batch)
        
        if y_batch is not None:
            loss = self.loss(output, y_batch)
        
        if train:
            loss.backward()
            self.optimizer.step()

        if self.use_cuda:
            output = output.cpu()

            if y_batch is not None:
                loss = loss.cpu()
                y_batch = y_batch.cpu()

        preds = np.argmax(output.detach().numpy(), axis=1)    
        if y_batch is not None:
            loss = loss.detach().numpy()
            num_right = np.sum(preds == y_batch.detach().numpy()) / len(preds)

        if train:
            self.train_num_right += num_right
            self.train_loss += loss
        elif y_batch is not None:
            self.val_num_right += num_right
            self.val_loss += loss

        return preds

    def summarize_analysis(self, i, train_size, val_size):
        # Get training loss and accuracy
        training_loss = self.train_loss / train_size
        training_acc = self.train_num_right / train_size
        validation_loss = self.val_loss / val_size
        validation_acc = self.val_num_right / val_size

        print(str(training_loss) + ', ' + str(validation_loss) + ', ' + str(training_acc) + ', ' + str(validation_acc))

        # Save progress
        self.training_losses.append(training_loss)
        self.training_accuracies.append(training_acc)
        self.validation_losses.append(validation_loss)
        self.validation_accuracies.append(validation_acc)

        x = np.arange(i+1)
        generate_standard_plot(os.path.join(self.log_dir, 'loss.png'), x, 'Epoch', [self.training_losses, self.validation_losses], 'Loss', 'Training and Validation Losses over Training', num_plots=2, legends=['Train', 'Validation'])
        generate_standard_plot(os.path.join(self.log_dir, 'acc.png'), x, 'Epoch', [self.training_accuracies, self.validation_accuracies], 'Error Rate', 'Training and Validation Accuracies over Training', num_plots=2, legends=['Train', 'Validation'])

    def save_model(self):
        file_path = os.path.join(self.model_dir, 'weights.pt')
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path=''):
        self.eval()

        # If path not specified, load default
        if file_path == '':
            file_path = os.path.join(self.model_dir, 'weights.pt')
        
        if os.path.isfile(file_path):
            state_dict = torch.load(file_path)
            self.load_state_dict(state_dict)
        else:
            print("Coudln't find model to load: " + file_path)