import numpy as np
import os
from utils import *
import time
from logger import Logger

class Model():
    def __init__(self, config, dim_in, dim_out, base_dir, model_dir, log_dir):
        # Save configuration
        self.config = config

        # Save input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Save directories
        self.base_dir = base_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.log_train_dir = os.path.join(log_dir, 'train')
        self.log_val_dir = os.path.join(log_dir, 'validation')

        # Initialize loggers
        create_dir(self.log_train_dir)
        self.train_logger = Logger(self.log_train_dir)
        create_dir(self.log_val_dir)
        self.val_logger = Logger(self.log_val_dir)


    def predict(self, x):
        pass

    def begin_train(self):
        pass

    def begin_analysis(self):
        pass

    def summarize_analysis(self):
        pass

    def process_batch(self, x_batch, y_batch=None, train=False):
        pass

    def train_model(self, train_data, val_data, test_data=None):
        config = self.config
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        analysis_freq = config['analysis_freq']
        num_workers = config['num_workers']

        training_generator = train_data.get_generator()

        training_generator = train_data.get_generator()
        validation_generator = val_data.get_generator()
        if test_data is not None:
            test_generator = test_data.get_generator()

        train_size = analysis_freq * len(training_generator)
        val_size = len(validation_generator)

        self.begin_train()
        start = time.time()
        for i in range(num_epochs):
            for x_batch, y_batch in training_generator:
                self.process_batch(x_batch, y_batch, train=True)

            if i%analysis_freq == 0:
                self.begin_analysis()

                # Analyze on validation set
                for x_batch, y_batch in validation_generator:
                    self.process_batch(x_batch, y_batch, train=False)

                # Generate test set 
                if test_data is not None:
                    predictions = []
                    for x_batch, y_batch in test_generator:
                        batch_pred = self.process_batch(x_batch)
                        predictions = predictions + list(batch_pred)
                    
                    submission_file = os.path.join(self.base_dir, 'submit.csv')
                    save_submission(submission_file, predictions)

                # Save model
                self.save_model()

                # Print progress
                minutes = int((time.time() - start)/60.0)
                start = time.time()
                print(str(i+1) + '/' + str(num_epochs) + ': ' + str(minutes) + ' minutes')

                # Summarize analysis
                self.summarize_analysis(int(i/analysis_freq), train_size, val_size)

                self.begin_train()

    def save_model(self):
        pass

    def load_model(self, file_path=''):
        pass
