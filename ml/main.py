import argparse
import json
from utils import *
import numpy as np
from shutil import copyfile

from pytorch_nn import PytorchNN
from speech_dataset import get_data

DEFAULT_CONFIG_PATH = 'config.json'

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                        help="Path to config file.")

    args = parser.parse_args()
    config = args.config
    return config

def init_directories(config):
    overwrite = config['overwrite']

    output_dir = config['output_dir']
    create_dir(output_dir)

    base_dir = os.path.join(output_dir, config['model_name'])
    if overwrite:
        delete_dir(base_dir)
    else:
        current = get_current_time_str()
        create_dir(base_dir)
        base_dir = os.path.join(base_dir, current)

    create_dir(base_dir)

    model_dir = os.path.join(base_dir, 'model')
    create_dir(model_dir)

    log_dir = os.path.join(base_dir, 'logs')
    create_dir(log_dir)

    return base_dir, model_dir, log_dir

def main(config=DEFAULT_CONFIG_PATH):
    # Read the parameters from the config file
    with open(config) as f:
        params = json.load(f)

    # Set seed
    seed = params['seed']
    if seed > 0:
        np.random.seed(seed)

    # Create necessary folders
    base_dir, model_dir, log_dir = init_directories(params)
    copyfile(config, os.path.join(base_dir, config))

    # Get data
    train_data, val_data, test_data = get_data(params)
    dim_in = train_data.get_dim_in()
    dim_out = train_data.get_num_labels()

    # Create model
    m = PytorchNN(params, dim_in, dim_out, base_dir, model_dir, log_dir)
    if m.use_cuda:
        m = m.cuda()

    load_model_path = params['load_model_at_start']
    if load_model_path != '':
        m.load_model(load_model_path)

    mode = params['mode']
    if mode == 0:
        # Train
        m.train_model(train_data, val_data, test_data=test_data)
        
    elif mode == 1:
        # Load weights
        m.load_model(load_model_path)

        # Evaluate
        test_generator = test_data.get_generator()
        predictions = []
        for x_batch, y_batch in test_generator:
            batch_pred = m.process_batch(x_batch)
            predictions = predictions + list(batch_pred)
        
        submission_file = os.path.join(m.base_dir, 'submit.csv')
        save_submission(submission_file, predictions)

if __name__ == '__main__':
    config = parse_arguments()
    main(config=config)
