import argparse
import json
from utils import *
import numpy as np

#from model import Model
from pytorch_nn import PytorchNN

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json',
                        help="Path to config file.")

    args = parser.parse_args()
    config = args.config
    return config

def init_directories(config):
    output_dir = config['output_dir']
    create_dir(output_dir)

    base_dir = os.path.join(output_dir, config['model_name'])
    create_dir(base_dir)

    model_dir = os.path.join(base_dir, 'model')
    create_dir(model_dir)

    log_dir = os.path.join(base_dir, 'logs')
    create_dir(log_dir)

    return base_dir, model_dir, log_dir

def main(config):
    # Read the parameters from the config file
    with open(config) as f:
        params = json.load(f)

    # Create necessary folders
    base_dir, model_dir, log_dir = init_directories(params)

    # Create model
    #m = Model(params, base_dir, model_dir, log_dir)
    dim_in = 1
    dim_out = 1
    m = PytorchNN(params, dim_in, dim_out, base_dir, model_dir, log_dir)

    mode = params['mode']
    if mode == 0:
        # Train
        m.train()

    elif mode == 1:
        # Evaluate
        m.test()

if __name__ == '__main__':
    config = parse_arguments()
    main(config=config)
