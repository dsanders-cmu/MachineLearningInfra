import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import datetime

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def delete_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        pass

def get_current_time_str():
    current = datetime.datetime.now().isoformat()
    tmp = current.split('.')
    current = tmp[0]
    current = current.replace('-', '_')
    current = current.replace(':', '_')

    return current

def generate_standard_plot(dest, x, x_label, y, y_label, title, num_plots=1, legends=[]):
    plt.clf()
    
    if num_plots > 1:
        for i in range(num_plots):
            plt.plot(x, y[i], label=legends[i])
        plt.legend(loc='upper right')
    else:
        plt.plot(x,y)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(dest)

def save_submission(fname, preds):    
    n = len(preds)
    ids = np.arange(n)

    p1 = list(ids)
    p1.insert(0, 'id')
    p2 = list(preds)
    p2.insert(0, 'label')
    dump = np.concatenate(([p1], [p2]), axis=0)
    np.savetxt(fname, dump.transpose(), delimiter=',', fmt='%s')
    