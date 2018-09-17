import os
import matplotlib.pyplot as plt
import numpy as np

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
