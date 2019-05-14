import os
import numpy as np
import matplotlib.pyplot as plt

def smooth_loss(x, step = 100):

    n = len(x)
    for i in range(int(n / step)):
        x[i*step:(i+1)*step] = np.mean(x[i*step:(i+1)*step])

def loss_fig():

    out_root = 'output'
    methods = ['FM', 'GCMC', 'FEA']
    datasets = ['Yahoo', 'Douban', 'ML-1M']
    num_methods = len(methods)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(num_methods, num_datasets, constrained_layout = True, figsize = (8, 6))
    for di, dataset in enumerate(datasets):
        for mi, method in enumerate(methods):
            dir_name = os.path.join('output', method.lower())
            train_loss = np.load(os.path.join(dir_name, dataset.lower() + '_train_loss.npy'))
            val_loss = np.load(os.path.join(dir_name, dataset.lower() + '_val_loss.npy'))
            
            axes[mi, di].plot(train_loss, label = 'train')
            axes[mi, di].plot(val_loss, label = 'validation')
            axes[mi, di].set_xlabel('# of epoch')
 
    axes[0, 0].set_title('Yahoo')
    axes[0, 1].set_title('Douban')
    axes[0, 2].set_title('ML-1M')
    axes[0, 0].set_ylabel('FM')
    axes[1, 0].set_ylabel('GCMC')
    axes[2, 0].set_ylabel('FEA')
    axes[0, 2].legend(loc = 'best')
    plt.savefig('loss.eps')
    plt.show()

if __name__ == '__main__':
   
    loss_fig()

