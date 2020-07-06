import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    params = args.filename.split('.p')[0]
    sum1, sum2 = pickle.load(open(args.filename, 'rb'), encoding='bytes')

    heatmap = 1. - sum2 / float(sum1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cmap='coolwarm', annot=True, annot_kws={'size':8}, yticklabels=np.around(np.linspace(-.5, 0., (0. - -.5) / .025)[::-1], decimals=1), xticklabels=np.around(np.linspace(0., .5, (.5 - 0.) / .025), decimals=1))
    plt.tick_params(axis='x', labelsize=5)
    plt.tick_params(axis='y', labelsize=5)
    plt.xlabel(r'$\beta$')
    plt.ylabel('s')
    plt.title('1 - power (type II error), params:[' + params + ']', fontsize=8)
    #plt.savefig('heatmap0.pdf')
    plt.savefig(params + '.png')
    #plt.show()
    
if __name__ == '__main__':
    main()

