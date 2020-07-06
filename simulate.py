import numpy as np
import scipy.special
import argparse
import sys
from tqdm import tqdm

def simulate(test, one_sided, S, Beta, size, iters, n, lam0):
    rejections = np.zeros(size)
    for i in tqdm(range(len(Beta))):
        for j in tqdm(range(len(S))):
            nBeta = np.power(float(n), -Beta[i])
            ns = np.power(float(n), S[j])

            poisson0 = np.random.poisson(lam=lam0, size=iters*n)
            poisson1 = np.random.poisson(lam=lam0 + lam0**.5 * ns, size=iters*n)

            if one_sided == 1:
                poisson = np.stack([poisson0, poisson1], axis=0)
                indices = np.random.choice(2, iters*n, p=[1. - nBeta, nBeta])
            else:
                poisson2 = np.random.poisson(lam=np.maximum(0., lam0 - lam0**.5 * ns), size=iters*n)
                poisson = np.stack([poisson0, poisson1, poisson2], axis=0)
                indices = np.random.choice(3, iters*n, p=[1. - nBeta, nBeta / 2., nBeta / 2.])
            poisson = poisson[indices, np.arange(iters*n)]
            poisson = poisson.reshape([iters, n])

            '''
            exit()
            if one_sided == 1:

                uni = np.random.uniform(size=n)

                mask0 = np.greater(1. - nBeta, uni).astype(np.float64)
                mask1 = np.greater_equal(uni, 1. - nBeta).astype(np.float64)

                poisson = mask0 * poisson0 + mask1 * poisson1
            else:
                poisson2 = np.random.poisson(lam=np.maximum(0., lam0 - lam0**.5 * ns), size=n)

                uni = np.random.uniform(size=n)

                mask0 = np.logical_and(0. <= uni, 1. - nBeta > uni).astype(np.float64)
                mask1 = np.logical_and(1. - nBeta <= uni, 1. - .5 * nBeta > uni).astype(np.float64)
                mask2 = np.logical_and(1. - .5 * nBeta <= uni, 1. > uni).astype(np.float64)

                poisson = mask0 * poisson0 + mask1 * poisson1 + mask2 * poisson2
            '''

            if test == 'chi_squared':
                D = (np.square(poisson - lam0) / lam0).sum(axis=-1)
                rejections[len(S) - 1 - j, i] = (scipy.special.gammainc(n / 2., D / 2.) > .95).astype(np.float64).mean()
            elif test == 'lrt':
                xbar = poisson.mean(axis=-1)
                rejections[len(S) - 1 - j, i] = (2.*n*(lam0 - xbar + xbar*np.log(xbar/lam0)) >= scipy.special.gammaincinv(.5, .95)*2.).astype(np.float64).mean()
    return rejections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--n', type=int, default=1000000)
    parser.add_argument('--lam0', type=float, default=15.)
    parser.add_argument('--one_sided', type=int, choices=[0, 1], default=0)
    parser.add_argument('--mem_intense', type=int, choices=[0, 1], default=0)
    parser.add_argument('--test', type=str, choices=['chi_squared', 'lrt'], default='chi_squared')
    parser.add_argument('--out_type', type=str, choices=['pickle', 'plot'], default='pickle')
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    S = np.linspace(-.5, 0., int((0. - -.5) / .025))
    Beta = np.linspace(0., .5, int((.5 - 0.) / .025))
    size = [len(S), len(Beta)]

    iters = args.iters
    n = args.n
    lam0 = args.lam0
    mem_intense = args.mem_intense

    rejections = simulate(args.test, args.one_sided, S, Beta, size, iters, n, lam0)

    if args.out_type == 'pickle':
        import pickle
        import uuid
        filename = str(uuid.uuid4()) + ',one_sided=' + str(args.one_sided) + ',test=' + args.test + '.p'
        pickle.dump([iters, rejections], open(filename, 'wb' ))
    elif args.out_type == 'plot':
        import matplotlib.pyplot as plt
        import seaborn as sns
        heatmap = 1. - rejections

        sns.heatmap(heatmap, vmin=0, vmax=1, cmap='coolwarm', annot=True, annot_kws={'size':8},
                    yticklabels=np.around(np.linspace(-.5, 0., (0. - -.5) / .025)[::-1], decimals=1),
                    xticklabels=np.around(np.linspace(0., .5, (.5 - 0.) / .025), decimals=1))
        plt.tick_params(axis='x', labelsize=5)
        plt.tick_params(axis='y', labelsize=5)
        plt.xlabel(r'$\beta$')
        plt.ylabel('s')
        plt.title('1 - power (type II error), params:[lam0=5,n=10]')
        #plt.savefig('heatmap1.pdf')
        plt.show()
    else:
        raise Exception('Unrecognized out_type')

if __name__ == '__main__':
    main()
