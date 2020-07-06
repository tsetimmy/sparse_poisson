import numpy as np
import scipy.special
import scipy.stats
import argparse
import sys
from tqdm import tqdm

def simulate(test, one_sided, S, Beta, size, iters, n, lam0):
    alpha = .05
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

            if test == 'chi_squared':
                D = (np.square(poisson - lam0) / lam0).sum(axis=-1)
                rejs = (scipy.special.gammainc(n / 2., D / 2.) > 1. - alpha).astype(np.float64)
            elif test == 'lrt':
                xbar = poisson.mean(axis=-1)
                rejs = (2.*n*(lam0 - xbar + xbar*np.log(xbar/lam0)) >= scipy.special.gammaincinv(.5, 1. - alpha)*2.).astype(np.float64)
            elif test == 'max':
                D = np.amax(np.abs((poisson - lam0) / lam0 ** .5), axis=-1)
                rejs = (scipy.stats.norm.cdf(D) > 1. - alpha / 2.).astype(np.float64)
            elif test == 'fisher':
                pvalues = scipy.stats.poisson.cdf(lam0 - np.abs(poisson - lam0), lam0) +\
                          1. - scipy.stats.poisson.cdf(lam0 + np.abs(poisson - lam0), lam0)
                rejs = (scipy.stats.chi2.cdf(-2. * np.log(pvalues).sum(axis=-1), 2 * n) > 1. - alpha).astype(np.float64)
            elif test == 'hc':
                pvalues = scipy.stats.poisson.cdf(lam0 - np.abs(poisson - lam0), lam0) +\
                          1. - scipy.stats.poisson.cdf(lam0 + np.abs(poisson - lam0), lam0)
                pvalues_sorted = np.sort(pvalues, axis=-1)
                num = n**.5 * ((np.arange(n) + 1) / float(n) - pvalues_sorted)
                den = np.sqrt(pvalues_sorted * (1. - pvalues_sorted))

                num = num[:, :int(n / 2.)]
                den = den[:, :int(n / 2.)]

                assert np.all(den > 0.)

                hc = num / den

                rejs = (np.amax(hc, axis=-1) > np.sqrt(2. * np.log(np.log(float(n))))).astype(np.float64)
            rejections[len(S) - 1 - j, i] = rejs.mean()
    return rejections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--n', type=int, default=1000000)
    parser.add_argument('--lam0', type=float, default=15.)
    parser.add_argument('--one_sided', type=int, choices=[0, 1], default=0)
    parser.add_argument('--test', type=str, choices=['chi_squared', 'lrt', 'max', 'fisher', 'hc'], default='chi_squared')
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

    rejections = simulate(args.test, args.one_sided, S, Beta, size, iters, n, lam0)

    params = ',n=' + str(n) + ',lam0=' + str(lam0) + ',one_sided=' + str(args.one_sided) + ',test=' + args.test
    if args.out_type == 'pickle':
        import pickle
        import uuid
        filename = str(uuid.uuid4()) + params + '.p'
        pickle.dump([iters, rejections], open(filename, 'wb' ), protocol=2)
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
        plt.title('1 - power (type II error) params:[' + params + ']', fontsize=8)
        #plt.savefig('heatmap1.pdf')
        plt.savefig(params + '.png')
        #plt.show()
    else:
        raise Exception('Unrecognized out_type')

if __name__ == '__main__':
    main()
