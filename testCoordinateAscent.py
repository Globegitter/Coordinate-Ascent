__author__ = 'markus'

import numpy as np
from coordinateAscent import coordinateAscent
import timeit

def main():
    #ground truth values
    beta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    #size of test set or number of data sets/measurements, etc.
    N = 100

    #number of features (getting the size of beta)
    p = beta.shape[0]

    #generating a random array in R^N*p
    X = np.random.random((N, p))

    #adding a column of ones for beta0
    X = np.append(np.ones((N, 1)), X, 1)

    #beta[np.newaxis] is transforming beta into a two dimensional array and .T to a column vector
    y = np.dot(X, beta[np.newaxis].T)

    #debugging
    #print(beta)
    #print(p)
    #print('x = ')
    #print(X)
    #print('y = ')
    #print(y)
    ca = coordinateAscent()
    print('Solution')
    print(ca.coordinateAscent(y, X, []))
    print('----------')

    sigma = np.sqrt(0.1)
    noise = sigma * np.random.random((X.shape[0], 1))
    print('lx = ')
    print(X.shape[0])
    y = y + noise
    print(ca.coordinateAscent(y, X, []))


if __name__ == '__main__':
    main()
