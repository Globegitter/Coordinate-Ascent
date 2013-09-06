__author__ = 'markus'

from coordinateAscent import coordinateAscent
from coordinateAscentLasso import CoordinateAscentLasso
from standardize import Standardize
from synthData import SynthData
from sklearn import linear_model
import numpy as np


def main():
    sd = SynthData()
    ca = coordinateAscent()
    cal = CoordinateAscentLasso()
    st = Standardize()
    beta0Seperate = True
    lam = 0.1

    X, y, b = sd.generateData(noise=False,  w=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])[np.newaxis].T)
    #if beta0Seperate:
    #    beta = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T
    #else:
    #    beta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T

    #if beta0Seperate:
    #    y = 1 + np.dot(X, beta)
    #else:
    #    X = np.append(np.ones((X.shape[0], 1)), X, 1)
    #    y = np.dot(X, beta)

    print('Fitting the model with Lasso:')
    print('Lambda = ' + str(lam))
    print('beta0, array of betas:')
    print(cal.coordinateAscentLasso(y, X, lam, [], False, beta0Seperate))

    print()
    print('Fitting the model with plain \'ol Coordinate Ascent')
    print('beta0, array of betas:')
    print(ca.coordinateAscent(y, X, [], False))
    print()

    print('Fitting the model with LARS (from scikit library)')
    clf = linear_model.Lars()
    print(clf.fit(X, y))
    print('array of betas:')
    print(clf.coef_)
    #return 1

    #y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    #print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
