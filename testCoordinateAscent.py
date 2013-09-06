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

    X, y, b = sd.generateData(noise=False)
    #if beta0Seperate:
    #    beta = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T
    #else:
    #    beta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T


    #if beta0Seperate:
    #    y = 1 + np.dot(X, beta)
    #else:
    #    X = np.append(np.ones((X.shape[0], 1)), X, 1)
    #    y = np.dot(X, beta)

    print('Starting Lasso:')
    print(cal.coordinateAscentLasso(y, X, 1, [], True, beta0Seperate))
    return 1

    y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    print(ca.coordinateAscent(y, D, [], True))

    clf = linear_model.Lars()
    print(clf.fit(D, y))
    print(clf.coef_)


if __name__ == '__main__':
    main()
