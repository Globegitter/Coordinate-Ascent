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
    print(cal.coordinateAscentLasso(y, X, 0.1, [], False, beta0Seperate))

    print('Fitting the model with plain \'ol Coordinate Ascent')
    print(ca.coordinateAscent(y, X, [], False))

    print('Fitting the model with LARS')
    clf = linear_model.Lars()
    print(clf.fit(X, y))
    print(clf.coef_)
    #return 1

    #y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    #print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
