__author__ = 'markus'

import numpy as np
from coordinateAscentLasso import CoordinateAscentLasso
import sys
import matplotlib.pyplot as plt
from sklearn import decomposition


class LassoDictionaryLearning:
    """Lasso Dictionary learning using Coordinate Ascent"""

    def __init__(self):
        self. c = 0

    def updateDict(self, D, A, B):

        for j in range(D.shape[1]):
            #print('Update dictionary column ' + str(j + 1))
            if A[j, j] == 0:
                self.c += 1
                uj = D[:, j]
            else:
                uj = 1 / A[j, j] * (B[:, j] - np.dot(D, A[:, j])) + D[:, j]
            #print(uj)
            uj = uj[np.newaxis].T
            D[:, j] = (1 / np.maximum(np.absolute(uj), 1) * uj).flatten()

        return D

    def updateDict2(self, X, Y, W):
        xl = X.shape[1]
        for k in range(xl):
            #selecting weights column-vector wise
            print('w ' + str(k + 1) + 'th column-vector')
            w = W[:, k][np.newaxis].T
            #print(w)
            #exit()
            XNok = np.append(X[:, 0:k], X[:, k + 1:xl], axis=1)
            wNok = np.append(W[0:k, :], W[k + 1:W.shape[0], :], axis=0)
            ymk = Y - np.dot(XNok, wNok)
            print(ymk)
            print(np.shape(ymk))
            print('-------')
            print(w)
            print(np.shape(w))
            exit()
            print(np.dot(ymk, w))
            #return
            X[:, k] = np.dot(ymk, w) / sum(w)
        return X

    def updateDict5(self, X, Y, W):
        xw = X.shape[1]
        xl = X.shape[0]
        for k in range(xw):
            print(W)
            print(W[k, :])
            print(sum(W[k, :]))
            print(k)
            for m in range(xl):
                #print(np.dot(Y[m, :], W[k, :]) / sum(W[k, :]))
                #exit()
                #print(Y[m, :])
                #print(Y[m, :][np.newaxis])
                #print(W[k, :][np.newaxis].T)
                #print('----------')
                #print(W[:, k][np.newaxis].T)
                #print(Y[m, :][np.newaxis])
                #print(np.dot(Y[m, :][np.newaxis], W[k, :][np.newaxis].T))
                #print(np.dot(Y[m, :][np.newaxis], W[k, :][np.newaxis].T))
                #print( sum(W[k, :]))
                #XNok = np.append(X[:, 0:k], X[:, k + 1:xl], axis=1)
                #wNok = np.append(W[0:k, :], W[k + 1:W.shape[0], :], axis=0)
                #ymk = Y[i, ] - np.dot(XNok, wNok
                nok = 0
                ymk = 0
                for i in range(Y.shape[1]):
                    for l in range(xw):
                        if l != k:
                            nok += X[m, l] * W[l, i]

                    ymk += (Y[i, m] - nok) * W[k, i]

                #XNok = np.append(X[:, 0:k], X[:, k + 1:xl], axis=1)
                #wNok = np.append(W[0:k, :], W[k + 1:W.shape[0], :], axis=0)
                #ymk = Y[i, ] - np.dot(XNok, wNok)
                #print('ymk = ')
                #print(ymk)
                #exit()
                X[m, k] = np.dot(Y[m, :], W[k, :]) / sum(W[k, :])
        return X

    def updateDict3(self, X, y, w):
        xl = X.shape[0]
        xw = X.shape[1]
        print('Size of X = ' + str(xl) + ' * ' + str(xw))
        for m in range(xl):
            for k in range(xw):
                print(sum(w[k, :]))
                XNok = np.append(X[m, 0:k], X[m, k + 1:xw], axis=1)
                wNok = np.append(w[0:k, :], w[k + 1:w.shape[0], :])[:, np.newaxis]
                print('m before shape(y) = ' + str(m))
                print(np.shape(y[:, m]))
                print(XNok[np.newaxis])
                print(wNok)
                ymk = y[:, m] - np.dot(XNok, wNok)
                ymk = ymk[np.newaxis]
                print('ymk = ')
                print(ymk)
                print('w = ')
                print(w)
                print('m = ')
                print(m)
                X[m, k] = sum(ymk[:, m] * w) / sum(w)
        print('X = ')
        print(X)
        exit()
        return X

    #def updateDict4(self, X, y, w):
    #    s = X.shape[0]
    #    g = X.shape[1]
    #    for m in range(s):
    #        a, b = 0
    #        for k in range(g):




    def dictLearning(self):
        #p > n?
        n = 5
        p = 10
        r = 5

        #Dictionary - Line generates a random normal distributed Matrix with dimensions of R^n*p
        #will be x in lasso
        X = np.random.random((n, p))
        self.X = X
        #same as D = np.random.randn(s, r)

        #sparse code R^p*1 with # of zeros < p (or n?), rest are ones
        w = np.zeros((p, r))
        for i in range(r):
            w[:np.random.randint(1, n), i] = 1
            np.random.shuffle(w[:, i])
        self.w = w
        print(w)
        #print(np.dot(w, w.T))

        #Data matrix - R^n*1
        #will be y in Lasso
        self.y = np.dot(X, w) + 0.1 * np.random.randn(n, r)
        y = self.y
        print(y)

        print('X = ')
        print(np.shape(X))
        print('m*k')
        print('w = ')
        print(np.shape(w))
        print('k*n')
        print('y = ')
        print(np.shape(y))
        print('m*n')

        #assume default tolerance and number of iterations
        TOL = 1e-6
        MAXIT = 100

        A = 0
        B = 0

        #Coordinate Ascent Lasso
        cal = CoordinateAscentLasso()

        for i in range(MAXIT):
            #each column of weights 'corresponds' to to columns of y.
            for k in range(r):
                #updating the weights column-wise
                w[:, k] = cal.coordinateAscentLasso(y[:, k][np.newaxis].T, X, 0.1)[1].flatten()
            #print('w updated using Lasso: ')
            #print(w)

            #A = A + np.dot(w, w.T)
            #B = B + np.dot(y, w.T)
            #print('A')
            #print(A)
            #exit()

            #X = self.updateDict(X, A, B)
            X = self.updateDict5(X, y, w)
            #X = self.updateDict3(X, y, w)
            #print()
            #print(X)
            #print(np.shape(X))
            #return 1
            #select complete k-th column, select complete k-th row
            #X[:, k], w[k, :]
            #print(y - np.dot(X, w))
            #print(np.shape(y - np.dot(X, w)))
            #print(np.shape(w))
            #print(np.dot(y - np.dot(X, w), w.T))
            #X = np.divide(np.dot(y - np.dot(X, w), w.T).T, w).T
            #print(np.shape(X))

        return w, X

ldl = LassoDictionaryLearning()
print(ldl.dictLearning())
print(ldl.c)

#ldl.y is the data matrix, so R^p*1 in this case. And ldl.y.shape[1] is therefore 1
#w, X, e = decomposition.dict_learning(ldl.y, ldl.y.shape[1], 0.1, method='cd')
#print(w, X)

print('Dictionary Learning')
print(ldl.y)
#dl = decomposition.DictionaryLearning(fit_algorithm='cd')
#print(dl.fit(ldl.y))
#print(np.shape(dl.components_))
#print(dl.components_)