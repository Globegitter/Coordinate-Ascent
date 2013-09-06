__author__ = 'markus'

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

class CoordinateAscentLasso:
    """Coordinate Ascent for Lasso"""

    def logLikelihood(self, y, X, beta, lam=1, beta0=None):
        if beta0 is None:
            logl = -1 / 2 * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))) - lam * sum(np.absolute(beta))
        else:
            logl = -1 / 2 * np.dot((y - beta0 - np.dot(X, beta)).T, (y - beta0 - np.dot(X, beta))) - lam * sum(np.absolute(beta))
        return logl[0][0]

    def shrinkage(self, x, lam=1):
        s = np.sign(x) * np.maximum(np.absolute(x) - lam, 0)
        return s[0]

    def coordinateAscentLasso(self, y, X, lam, init, drawGraph=False, beta0Seperate=True):
        assert X.shape[0] == y.shape[0] and y.shape[0] > 0, \
            'Matrices must have more than 0 rows and they have to be of the same dimension'
        #np.set_printoptions(suppress=True)
        n = y.shape[0]
        k = X.shape[1]

        if init:
            #beta0 = init[1]
            beta = init[0]
        else:
            #sigmasq = y.var(axis=0, ddof=1)[0]
            beta0 = y.mean(axis=0)[0]
            if beta0Seperate:
                beta = np.ones((k, 1))
            else:
                beta = np.ones((k - 1, 1))
                beta = np.append([[beta0]], beta, 0)
                beta0 = None
        #assume default tolerance and number of iterations
        TOL = 1e-5
        MAXIT = 100

        #tracking likelihood
        logls = np.zeros((MAXIT, 1))
        prevlogl = -sys.float_info.max

        logl = self.logLikelihood(y, X, beta, lam, beta0)
        print('logl before loop')
        print(logl)
        i = 0
        plt.figure(1)

        while logl - prevlogl > TOL and i < MAXIT:
            prevlogl = logl

            #updates
            if beta0Seperate:
                beta0 = (1 / n) * np.sum((y - np.dot(X, beta)))

            for j in range(0, k):

                beta[j] = 0
                #beta[j] = np.dot((y - np.dot(X, beta)).T, X[:, j]) / (sum(X[:, j] ** 2))

                #XNoj = np.append(X[i, 0:j], X[i, j + 1:k])
                #betaNoj = np.append(beta[0:j], beta[j + 1:k])
                #yminj = y - np.dot(XNoj, betaNoj)
                #xj = X[:, j][np.newaxis]
                #x = np.dot(yminj, xj)
                #x = yminj
                #print(sum(X[:, j] ** 2))
                #print(sum(sum(X ** 2)))
                #print('!!!!!!!!!!!!!')
                #print('test = ')
                #print((y - beta0 - np.dot(X, beta)))
                #print(np.sum(X[:, j] ** 2))
                if beta0Seperate:
                    x = np.dot((y - beta0 - np.dot(X, beta)).T, X[:, j])
                else:
                    x = np.dot((y - np.dot(X, beta)).T, X[:, j])
                beta[j] = 1 / np.sum(X[:, j] ** 2) * self.shrinkage(x, lam)

            #likelihood for new state
            logl = self.logLikelihood(y, X, beta, lam, beta0)
            print(beta0)
            print('ll = ')
            print(logl)

            #print('Assert stuff:')
            #print(prevlogl)
            #print(logl - prevlogl)
            assert logl - prevlogl > 0, 'Difference must be bigger than 0'

            logls[i] = logl
            i += 1

        if drawGraph:
            #just plot all the log likelihoods not 0
            plt.plot(logls[logls != 0])
            plt.xlabel('iteration')
            plt.ylabel('log-likelihood')
            plt.show()

        #sigma = np.sqrt(sigmasq)
        print('Solutions: ')
        if beta0Seperate:
            return beta0, beta
        else:
            return beta