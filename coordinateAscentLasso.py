__author__ = 'markus'

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

class CoordinateAscentLasso:
    """Coordinate Ascent for Lasso"""

    def logLikelihood(self, y, X, beta, lam):
        #print(lam * sum(beta))
        #print(np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))))
        #At lam * sum(beta)...including or excluding beta0?
        #n = X.shape[0]
        #right = lam * sum(beta)
        #print(right)
        #left = (1 / (2 * n)) * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta)))
        #print(left)
        #print((1 / 2) * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))))
        #leftminright = (left - right)
        #print('leftminright = ')
        #print(leftminright)
        #logl = leftminright[0, 0] * -1
        #print(logl)
        #print(0.5 * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))))
        #print('----------')
        logl = -1 / 2 * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))) - lam * sum(np.absolute(beta))
        #print('--------')
        #print(logl)
        return logl[0][0]

    def logLikelihoodBeta0(self, y, X, beta0, beta, lam):
        logl = -1 / 2 * np.dot((y - beta0 - np.dot(X, beta)).T, (y - beta0 - np.dot(X, beta))) - lam * sum(np.absolute(beta))
        return logl[0][0]

    def shrinkage(self, x, lam):
        #print(np.sign(x))
        #print(np.maximum(np.absolute(x) - lam, 0))
        #print(np.size(np.sign(x)))
        #print(np.size(np.maximum(np.absolute(x) - lam, 0)))
        print('b = ')
        print(x)
        print('sigb = ')
        print(np.sign(x))
        print('ma = ')
        print(np.maximum(np.absolute(x) - lam, 0))
        s = np.sign(x) * np.maximum(np.absolute(x) - lam, 0)
        print('Shrinkage:')
        print(s[0])
        #print()
        return s[0]

    def coordinateAscentLasso(self, y, X, lam, init, drawGraph=False):
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
            beta = np.ones((k, 1))
            #beta = np.append([[beta0]], beta, 0)
            #print('beta =')
        #print(beta)
        #print('beta0 = ')
        #print(beta0)

        #assume default tolerance and number of iterations
        TOL = 1e-5
        MAXIT = 100

        #tracking likelihood
        logls = np.zeros((MAXIT, 1))
        prevlogl = -sys.float_info.max

        #print(y)
        #print()
        #print(X)
        #print()
        #print(beta)
        #print()
        #print(lam)
        #logl = self.logLikelihood(y, X, beta, lam)
        logl = self.logLikelihoodBeta0(y, X, beta0, beta, lam)
        print('logl before loop')
        print(logl)
        i = 0
        plt.figure(1)

        while logl - prevlogl > TOL and i < MAXIT:
            prevlogl = logl

            #updates
            #sigmasq = 1 / n * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta)))[0][0]
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
                print('test = ')
                #print((y - beta0 - np.dot(X, beta)))
                print(np.sum(X[:, j] ** 2))
                x = np.dot((y - beta0 - np.dot(X, beta)).T, X[:, j])
                beta[j] = 1 / np.sum(X[:, j] ** 2) * self.shrinkage(x, lam)
                print('betaj = ')
                print(beta[j])

            #likelihood for new state
            print(beta)
            print('Calculating Log Likelihood')
            #logl = self.logLikelihood(y, X, beta, lam)
            logl = self.logLikelihoodBeta0(y, X, beta0, beta, lam)
            print(logl)

            print('Assert stuff:')
            print(prevlogl)
            print(logl - prevlogl)
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
        return beta0, beta