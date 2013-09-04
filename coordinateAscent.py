__author__ = 'markus'

import numpy as np
import sys
import math
import matplotlib.pyplot as plt
#from numpy import  *


class coordinateAscent:

    def logLikelihood(self, y, X, beta, sigmasq):
        n = y.shape[0]
        logl = -n / 2 * math.log(2 * math.pi * sigmasq) \
            - (1 / (2 * sigmasq)) * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta)))
        return logl[0][0]

    def coordinateAscent(self, y, X, init):
        assert X.shape[0] == y.shape[0] and y.shape[0] > 0, \
            'Matrices must have more than 0 rows and they have to be of the same dimension'
        n = y.shape[0]
        k = X.shape[1]

        if init:
            sigmasq = init[0]
            beta0 = init[1]
            beta = init[2]
        else:
            sigmasq = y.var(axis=0, ddof=1)[0]
            beta0 = y.mean(axis=0)[0]
            beta = 1e-6 * np.ones((k - 1, 1))
            beta = np.append([[beta0]], beta, 0)
        print('beta =')
        print(beta)
        print('beta0 = ')
        print(beta0)

        #assume default tolerance and number of iterations
        TOL = 1e-5
        MAXIT = 100

        #tracking likelihood
        logls = np.zeros((MAXIT, 1))
        prevlogl = -sys.float_info.max

        logl = self.logLikelihood(y, X, beta, sigmasq)
        i = 0
        plt.figure(1)

        #print(logl)

        while logl - prevlogl > TOL and i < MAXIT:
            prevlogl = logl

            #updates
            sigmasq = 1 / n * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta)))[0][0]
            #beta0 = 1 / n * sum(y - np.dot(X, beta))
            for j in range(k):
                #print('X[:,j] = ')
                #print(X[:, j])
                #select the whole column j from X (as a row vector)
                #print(sum(X[:, j]))
                #print(sum(X[:, j] ** 2))

                beta[j] = 0
                beta[j] = np.dot((y - np.dot(X, beta)).T, X[:, j]) / (sum(X[:, j] ** 2))

            #likelihood for new state
            logl = self.logLikelihood(y, X, beta, sigmasq)

            assert logl - prevlogl > 0, 'Difference must be bigger than 0'

            logls[i] = logl
            #if i % 10 == 0:

                #plt.draw()
            i += 1

        #print('X = ')
        #print(X)
        #print('logls')
        #print(logls)
        #plt.plot(logls[logls != 0])
        #plt.xlabel('iteration')
        #plt.ylabel('log-likelihood')
        #plt.show()

        sigma = np.sqrt(sigmasq)
        return [sigma, beta]
