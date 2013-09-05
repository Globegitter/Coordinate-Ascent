import numpy as np

class SynthData:
    """
    Generating/Synthesyzing data for Coordinate Ascent/Lasso
    Uppercase letter variables are matrices. Lowercase either vectors or scalars.
    Dictionary D (often also X), will be of size n*p
    y will be of size n*1
    w (often also beta or theta) will be of size p*1
    w is sparse, with amount of 0s < p
    """

    def __init__(self, size=100, features=10):
        self.D = np.array([])
        self.n = size
        self.p = features
        self.w = np.array([])

    def generateDictionary(self):
        self.D = np.random.random((self.n, self.p - 1))
        #adding all ones for beta0 ??do I need that??
        self.D = np.append(np.ones((self.n, 1)), self.D, 1)
        return self.D

    def generateWeight(self):
        #Create a zero column-vector
        self.w = np.zeros((self.p, 1))
        #Get set the first x elements 1; x is a random number between 1 and 8
        self.w[:np.random.randint(1, self.p - 1)] = 1
        #shuffle the vector of zeros and ones
        np.random.shuffle(self.w)
        #might remove that
        self.w[0][0] = 1
        return self.w


    def generateY(self, noise=True, noiseLevel=0.1):
        if not noise:
            noiseLevel = 0

        #Generate y with (or without) some noise
        self.y = np.dot(self.D, self.w) + noiseLevel * np.random.random((self.D.shape[0], 1))
        return self.y

    def generateData(self, D=None, w=None, y=None, noise=True, noiseLevel=0.1):
        if D is None:
            D = self.generateDictionary()
        else:
            self.D = D

        if w is None:
            w = self.generateWeight()
        else:
            self.w = w

        if y is None:
            y = self.generateY(noise, noiseLevel)
        else:
            self.y = y

        return D, y, w