__author__ = 'markus'

import numpy as np
from coordinateAscent import coordinateAscent
from synthData import SynthData

def main():
    sd = SynthData()
    D, y, w = sd.generateData(noise=False)
    print(y)
    print('-------')
    ca = coordinateAscent()
    print(ca.coordinateAscent(y, D, [], True))

    y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    print(y)
    print('-------')
    print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
