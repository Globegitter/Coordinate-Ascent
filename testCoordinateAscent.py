__author__ = 'markus'

from coordinateAscent import coordinateAscent
from synthData import SynthData

def main():
    sd = SynthData()
    ca = coordinateAscent()

    D, y, w = sd.generateData(noise=False)
    print(ca.coordinateAscent(y, D, [], True))

    y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
