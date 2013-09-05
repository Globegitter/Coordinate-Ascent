__author__ = 'markus'

from coordinateAscent import coordinateAscent
from synthData import SynthData
from sklearn import linear_model


def main():
    sd = SynthData()
    ca = coordinateAscent()

    D, y, w = sd.generateData(noise=False)
    print(ca.coordinateAscent(y, D, [], True))

    y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    print(ca.coordinateAscent(y, D, [], True))

    clf = linear_model.Lars()
    print(clf.fit(D, y))
    print(clf.coef_)


if __name__ == '__main__':
    main()
