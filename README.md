# measure
Various distance and similarity measures in python. 
Updated version will include implementation of metrics in 'Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions' by Sung-Hyuk Cha    

```python
import numpy as np

class Distance(object):

    def braycurtis(self, a, b):
        return np.sum(np.fabs(a - b)) / np.sum(np.fabs(a + b))

    def canberra(self, a, b):
        return np.sum(np.fabs(a - b) / (np.fabs(a) + np.fabs(b)))

    def chebyshev(self, a, b):
        return np.amax(a - b)

    def cityblock(self, a, b):
        return self.manhattan(a, b)

    def correlation(self, a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        return 1.0 - np.mean(a * b) / np.sqrt(np.mean(np.square(a)) * np.mean(np.square(b)))

    def cosine(self, a, b):
        return 1 - np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

    def dice(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))

    def euclidean(self, a, b):
        return np.sqrt(np.sum(np.dot((a - b), (a - b))))

    def hamming(self, a, b, w = None):
        if w is None:
            w = np.ones(a.shape[0])
        return np.average(a != b, weights = w)

    def jaccard(self, u, v):
        return np.double(np.bitwise_and((u != v), np.bitwise_or(u != 0, v != 0)).sum()) / np.double(np.bitwise_or(u != 0, v != 0).sum())

    def kulsinski(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (ntf + nft - ntt + len(a)) / (ntf + nft + len(a))

    def mahalanobis(self, a, b, vi):
        return np.sqrt(np.dot(np.dot((a - b), vi),(a - b).T))

    def manhattan(self, a, b):
        return np.sum(np.fabs(a - b))

    def matching(self, a, b):
        return self.hamming(a, b)

    def minkowski(self, a, b, p):
        return np.power(np.sum(np.power(np.fabs(a - b), p)), 1 / p)

    def rogerstanimoto(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))

    def russellrao(self, a, b):
        return float(len(a) - (a * b).sum()) / len(a)

    def seuclidean(self, a, b, V):
        return np.sqrt(np.sum((a - b) ** 2 / V))

    def sokalmichener(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))

    def sokalsneath(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / np.array(ntt + 2.0 * (ntf + nft))

    def sqeuclidean(self, a, b):
        return np.sum(np.dot((a - b), (a - b)))

    def wminkowski(self, a, b, p, w):
        return np.power(np.sum(np.power(np.fabs(w * (a - b)), p)), 1 / p)

    def yule(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))

def main():
    from scipy.spatial import distance
    a = np.array([1, 2, 43])
    b = np.array([3, 2, 1])

    d = Distance()
    print('-----------------------------------------------------------------')

    print('My       braycurtis: {}'.format(d.braycurtis(a, b)))
    print('SciPy    braycurtis: {}'.format(distance.braycurtis(a, b)))
    print('-----------------------------------------------------------------')
    
    print('My       canberra: {}'.format(d.canberra(a, b)))
    print('SciPy    canberra: {}'.format(distance.canberra(a, b)))
    print('-----------------------------------------------------------------')

    print('My       chebyshev: {}'.format(d.chebyshev(a, b)))
    print('SciPy    chebyshev: {}'.format(distance.chebyshev(a, b)))
    print('-----------------------------------------------------------------')

    print('My       cityblock: {}'.format(d.cityblock(a, b)))
    print('SciPy    cityblock: {}'.format(distance.cityblock(a, b)))
    print('-----------------------------------------------------------------')

    print('My       correlation: {}'.format(d.correlation(a, b)))
    print('SciPy    correlation: {}'.format(distance.correlation(a, b)))
    print('-----------------------------------------------------------------')

    print('My       euclidean: {}'.format(d.euclidean(a, b)))
    print('SciPy    euclidean: {}'.format(distance.euclidean(a, b)))
    print('-----------------------------------------------------------------')

    print('My       hamming: {}'.format(d.hamming(a, b)))
    print('SciPy    hamming: {}'.format(distance.hamming(a, b)))
    print('-----------------------------------------------------------------')

    print('My       jaccard: {}'.format(d.jaccard(a, b)))
    print('SciPy    jaccard: {}'.format(distance.jaccard(a, b)))
    print('-----------------------------------------------------------------')

    print('My       manhattan: {}'.format(d.cityblock(a, b)))
    print('SciPy    manhattan: {}'.format(distance.cityblock(a, b)))
    print('-----------------------------------------------------------------')

    print('My       cosine: {}'.format(d.cosine(a, b)))
    print('SciPy    cosine: {}'.format(distance.cosine(a, b)))
    print('-----------------------------------------------------------------')

    print('My       dice: {}'.format(d.dice(a, b)))
    print('SciPy    dice: {}'.format(distance.dice(a, b)))
    print('-----------------------------------------------------------------')

    print('My       kulsinski: {}'.format(d.kulsinski(a, b)))
    print('SciPy    kulsinski: {}'.format(distance.kulsinski(a, b)))
    print('-----------------------------------------------------------------')

    iv = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    print('My       mahalanobis: {}'.format(d.mahalanobis(a, b, iv)))
    print('SciPy    mahalanobis: {}'.format(distance.mahalanobis(a, b, iv)))
    print('-----------------------------------------------------------------')

    print('My       seuclidean: {}'.format(d.seuclidean(a, b, np.array([0.1, 0.1, 0.1]))))
    print('SciPy    seuclidean: {}'.format(distance.seuclidean(a, b, [0.1, 0.1, 0.1])))
    print('-----------------------------------------------------------------')

    print('My       sokalmichener: {}'.format(d.sokalmichener(a, b)))
    print('SciPy    sokalmichener: {}'.format(distance.sokalmichener(a, b)))
    print('-----------------------------------------------------------------')

    print('My       sokal_sneath: {}'.format(d.sokalsneath(a, b)))
    print('SciPy    sokal_sneath: {}'.format(distance.sokalsneath(a, b)))
    print('-----------------------------------------------------------------')

    print('My       sqeuclidean: {}'.format(d.sqeuclidean(a, b)))
    print('SciPy    sqeuclidean: {}'.format(distance.sqeuclidean(a, b)))
    print('-----------------------------------------------------------------')
    
    print('My       minkowski: {}'.format(d.minkowski(a, b, 2)))
    print('SciPy    minkowski: {}'.format(distance.minkowski(a, b, 2)))
    print('-----------------------------------------------------------------')

    print('My       rogerstanimoto: {}'.format(d.rogerstanimoto(a, b)))
    print('SciPy    rogerstanimoto: {}'.format(distance.rogerstanimoto(a, b)))
    print('-----------------------------------------------------------------')

    print('My       russellrao: {}'.format(d.russellrao(a, b)))
    print('SciPy    russellrao: {}'.format(distance.russellrao(a, b)))
    print('-----------------------------------------------------------------')

    print('My       wminkowski: {}'.format(d.wminkowski(a, b, 2, np.ones(3))))
    print('SciPy    wminkowski: {}'.format(distance.wminkowski(a, b, 2, np.ones(3))))
    print('-----------------------------------------------------------------')

    print('My       yule: {}'.format(d.yule(a, b)))
    print('SciPy    yule: {}'.format(distance.yule(a, b)))
    print('-----------------------------------------------------------------')

if __name__ == '__main__':
    main()

```

```
-----------------------------------------------------------------
My       braycurtis: 0.8461538461538461
SciPy    braycurtis: 0.8461538461538461
-----------------------------------------------------------------
My       canberra: 1.4545454545454546
SciPy    canberra: 1.4545454545454546
-----------------------------------------------------------------
My       chebyshev: 42
SciPy    chebyshev: 42
-----------------------------------------------------------------
My       cityblock: 44.0
SciPy    cityblock: 44
-----------------------------------------------------------------
My       correlation: 1.8762686682028846
SciPy    correlation: 1.8762686682028846
-----------------------------------------------------------------
My       euclidean: 42.04759208325728
SciPy    euclidean: 42.04759208325728
-----------------------------------------------------------------
My       hamming: 0.6666666666666666
SciPy    hamming: 0.6666666666666666
-----------------------------------------------------------------
My       jaccard: 0.6666666666666666
SciPy    jaccard: 0.6666666666666666
-----------------------------------------------------------------
My       manhattan: 44.0
SciPy    manhattan: 44
-----------------------------------------------------------------
My       cosine: 0.6896504488650589
SciPy    cosine: 0.6896504488650589
-----------------------------------------------------------------
My       dice: -0.9230769230769231
SciPy    dice: -0.9230769230769231
-----------------------------------------------------------------
My       kulsinski: 2.111111111111111
SciPy    kulsinski: 2.111111111111111
-----------------------------------------------------------------
My       mahalanobis: 41.036569057366385
SciPy    mahalanobis: 41.036569057366385
-----------------------------------------------------------------
My       seuclidean: 132.9661611087573
SciPy    seuclidean: 132.9661611087573
-----------------------------------------------------------------
My       sokalmichener: 2.1333333333333333
SciPy    sokalmichener: 2.1333333333333333
-----------------------------------------------------------------
My       sokal_sneath: 2.0869565217391304
SciPy    sokal_sneath: 2.0869565217391304
-----------------------------------------------------------------
My       sqeuclidean: 1768
SciPy    sqeuclidean: 1768.0
-----------------------------------------------------------------
My       minkowski: 42.04759208325728
SciPy    minkowski: 42.04759208325728
-----------------------------------------------------------------
My       rogerstanimoto: 2.1333333333333333
SciPy    rogerstanimoto: 2.1333333333333333
-----------------------------------------------------------------
My       russellrao: -15.666666666666666
SciPy    russellrao: -15.666666666666666
-----------------------------------------------------------------
My       wminkowski: 42.04759208325728
SciPy    wminkowski: 42.04759208325728
-----------------------------------------------------------------
My       yule: 1.5575221238938053
SciPy    yule: 1.5575221238938053
-----------------------------------------------------------------
```
