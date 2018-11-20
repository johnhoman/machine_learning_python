from random import randint
import operator
import random
import sys


def to_list(filename):
    """return a list of values from the space delimited file
    `filename`

    Parameters
    ----------
    filename: str
        name of file to be read

    Returns
    -------
    [[float]]:
        2-d list of floats
    """

    with open(filename, 'rt') as textfile:
        lines = textfile.readlines()
        matrix_array = [tuple(float(i) for i in line.split()) for line in lines]
        return matrix_array



def kmeans(data, k):
    """given the two dimensional data `data` cluster
    the points in `data` into `k` clusters

    Parameters
    ----------
    data: array-like, shape (m, 2)
        2 dimensional data
    k: int
        number of clusters
    """

    # sample index values
    ix = range(len(data))
    indexes = [i for i in random.sample(ix, k)]

    clusters = [[data[i]] for i in indexes]
    data = [element for n, element in enumerate(data) if n not in indexes]

    def calculate_centers(clust):

        def mean(it):
            n = len(it)
            return sum(it) / n

        centers = []
        for cluster in clust:
            if len(cluster) == 1:
                centers.append(cluster[0])
            else:
                centers.append(tuple(mean(i) for i in zip(*cluster)))
        return centers


    for point in data:
        centers = calculate_centers(clusters)
        minimum = 1000000 # something  absurdly high
        ix = 0
        for x, center in enumerate(centers):
            diff = sum(abs(a - b) for a, b in zip(point, center))
            if diff < minimum:
                minimum = diff
                ix = x
        clusters[ix].append(point)

    for k, cluster in enumerate(clusters):
        for point in cluster:
            print(k, point)


print('k = 2')
kmeans(to_list('data'), 2)
print('k = 3')
kmeans(to_list('data'), 3)
print('k = 4')
kmeans(to_list('data'), 4)
print('k = 5')
kmeans(to_list('data'), 5)
