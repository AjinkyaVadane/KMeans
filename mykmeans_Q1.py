# Function for getting random data from covariancce matrix and mean values

import numpy as np
import matplotlib.pyplot as plt
import copy as copy

mean1 = np.array([1, 0])
covariance1 = np.array([[0.9, 0.4], [0.4, 0.9]])
mean2 = np.array([0, 1.5])
covariance2 = np.array([[0.9, 0.4], [0.4, 0.9]])

data1 = np.random.multivariate_normal(mean1, covariance1, size=500)
data2 = np.random.multivariate_normal(mean2, covariance2, size=500)
X = np.append(data1, data2, axis=0)

m = X.shape[0]  # examples
n = X.shape[1]  # features

# n_iter is number of iteration that our alogirthm  will be computing till it reaches to the convergence point
n_iter = 10000
K = int(input('Enter the Number of Clusters: '))  # number of clusters

# Algorithm
'''
#Step 1 :- Initialize the centroids randomly from the data points:
Centroids = np.array([]).reshape(2,0)
'''

Centroids = []
for i in range(K):
    x, y = input('Enter coordinates: ').split()
    Centroids.append([x, y])

for i in range(len(Centroids)):
    for j in range(len(Centroids[0])):
        Centroids[i][j] = float(Centroids[i][j])
print(Centroids)
Centroids = np.array(Centroids)
Centroids = Centroids.T


# Step 2.1: For each training example compute the euclidean
# distance from the centroid and assign the cluster based on
# the minimal distance
count=0
def mykmeans(X, K, Centroids):
    global count
    for i in range(n_iter):
        EuclideanDist = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
            EuclideanDist = np.c_[EuclideanDist, tempDist]
        C = np.argmin(EuclideanDist, axis=1) + 1
        c_old = np.zeros(Centroids.shape)
        c_old = copy.deepcopy(Centroids)
        # step 2.b
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(2, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]

        for k in range(K):
            Y[k + 1] = Y[k + 1].T

        for k in range(K):
            Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        count = count + 1
        if (np.linalg.norm((c_old - Centroids), axis=None)) <= 0.001:
            break
        EuclidianDist = np.array([]).reshape(m, 0)
        Output = Y

    plt.scatter(X[:, 0], X[:, 1], c='black', label='unclustered data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Plot of data points')
    plt.show()

    color = ['red', 'blue', 'green', 'cyan']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.legend()
    plt.show()
    return Output

cluster= mykmeans(X, K, Centroids)
print(count)
print(cluster)