
import random
import math
import numpy as np

# Read in file
D = np.loadtxt("input3.txt", delimiter=',', dtype=float)

# Set the parameter:
k = 5
eps = 0.001

Y = D[:, -1]
#D = D[:, 0:(D.shape[1]-1)]
m = D.shape[0]
n = len(D)
d = len(D[0])

def f(x, mu, cov, d):
    cov += 0.0001 * np.identity(d)
    value = (2 * math.pi) ** (d / 2) * (np.linalg.det(cov) ** (1 / 2))
    value = 1 / value
    g = -0.5 * np.transpose(x - mu)
    g = np.dot(g, np.linalg.inv(cov))
    g = np.dot(g, x - mu)
    value *= math.exp(g)
    return value

def find_range(D, a):
    value_l = -float('inf')
    value_r = float('inf')
    for i in D:
        value_l = max(i[a], value_l)
        value_r = min(i[a], value_r)
    return (value_l, value_r)

def exp_max(D, k, eps, d):
    n = len(D)
    t = 0
    mu = np.zeros(shape=(k, d))
    for i in range(k):
        for a in range(d):
            min, max = find_range(D, a)
            mu[i][a] = random.uniform(min, max)
    cov = []
    for i in range(k):
        cov.append(np.identity(d))
    cov = np.array(cov)
    P = np.zeros(k)
    for i in range(k):
        P[i] = 1.0 / k
    w = np.zeros((k, n))
    while True:
        t += 1
        mu_copy = np.copy(mu)
        for j in range(n):
            s = 0
            for a in range(k):
                s += f(D[j], mu[a], cov[a], d) * P[a]
            for i in range(k):
                w[i][j] = f(D[j], mu[i], cov[i], d) * P[i] / s
        for i in range(k):
            s1 = 0
            s2 = 0
            for j in range(n):
                s1 += w[i][j] * D[j]
                s2 += w[i][j]
            mu[i] = s1 / s2
            s1 = 0
            for j in range(n):
                s1 += w[i][j] * np.outer(D[j] - mu[i], np.transpose(D[j] - mu[i]))
            cov[i] = s1 / s2
            P[i] = s2 / n
        s = 0
        for i in range(k):
            s += np.linalg.norm(mu[i] - mu_copy[i]) ** 2
        if s <= eps:
            C = [-1] * n
            for j in range(n):
                c = -1
                max_prob = 0
                for i in range(k):
                    if w[i][j] > max_prob:
                        max_prob = w[i][j]
                        c = i
                C[j] = c
            counts = []
            for i in range(k):
                count = C.count(i)
                counts.append(count)
            break
    return (mu, cov, t, C, counts, w)

# E-M method
em = exp_max(D, k, eps, d)
K = len(set(Y))

# solve for purity
s = 0.
for i in range(k):
    maxi = 0
    Ci = set()
    for a in range(n):
        if em[3][a] == i:
            Ci.add(a)

    for j in list(set(Y)):
        Tj = set()
        for a in range(n):
            if Y[a] == j:
                Tj.add(a)
        intersect = len(Ci.intersection(Tj))
        maxi = max(maxi, intersect)
    s += maxi
purity = s / n

for i in range(len(em[0])):
    print("The mean for cluster is:\n", i, em[0][i])
print()

for i in range(len(em[1])):
    print("The covariance for cluster is:\n", i, em[1][i])
print()

print("The number of iterations is:", em[2])
print()

print("The final cluster assignment of all points is:")
print(em[3])

for i in range(len(em[4])):
    print("The size of cluster is:", i, em[4][i])
print()

print("The purity score is:", purity)