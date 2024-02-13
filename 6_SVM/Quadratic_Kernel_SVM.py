import pandas as pd
import numpy as np

# Quadratic Kernel
# -------------------------------------------------------------
# Load in the data files:
train = pd.read_csv("train.txt", sep=',', header=None)
test = pd.read_csv("test.txt", sep=',', header=None)

# Convert the data into a matrix:
train_m = np.array(train)
test_m = np.array(test)
#print(test_m.shape)

# Map Xi to R(d+1):
X0 = np.ones([train_m.shape[0], 1], dtype=np.float32)
D = np.hstack((train_m[:, 0:train_m.shape[1]-1], X0))
n = D.shape[0]
d = D.shape[1]-1
#print(D)

# For Loss = Hinge, Compute the homogeneous quadratic kernel matrix:
# First compute the kernel matrix:
K = np.zeros([n, n], dtype=np.float32)
for i in range(0, n):
    for j in range(0, n):
        temp = np.dot(D[i, :], D[j, :])
        K[i][j] = (temp) ** 2
#print(K)

# Set step size:
eta = np.zeros(n, dtype=np.float32)
for i in range(0, n):
    eta[i] = 1/K[i][i]
#print(eta.shape)

# Set eps, C, alpha0 and t0:
eps = 0.001
C = 4
t = 0
alpha0 = np.zeros(n, dtype=np.float32)
alpha = np.copy(alpha0)

while True:
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum = sum + alpha[j] * train_m[j, d] * K[j, i]
        alpha[i] = alpha[i] + eta[i] * (1 - train_m[i, d] * sum)

        # Judge the value of alpha:
        if alpha[i] < 0:
            alpha[i] = 0
        if alpha[i] > C:
            alpha[i] = C

    # Repeat until:
    print(np.linalg.norm(alpha - alpha0))
    if np.linalg.norm(alpha - alpha0) <= eps:
        break
    else:
        # Update alpha and t:
        alpha0 = np.copy(alpha)
        t = t + 1

print(alpha)
alpha_quadratic = np.copy(alpha)

# Set the weight factor:
w = np.zeros(d+1, dtype=np.float32)
for i in range(0, n):
    if alpha[i] > 0:
        w = w + alpha[i] * train_m[i, d] * D[i,:]
print(w)

# Apply for the testing data:
b = w[d]
y_test = test_m[:,d].reshape(-1)
y_test = y_test.astype(int)

# Map Xi to R(d+1):
X0 = np.ones([test_m.shape[0], 1], dtype=np.float32)
D_test = np.hstack((test_m[:, 0:test_m.shape[1]-1], X0))
n_test = D_test.shape[0]
d = D_test.shape[1]-1

# For Loss = Hinge, compute the Quadratic kernel matrix:
K_test = np.zeros([n_test, n_test], dtype=np.float32)
for i in range(0, n_test):
    for j in range(0, n_test):
        temp = np.dot(D_test[i, :], D_test[j, :])
        K_test[i][j] = (temp) ** 2
#print(K_test)

# Test the accuracy:
count = 0
for z in range(n_test):
    sum = 0
    for i in range(n):
        if alpha[i] > 0:
            sum = sum + alpha[i] * train_m[i, d] * K_test[z,i]
    sum += b
    if sum > 0:
        yhat = 1
    if sum < 0:
        yhat = -1
    if yhat == y_test[z]:
        count += 1

# Record the index for the support vector:
index = []
for i in range(0, n):
    if alpha_quadratic[i] > 0:
        index.append(i)
    else:
        continue

accuracy = count / n_test
print("For Quadratic Kernel, the best c is:")
print("c = 1")
print("The accuracy rate is:")
print(accuracy, "\n")
print("The weight vector is:")
print(w, "\n")
print("The index i for the support vectors are:")
print(index)
