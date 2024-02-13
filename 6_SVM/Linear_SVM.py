import pandas as pd
import numpy as np

# Linear Kernel
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

# For Loss = Hinge, Compute the linear kernel matrix:
K = np.zeros([n, n], dtype=np.float32)
for i in range(0, n):
    for j in range(0, n):
        K[i, j] = np.dot(D[i, :], D[j, :])

# Set step size:
eta = np.zeros(n, dtype=np.float32)
for i in range(0, n):
    eta[i] = 1/K[i][i]
#print(eta.shape)

'''
# Set eps, C, alpha0 and t0:
eps = 0.001
C = 1
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
alpha_linear = np.copy(alpha)
'''
alpha = pd.read_csv("alpha.csv", sep='\n', header=None)
alpha= np.array(alpha)

# Set the weight factor:
w = np.zeros(d+1, dtype=np.float32)
for i in range(0, n):
    if alpha[i,0] > 0:
        w = w + alpha[i,0] * train_m[i, d] * D[i,:]
#print(w)

# Apply for the testing data:
b = w[d]
y_test = test_m[:,d].reshape(-1)
y_test = y_test.astype(int)

# Map Xi to R(d+1):
X0 = np.ones([test_m.shape[0], 1], dtype=np.float32)
D_test = np.hstack((test_m[:, 0:test_m.shape[1]-1], X0))
n_test = D_test.shape[0]
d = D_test.shape[1]-1

# For Loss = Hinge, Compute the linear kernel matrix:
K_test = np.zeros([n_test, n_test], dtype=np.float32)
for i in range(0, n_test):
    for j in range(0, n_test):
        K_test[i, j] = np.dot(D_test[i, :], D_test[j, :])


# Test accuracy
count = 0
for z in range(n_test):
    sum = 0
    for i in range(n):
        if alpha[i,0] > 0:
            sum = sum + alpha[i,0] * train_m[i, d] * np.dot(np.mat(D[i,:]), np.mat(D_test[z,:]).transpose())
    #sum += b
    if sum > 0:
        yhat = 1
    if sum < 0:
        yhat = -1
    if yhat == y_test[z]:
        count += 1

count = 0
for z in range(n_test):
    y = np.dot(w.transpose(), D_test[z,:])
    if y > 0:
        yhat = 1
    if y < 0:
        yhat = -1
    if yhat == y_test[z]:
        count += 1

accuracy = count / n_test
print(count)
print(accuracy)
