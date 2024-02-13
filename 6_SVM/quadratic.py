import pandas as pd
import numpy as np
import random
import math
pd.set_option('display.max_columns', None)

maxiter = 50
loss = "quadratic"
kernel = "linear"
C = 1
eps = 0.0000001
spread = 8
c = 2.5
q = 2

# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')

# Shuffle the dataset first and use the first 5000 rows:
data = pd.DataFrame(data)
data = data.sample(frac=1)
data = data.iloc[0:2000, :]
data = np.array(data)
nrows = data.shape[0]

# Label the y's for the trimmed data:
y = np.empty([nrows, 1])
for i in range (0, nrows):
    if (data[i, 0] <= 50):
        y[i, 0] = 1
    elif(data[i, 0] > 50):
        y[i, 0] = -1

# Use the first 70% rows of the trimmed data as the training set and the remaining 30% as test set:
data = pd.DataFrame(data)
train = data.iloc[0:int(0.7*data.shape[0]), :]
test = data.iloc[int(0.7*data.shape[0]):, :]
y = pd.DataFrame(y)
y_train = y.iloc[0:int(0.7*y.shape[0]), :]
y_test = y.iloc[int(0.7*y.shape[0]):, :]

# Transform the dataframes into matrix:
data = np.array(data)
train = np.array(train)
test = np.array(test)
y = np.array(y)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Map the train dataset onto R^(d+1) and use as input matrix:
X = np.ones([train.shape[0], 1])
D = np.hstack((train[:, 1:train.shape[1]], X))

# Computer kernel matrix according to specific kernel and loss:
n = train.shape[0]
K = np.zeros([n, n], dtype=np.float32)
if (loss == "quadratic" and kernel == "linear"):
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                K[i, j] = np.dot(D[i, :], D[j, :].transpose())
            elif (i == j):
                K[i, j] = np.dot(D[i, :], D[j, :].transpose()) + 0.5/C
if (loss == "quadratic" and kernel == "gaussian"):
    for i in range(0, n):
        for j in range(0, n):
            distance = np.linalg.norm(D[i, :] - D[j, :])
            if (i != j):
                K[i, j] = math.exp(-(distance * distance) / (2 * spread))
            elif (i == j):
                K[i, j] = math.exp(-(distance * distance) / (2 * spread)) + 0.5/C
if (loss == "quadratic" and kernel == "polynomial"):
    for i in range(0, n):
        for j in range(0, n):
            if(i != j):
                temp = c + np.dot(D[i, :], D[j, :].transpose())
                K[i, j] = temp * temp
            elif(i == j):
                temp = c + np.dot(D[i, :], D[j, :].transpose())
                K[i, j] = temp * temp + 0.5/C
# Set step size:
eta = np.empty([n, 1])
for i in range (0, n):
    eta[i, 0] = 1/K[i, i]

# Initialize iteration counter, alpha, and convergence:
t = 0
alpha_0 = np.zeros([n, 1])

# Repeat iterations:
while (t < maxiter):
    alpha = np.copy(alpha_0)
    # Shuffle the indexes:
    r = list(range(0, n))
    random.shuffle(r)
    for k in r:
        sum = 0
        # Update the k-th component of alpha:
        for i in range (0, n):
            sum = sum + alpha[i, 0] * y_train[i, 0] * K[i, k]
        temp = 1 - y_train[k, 0] * sum
        alpha[k, 0] = alpha[k, 0] + eta[k, 0] * temp
        # Adjust alpha:
        if (alpha[k, 0] < 0):
            alpha[k, 0] = 0

    # Compute convergence threshold:
    conv = np.linalg.norm(alpha - alpha_0)
    if(conv < eps):
        break
    else:
        # Update alpha and t for next iteration:
        alpha_0 = np.copy(alpha)
        t = t + 1
#print("t is: %d." %(t))

# Calculate weight and bias for linear kernels:
if(kernel == "linear"):
    w = np.zeros([train.shape[1]])
    for k in range(0, n):
        if (alpha[k, 0] > 0):
            temp = alpha[k, 0] * y_train[k, 0] * D[k, :]
            w = w + temp
'''
    # Calculate b:
    sum_b = 0
    count_b = 0
    for i in range(0, n):
        if(0 < alpha[i, 0] < C):
            count_b = count_b + 1
            sum_b = sum_b + (y_train[i, 0] - np.array([w]) @ train[i, 1:train.shape[1]])
    b = sum_b/count_b
'''
# Calculate the number of support vectors:
num_sv = 0
for i in range(0, n):
    if(0 < alpha[i, 0]):
        num_sv = num_sv + 1

# ---------------------------------Now consider test dataset-----------------------------------------------
# Map the test dataset onto R^(d+1) and use as input matrix:
X_test = np.ones([test.shape[0], 1])
D_test = np.hstack((test[:, 1:test.shape[1]], X_test))
n_test = D_test.shape[0]
y_pred = np.empty([n_test, 1])

# Generate predications on testing data:
if (loss == "quadratic" and kernel == "linear"):
    for i in range (0, n_test):
        sum = 0
        for k in range (0, n):
            if (alpha[k, 0] > 0):
                k_temp = np.array([D[k, :]]) @ np.array([D_test[i, :]]).transpose()
                sum = sum + alpha[k, 0] * y_train[k, 0] * k_temp
            else:
                continue
        if (sum > 0):
            y_pred[i, 0] = 1
        elif (sum < 0):
            y_pred[i, 0] = -1
if (loss == "quadratic" and kernel == "gaussian"):
    for i in range (0, n_test):
        sum = 0
        for k in range (0, n):
            if (alpha[k, 0] > 0):
                distance = np.linalg.norm(D[k, :] - D_test[i, :])
                temp = math.exp(-(distance * distance) / (2 * spread))
                sum = sum + alpha[k, 0] * y_train[k, 0] * temp
        if (sum > 0):
            y_pred[i, 0] = 1
        elif (sum < 0):
            y_pred[i, 0] = -1
if(loss == "quadratic" and kernel == "polynomial"):
    for i in range (0, n_test):
        sum = 0
        for k in range (0, n):
            if (alpha[k, 0] > 0):
                temp = c + np.dot(D[k, :], D_test[i, :].transpose())
                sum = sum + alpha[k, 0] * y_train[k, 0] * temp * temp
        if (sum > 0):
            y_pred[i, 0] = 1
        elif (sum < 0):
            y_pred[i, 0] = -1
# Compute prediction accuracy:
sum_true = 0
for i in range (0, n_test):
    if (y_pred[i, 0] == y_test[i, 0]):
        sum_true = sum_true + 1
    else:
        continue
accur = sum_true/n_test

# ---------------------------------Print some results-----------------------------------------------
if (loss == "quadratic" and kernel == "linear"):
    print("Quadratic loss and linear kernel:")
    print("   regularization constant C = %d." %(C))
    print("   Number of support vectors is %d." %(num_sv))
    print("   Prediction accuracy = %f." %(accur))
    print("   Weight vector is:")
    print(w)
    print("   The bias is %f." %(w[26]))

if (loss == "quadratic" and kernel == "gaussian"):
    print("Quadratic loss and Gaussian kernel:")
    print("   The best result is generated by spread = %d." %(spread))
    print("   regularization constant C = %d." %(C))
    print("   Number of support vectors is %d." %(num_sv))
    print("   Prediction accuracy = %f.\n" %(accur))

if (loss == "quadratic" and kernel == "polynomial"):
    print("Quadratic loss and polynomial kernel:")
    print("   regularization constant C = %d." %(C))
    print("   q = %d and c = %f." %(q, c))
    print("   Number of support vectors is %d." %(num_sv))
    print("   Prediction accuracy = %f.\n" %(accur))