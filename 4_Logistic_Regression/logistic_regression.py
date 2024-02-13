import pandas as pd
import numpy as np
import random
import math

# Parameters:
maxiter = 1000
eta = 0.0000005
eps = 0.0001

# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
nrows = data.shape[0]

# Apply one-hot encoding:
y = np.empty([nrows, 4])
y_class = np.empty([nrows, 1])
for i in range (0, nrows):
    if (data[i, 0]<=30):
        y[i, 0] = 1
        y[i, 1] = 0
        y[i, 2] = 0
        y[i, 3] = 0
        y_class[i, 0] = 1
    if (data[i, 0]>30 and data[i, 0]<=50):
        y[i, 0] = 0
        y[i, 1] = 1
        y[i, 2] = 0
        y[i, 3] = 0
        y_class[i, 0] = 2
    if (data[i, 0]>50 and data[i, 0]<=100):
        y[i, 0] = 0
        y[i, 1] = 0
        y[i, 2] = 1
        y[i, 3] = 0
        y_class[i, 0] = 3
    if (data[i, 0]>100):
        y[i, 0] = 0
        y[i, 1] = 0
        y[i, 2] = 0
        y[i, 3] = 1
        y_class[i, 0] = 4

data = pd.DataFrame(data)
y = pd.DataFrame(y)
y_class = pd.DataFrame(y_class)

# Split the dataset into training and testing set.
# Use the first 70% rows of dataset as the training set, and the remaining 30% as test set:
train = data.iloc[0:int(0.7*data.shape[0]), :]
test = data.iloc[int(0.7*data.shape[0]):, :]
y_train = y.iloc[0:int(0.7*y.shape[0]), :]
y_test = y.iloc[int(0.7*y.shape[0]):, :]
y_train_class = y_class.iloc[0:int(0.7*y_class.shape[0]), :]
y_test_class = y_class.iloc[int(0.7*y_class.shape[0]):, :]

# Transform the dataframes into matrix:
data = np.array(data)
train = np.array(train)
test = np.array(test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train_class = np.array(y_train_class)
y_test_class = np.array(y_test_class)

# Calculate the augmented data matrix D:
X0 = np.ones([train.shape[0], 1], dtype=np.float32)
D = np.hstack((X0, train[:, 1:train.shape[1]]))
n = D.shape[0]
d = D.shape[1]-1

# Set iteration counter:
t = 0

# Set initial weight vector:
K = 4
w_1_t = np.zeros([1, d+1])
w_2_t = np.zeros([1, d+1])
w_3_t = np.zeros([1, d+1])
w_4_t = np.zeros([1, d+1]) # use as reference class

while t < maxiter:
    # Iteration steps:
    for i in range(0, K):
        w_1 = w_1_t
        w_2 = w_2_t
        w_3 = w_3_t
        w_4 = w_4_t
    # Generate random order:
    r = list(range(n))
    random.shuffle(r)
    for i in r:
        sum = 1 + math.exp(w_1 @ D[i, :]) + math.exp(w_2 @ D[i, :]) + math.exp(w_3 @ D[i, :])
        pi_1 = math.exp(w_1 @ D[i, :])/sum
        pi_2 = math.exp(w_2 @ D[i, :])/sum
        pi_3 = math.exp(w_3 @ D[i, :])/sum
        # Compute gradient at w1, w2, w3:
        grad_1 = (y_train[i, 0] - pi_1) * D[i, :]
        grad_2 = (y_train[i, 1] - pi_2) * D[i, :]
        grad_3 = (y_train[i, 2] - pi_3) * D[i, :]
        # Update weight vectors w1, w2, w3:
        w_1 = w_1 + eta * grad_1
        w_2 = w_2 + eta * grad_2
        w_3 = w_3 + eta * grad_3
    # Compute sum of weight errors:
    error = np.linalg.norm(w_1-w_1_t) + np.linalg.norm(w_2-w_2_t) + np.linalg.norm(w_3-w_3_t)
    # Update w1_t, w2_t, w3_t:
    w_1_t = w_1
    w_2_t = w_2
    w_3_t = w_3
    # Update t:
    t = t + 1
    # Compare error sum to convergence threshold:
    if (error <= eps):
        break
    else:
        continue

# Print results.
print("The parameters used are:")
print("maxiter = 1000, eta = 0.0000005, eps = 0.0001.\n")

# Print the 4 weight vectors:
print("The weight vectors are:")
print("w1:")
print(w_1)
print("w2:")
print(w_2)
print("w3:")
print(w_3)
print("w4:")
print(w_4, "\n")

# Calculate the augmented data matrix D_test:
X0_test = np.ones([test.shape[0], 1], dtype=np.float32)
D_test = np.hstack((X0_test, test[:, 1:test.shape[1]]))
n_test = D_test.shape[0]
d = D_test.shape[1]-1

# Generate predictions on testing data classification:
y_hat = np.empty([n_test, 1])
for i in range(0, n_test):
    sum = 1 + math.exp(w_1 @ D_test[i, :]) + math.exp(w_2 @ D_test[i, :]) + math.exp(w_3 @ D_test[i, :])
    pi_1 = math.exp(w_1 @ D_test[i, :])/sum
    pi_2 = math.exp(w_2 @ D_test[i, :])/sum
    pi_3 = math.exp(w_3 @ D_test[i, :])/sum
    pi_4 = 1/sum
    if (max(pi_1, pi_2, pi_3, pi_4) == pi_1):
        y_hat[i, 0] = 1
    elif (max(pi_1, pi_2, pi_3, pi_4) == pi_2):
        y_hat[i, 0] = 2
    elif (max(pi_1, pi_2, pi_3, pi_4) == pi_3):
        y_hat[i, 0] = 3
    elif (max(pi_1, pi_2, pi_3, pi_4) == pi_4):
        y_hat[i, 0] = 4

# Compute prediction accuracy:
sum_true = 0
for i in range (0, n_test):
    if (y_hat[i, 0] == y_test_class[i, 0]):
        sum_true = sum_true + 1
    else:
        continue
accur = sum_true/n_test
print("The final accuracy rate on test data is:")
print(accur, "\n")
print("After trying multiple eta, eps, and maxiter, we found out that the predicted accuracy rate is around 50%.")