import pandas as pd
import numpy as np

print("###### train dataset ######")
# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
data = pd.DataFrame(data)
#print(data.shape)

# Split the dataset into training and testing set.
'''
ind = np.arange(data.shape[0])
np.random.shuffle(ind)
train = data.iloc[ind[:int(0.7*data.shape[0])], :]
test = data.iloc[ind[int(0.7*data.shape[0]):], :]
'''
# Use the first 70% rows of dataset as the training set, and the remaining 30% as test set:
train = data.iloc[0:int(0.7*data.shape[0]), :]
test = data.iloc[int(0.7*data.shape[0]):, :]
#print(train.shape)
#print(test.shape)

# Convert the train data frame into a matrix:
train = np.array(train)

# Calculate the augmented data matrix D:
X0 = np.ones([train.shape[0], 1], dtype=np.float32)
D = np.hstack((X0, train[:, 1:train.shape[1]]))
n = D.shape[0]
d = D.shape[1]-1
#print(D[0:10, :])

# Do QR decomposition:
# Set up the Q matrix:
Q = np.empty([n, d+1])
Q[:, 0] = np.copy(D[:, 0])
# Set up the R matrix:
R = np.zeros([d+1, d+1])
# Set up the diagonal values of R:
for i in range(0, d+1):
    R[i, i] = 1
# Calculate the rest of the values in R:
for i in range(0, d):
    # Calculate P[i,j]:
    for j in range(i+1, d+1):
        R[i, j] = np.dot(D[:,j].transpose(), Q[:,i])/np.linalg.norm(Q[:,i])/np.linalg.norm(Q[:,i])
    # Calculate U(i+1):
    sum = 0
    for x in range(0, i+1):
        sum = sum + R[x, i+1] * Q[:, x]
    Q[:, i+1] = D[:, i+1] - sum
print("The Q matrix of the training data is:")
print(Q, "\n")
#print(Q.shape, "\n")

# Calculate the 27*27 delta matrix, which records the squared norms of the basis vectors:
delta = np.zeros([d+1, d+1], dtype=np.float32)
for i in range(0, d+1):
    delta[i, i] = np.linalg.norm(Q[:, i]) * np.linalg.norm(Q[:, i])
# Calculate the inverse of delta matrix:
#delta_inv1 = np.linalg.inv(delta)
delta_inv = np.zeros([d+1, d+1], dtype=np.float32)
for i in range(0, d+1):
    delta_inv[i, i] = 1/delta[i, i]
#print(delta_inv.shape)

# Calculate the dot product of delta_inv, Q transpose and response variable Y:
S = np.empty([d+1, 1], dtype=np.float32)
S = np.dot((np.dot(delta_inv, Q.transpose())), train[:, 0])

# Set up the weight vectors:
w = np.empty(d+1, dtype=np.float32)
# Apply backward solution:
w[d] = S[d]
for i in range(0, d):
    sum = 0
    for x in range(0, i+1):
        sum = sum + w[d-x] * R[d-i-1, d-x]
    w[d-i-1] = S[d-i-1] - sum
print("The weight vectors are:")
print(w, "\n")

# Calculate the L2 norm of the weight vectors:
w_norm = np.linalg.norm(w, ord=2)
print("The L2 norm of the weight vectors of the training data is:")
print(w_norm, "\n")

# Compute the predicted response vector:
y_predict = np.dot(D, w)
print("The predicted response of train dataset is:")
print(y_predict)

# Calculate SSE:
SSE = 0
for i in range(0, n):
    temp = train[i, 0] - y_predict[i]
    SSE = SSE + temp * temp
print("The SSE on the training data is:")
print(SSE, "\n")

# Calculate MSE:
MSE = SSE/n
print("The MSE on the training data is:")
print(MSE, "\n")

# Calculate TSS:
mean_y = np.sum(train[:, 0])/n
TSS = 0
for i in range(0, n):
    TSS = TSS + (train[i, 0] - mean_y) * (train[i, 0] - mean_y)

# Calculate R^2:
R_squared = (TSS-SSE)/TSS
print("The R^2 on the training data is:")
print(R_squared, "\n")

# --------------------------------------------------------------------------------------
# Apply linear regression to test dataset:
print("###### test dataset ######")
# Convert the test data frame into a matrix:
test = np.array(test)

# Calculate the augmented data matrix D:
X0 = np.ones([test.shape[0], 1], dtype=np.float32)
D_test = np.hstack((X0, test[:, 1:test.shape[1]]))
n = D_test.shape[0]
d = D_test.shape[1]-1

# Compute the predicted response vector for test data:
y_predict_test = np.dot(D_test, w)
print("The predicted response of test dataset is:")
print(y_predict_test)

# Calculate SSE:
SSE_test = 0
for i in range(0, n):
    temp = test[i, 0] - y_predict_test[i]
    SSE_test = SSE_test + temp * temp
print("The SSE on the testing data is:")
print(SSE_test, "\n")

# Calculate MSE:
MSE_test = SSE_test/n
print("The MSE on the testing data is:")
print(MSE_test, "\n")

# Calculate TSS:
mean_y_test = np.sum(test[:, 0])/n
TSS_test = 0
for i in range(0, n):
    TSS_test = TSS_test + (test[i, 0] - mean_y_test) * (test[i, 0] - mean_y_test)

# Calculate R^2:
R_squared_test = (TSS_test-SSE_test)/TSS_test
print("The R^2 on the testing data is:")
print(R_squared_test, "\n")