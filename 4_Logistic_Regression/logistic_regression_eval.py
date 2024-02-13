import pandas as pd
import numpy as np
import random

# -------------------------------------------------------------
# Run for the train dataset first:
# Load in the data files:
train = pd.read_csv("train.txt", sep=',', header=None)
test = pd.read_csv("test.txt", sep=',', header=None)

# Convert the data into a matrix:
train_m = np.array(train)
test_m = np.array(test)

# Calculate the augmented data matrix D:
X0 = np.ones([train_m.shape[0], 1], dtype=np.float32)
D = np.hstack((X0, train_m[:, 0:train_m.shape[1]-1]))
n = D.shape[0]
d = D.shape[1]-1
#print(D)

# Set the true response vector:
Y = train_m[:, train_m.shape[1]-1]

# Define the initial weight vector:
w0 = np.zeros(d+1, dtype=np.float32)
w1 = np.copy(w0)

# Set the iteration number:
t = 0
# Set the learning rate and tolerence:
eta = 0.0009
epsilon = 0.001

# Define the logistic function:
def logis_fun(z):
    output = np.exp(z)/(1+np.exp(z))
    return(output)

while True:

    # Generate random order:
    r = list(range(n))
    random.shuffle(r)

    for i in r:
        # Compute the gradient at xi:
        z = np.dot(np.transpose(w1), D[i, :])
        theta = logis_fun(z)
        gradient = (Y[i] - theta) * D[i, :]
        w1 = w1 + eta * gradient

    if np.linalg.norm(w1 - w0) <= epsilon:
        break
    else:
        w0 = np.copy(w1)
        t = t + 1

# Test the training result:
X0 = np.ones([test_m.shape[0], 1], dtype=np.float32)
D_test = np.hstack((X0, test_m[:, 0:test_m.shape[1]-1]))
y_pred = np.dot(D_test, w1).reshape(-1)

def categorize_y(y):
    z = logis_fun(y)
    if z >= 0.5:
        return 1
    if z < 0.5:
        return 0

binary_y = np.zeros([n,])
y_test = test_m[:,d].reshape(-1)
n_test = test_m.shape[0]

for i in range(n_test):
    binary_y[i] = categorize_y(y_pred[i])

# Calculate the accuracy:
count = 0
for i in range(n_test):
    if binary_y[i] == y_test[i]:
        count += 1

acc = count/n_test
print("Assume eta = 0.0009, epsilon = 0.001. \n")
print("The accuracy rate using SGA algorithm of Logistic Regression is:")
print(acc, "\n")
print("w is:")
print(w1, "\n")

