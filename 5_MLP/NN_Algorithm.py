import pandas as pd
import numpy as np
import random

# Load in the data files:
train = pd.read_csv("train.txt", sep=',', header=None)
test = pd.read_csv("test.txt", sep=',', header=None)

# Convert the data into a matrix:
train_m = np.array(train)
test_m = np.array(test)

# Set the true response vector:
Y = train_m[:, train_m.shape[1]-1]
Y_test = test_m[:, train_m.shape[1]-1]

# Set n, d, m, and p:
n = train_m.shape[0]
n_test = test_m.shape[0]
d = train_m.shape[1]-1
m = 30
p = 7

# Set the input matrix:
x_train = train_m[:, 0:d]
x_test = test_m[:, 0:d]
X0 = np.ones([n,1])

# Set the true response matrix:
y = train_m[:, d].reshape(-1)
Y = np.zeros([n, p+1])
Y[np.arange(n), y] = 1
Y = np.delete(Y, 0, 1)

# Set the hidden layer:
z_train = np.empty([n, m])
# Set the output matrix:
o_train = np.empty([n, p])

# Set the initial weight matrix and bias vector:
wh = np.random.uniform(-0.01, 0.01, [d,m])
wo = np.random.uniform(-0.01, 0.01, [m,p])
bh = np.random.uniform(-0.01, 0.01, m)
bo = np.random.uniform(-0.01, 0.01, p)
#print(wh)
#print(wo)
#print(bh)
#print(bo)

# Set the max epoch number and eta:
t = 180
itr = 0
eta = 0.00001

# Define the ReLU function:
def ReLU(x):
    if x <= 0:
        y = 0
    if x > 0:
        y = x
    return y

# Define the derivative of ReLu function:
def d_ReLu(x):
    if x > 0:
        y = 1
    if x <= 0:
        y = 0
    return y

# Define the Softmax function:
def Softmax(array):
    exp = np.zeros(len(array))
    Softmax = np.zeros(len(array))
    sum = 0
    for i in range(len(array)):
        exp[i] = np.exp(array[i])
        sum += exp[i]
    for i in range(len(array)):
        Softmax[i] = exp[i]/sum
    return Softmax

# Repeat the iterations:
while itr < t:

    # Generate random order:
    r = list(range(n))
    random.shuffle(r)

    for i in r:
        # Feed-forward phase:
        # Compute the hidden layer matrix:
        for k in range(0, m):
            # Calculate net input at Zk:
            net_Zk = np.dot(wh.transpose(), x_train[i]) + bh
            for j in range(0, net_Zk.shape[0]):
                z_train[i, j] = ReLU(net_Zk[j])

        # Compute the output matrix:
        for k in range(0, p):
            # Calculate net input at Ok:
            net_Ok = np.dot(wo.transpose(), z_train[i]) + bo
            o_train[i] = Softmax(net_Ok)
        # print(o_train)

        # Do Backpropagation:
        # Compute delta_o:
        delta_o = o_train[i, :] - Y[i, :]

        # Compute delta_h:
        delta_h = np.empty([m, ])
        d_z = np.empty([m, ])
        for j in range(0, m):
            # Compute derivatives of hidden layer at Zj:
            d_z[j] = d_ReLu(z_train[i, j])
        delta_h = np.multiply(d_z, np.dot(delta_o, wo.transpose()))

        # Upgrade gradient descent for bias vectors:
        descent_bo = delta_o
        bo = bo - eta * descent_bo
        descent_bh = delta_h
        bh = bh - eta * descent_bh

        # Uprade gradient descent for weight matrices:
        descent_wo = np.outer(z_train[i, :], delta_o)
        wo = wo - eta * descent_wo
        descent_wh = np.outer(x_train[i, :], delta_h)
        wh = wh - eta * descent_wh

    itr = itr + 1
    #print(itr)

# Try the test data:
# Set the hidden layer:
z_test = np.empty([n_test, m])
# Set the output matrix:
o_test = np.empty([n_test, p])

for i in range(n_test):
    # Compute the hidden layer matrix:
    for k in range(0, m):
        # Calculate net input at Zk:
        net_Zk_test = np.dot(wh.transpose(), x_test[i]) + bh
        for j in range(0, net_Zk_test.shape[0]):
            z_test[i, j] = ReLU(net_Zk_test[j])

    # Compute the output matrix:
    for k in range(0, p):
        # Calculate net input at Ok:
        net_Ok_test = np.dot(wo.transpose(), z_test[i]) + bo
        o_test[i] = Softmax(net_Ok_test)
    # print(o_train)

# Generate the predicted output for testing data:
y_output = np.zeros([n_test,])
for i in range(n_test):
    y_output[i] = np.argmax(o_test[i,:]) + 1

# Calculate the accuracy:
count = 0
for i in range(n_test):
    if y_output[i] == Y_test[i]:
        count += 1

accuracy = count/n_test

print("The size of the hidden layer is:")
print(m, "\n")

print("The number of epochs is:")
print(itr, "\n")

print("The eta used is:")
print(eta, "\n")

print("The weight matrices for the hidden layer is:")
print(wh, "\n")
print("The weight matrices for the output layer is:")
print(wo, "\n")

print("The bias vectors for the hidden layer is:")
print(bh, "\n")
print("The bias vectors for the output layer is:")
print(bo, "\n")

print("The prediction accuracy rate is:")
print(accuracy)
