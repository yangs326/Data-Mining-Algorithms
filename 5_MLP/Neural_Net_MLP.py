import pandas as pd
import numpy as np
import random

maxiter = 1000
eta = 0.000001
hidden_size = 18
hidden_number = 30

# Define ReLU function:
def ReLU(x):
    y = np.empty(len(x))
    for i in range (0, len(x)):
        if (x[i] <= 0):
            y[i] = 0
        elif (x[i] > 0):
            y[i] = x[i]
    return y

# Define the derivative of ReLu function (25.9):
def d_ReLu(x):
    y = np.empty(len(x))
    for i in range (0, len(x)):
        if x[i] > 0:
            y[i] = 1
        if x[i] <= 0:
            y[i] = 0
    return y

'''
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
'''

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

# Shuffle the dataset first to make train-test split:
ind = np.arange(data.shape[0])
np.random.shuffle(ind)
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

# Define input & output layer size, hidden layer size, and hidden layer number:
n = train.shape[0]
d = train.shape[1]-1
n0 = d               #Input layer size
p = 4                #Output layer size
m = hidden_size      #Hidden layer size
h = hidden_number    #Hidden layer number

# Define input matrix D:
D = train[:, 1:train.shape[1]]
# Define output matrix O:
O = np.empty([n, p])

# Initialize the weight matrices and bias vectors:
b = []
w = []
# Initialize the net gradients container:
delta = [None] * (h+1)
#print(delta)
# Initialize the net container that holds (h+1) layers of net inputs:
net = [None] * (h+1)

# Set b and w for input layer:
b.append(np.random.uniform(-0.01, 0.01, m))
w.append(np.random.uniform(-0.01, 0.01, [n0, m]))
# Set b's and w's between the hidden layers:
for i in range (1, h):
        b.append(np.random.uniform(-0.01, 0.01, m))
        w.append(np.random.uniform(-0.01, 0.01, [m, m]))
# Set b and w for the output layer:
b.append(np.random.uniform(-0.01, 0.01, p))
w.append(np.random.uniform(-0.01, 0.01, [m, p]))

'''
#print(b[0],"\n")
#print(w[0].shape)
#print(D[0, :].shape)
temp = w[0].transpose() @ D[0, :]
net = b[0] + temp
print(ReLU(net))
'''

# Set iteration counter:
t = 0
n = 10
# Iteration steps:
while (t < maxiter):
    # Generate random order:
    r = list(range(n))
    random.shuffle(r)
    for i in r:
        # Feed-Forward Phase:
        z = []
        z.append(D[i, :])
        for l in range (0, h+1):         
            net[l] = b[l] + (w[l].transpose() @ z[l])
            # Use ReLU activation for hidden layers:
            z_next = ReLU(net[l])
            z.append(z_next)
        # Get output neuron:
        O[i, :] = z[h+1]
        #print(O[i, :])

        # Back-propagation Phase:
        # Compute net gradients at output layer using Softmax (25.52):
        delta[h] = O[i, :] - y_train[i]
        # Compute net gradients at hidden layer l:
        for l in range (h-1, -1, -1):
            delta[l] = d_ReLu(net[l]) * (w[l+1] @ delta[l+1])

        # Gradient descent step:
        for l in range (0, h+1):
            gradient_w = np.array([z[l]]).transpose() @ np.array([delta[l]])
            gradient_b = delta[l]
            # Update W(l) and b(l):
            w[l] = w[l] - eta * gradient_w
            b[l] = b[l] - eta * gradient_b
    # Update iteration counter
    t = t + 1

# Used the trained model to generate predictions testing data classification:
n_test = test.shape[0]
# Define input matrix D_test:
D_test = test[:, 1:test.shape[1]]
# Define output matrix O_test:
O_test = np.empty([n_test, p])

# Apply feed-forward steps and calculate the predicted output O_test
# Initialize the net_test container that holds (h+1) layers of net inputs:
net_test = [None] * (h+1)
for i in range (0, n_test):
    z_test = []
    z_test.append(D_test[i, :])
    for l in range(0, h+1):
        net_test[l] = b[l] + (w[l].transpose() @ z_test[l])
        # Use ReLU activation for hidden layers:
        z_next_test = ReLU(net_test[l])
        z_test.append(z_next_test)
    # Get output neuron:
    O_test[i, :] = z_test[h+1]

# Convert the outputs into class labels:
y_hat = np.empty([n_test, 1])
for i in range (0, n_test):
    max_p = max(O_test[i, 0],O_test[i, 1],O_test[i, 2],O_test[i, 3])
    if (max_p == O_test[i, 0]):
        y_hat[i, 0] = 1
    if (max_p == O_test[i, 1]):
        y_hat[i, 0] = 2
    if (max_p == O_test[i, 2]):
        y_hat[i, 0] = 3
    if (max_p == O_test[i, 3]):
        y_hat[i, 0] = 4

# Compute prediction accuracy:
sum_true = 0
for i in range (0 ,n_test):
    if (y_hat[i, 0] == y_test_class[i, 0]):
        sum_true = sum_true + 1
    else:
        continue
accur = sum_true/n_test

# -------------------------------------------------------------------------------------------------------------------------------------------
# Print some results:
print("After trying some combinations of the parameters, it has been noticed that the accuracy rate stops increasing at around 50%.\n")
print("The threshold parameters used that yield the best accuracy are:")
print("step_size_eta = %f, size_of_hidden_layer = %d, number_of_hidden_layers = %d.\n" %(eta, m, h))
print("The final prediction accuracy rate on testing data is:")
print(accur, "\n")
print("There %d*%d dimensional weight matrix between the input and first hidden layer is:" %(d, m))
print(w[0], "\n")
print("# -----------------------------------------------------------------------------------------------------------------------------------")
print("The %d*%d dimensional weight matrices between the %d hidden layers are:" %(m, m, h))
print(w[1:h], "\n")
print("# -----------------------------------------------------------------------------------------------------------------------------------")
print("The %d*%d dimensional weight matrix between the last hidden layer and output layer is:" %(m, p))
print(w[h], "\n")
print("# -----------------------------------------------------------------------------------------------------------------------------------")
print("The %d dimensional bias vectors between the input and %d hidden layers are:" %(m, h))
print(b[0:h], "\n")
print("# -----------------------------------------------------------------------------------------------------------------------------------")
print("The %d dimensional bias vector between the last hidden layer and output layer is:" %(p))
print(b[h], "\n")
# -------------------------------------------------------------------------------------------------------------------------------------------
