import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

args = sys.argv
f1 = args[1]
f2 = args[2]
alpha = float(f2)

def threeD_plot(x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = A[:, 0]
    y = A[:, 1]
    z = A[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('u3')
    plt.title("Best Three-dimensional Approximation")
    plt.show()

# Load in data file:
data = pd.read_csv(f1+'.csv', sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
# Calculate mean vector and centered data matrix:
nrows = data.shape[0]
ncols = data.shape[1]
mean_vector = np.sum(data, axis=0)/nrows
centered_data = data - mean_vector
# Calculate total variance:
squared_norm = np.empty(data.shape[0], dtype=np.float32)
for i in range(0, data.shape[0]):
    norm = np.linalg.norm(centered_data[i, :])
    squared_norm[i] = norm * norm
total_var = squared_norm.sum()/data.shape[0]

# Calculate the Covariance Matrix:
sum_inner = np.empty([ncols, ncols], dtype=np.float32)
for i in range(0, ncols):
    for j in range(0, ncols):
        zt = (centered_data[:, j]).transpose()
        z = (centered_data[:, i])
        sum_inner[i, j] = np.dot(z,zt)
cov = sum_inner/nrows

# Calculate the eigenvectors and eigen values:
eigen_vals, eigen_vecs = np.linalg.eig(cov)
# Sort the eigen values from largest to smallest:
idx = eigen_vals.argsort()[::-1]
eigen_vals = eigen_vals[idx]
eigen_vecs = eigen_vecs[:, idx]

# Compute fraction and choose dimensionality:
sum_eig = 0
for i in range(0, len(eigen_vals)):
    sum_eig = sum_eig + eigen_vals[i]
    frac = sum_eig/eigen_vals.sum()
    if (frac >= alpha):
        r = i
        break
print("The number of dimensions required to capture alpha=0.975 is:")
print("r = %d.\n" %(r+1))
# Generate reduced basis for the first three PCs:
Ur = eigen_vecs[:, 0:3]
# Get reduced dimensionality data:
A = np.empty([nrows, 3], dtype=np.float32)
for i in range(0, nrows):
    A[i, :] = Ur.transpose() @ centered_data[i, :]
print("The three largest eigen values are: %f, %f, and %f.\n" %(eigen_vals[0], eigen_vals[1], eigen_vals[2]))
print("The three PCs are:")
print(Ur, "\n")

# Calculate Mean Squared Error:
# Method 1:
row = A.shape[0]
mean_A = np.sum(A, axis=0)/row
centered_A = A - mean_A
squared_norm_A = np.empty(row, dtype=np.float32)
for i in range(0, row):
    norm = np.linalg.norm(centered_A[i, :])
    squared_norm_A[i] = norm * norm
var_A = squared_norm_A.sum()/row
mse1 = total_var - var_A
# Method 2:
mse2 = total_var - eigen_vals[0:3].sum()
print("The mean squared error is mse = %f.\n" %(mse1))
print("The reduced dimensionality dataset A on the first three PCs is:")
print(A,"\n")
# Plot A in 3D plot"
threeD_plot(A)