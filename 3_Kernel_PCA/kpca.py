import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.95
'''
print("###### Linear Kernel ######")
# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
n = data.shape[0]
d = data.shape[1]

# Step1. Compute the linear kernel matrix:
n = 5000
K_linear = np.zeros([n, n], dtype=np.float32)
for i in range(0, n):
    for j in range(0, n):
        K_linear[i, j] = np.dot(data[i, :], data[j, :].transpose())
print("The linear kernel is:")
print(K_linear, "\n")

# Step2. Center the linear kernel matrix:
# Create the n*n identity matrix:
I = np.identity(n)
# Create the n*n 1's matrix:
one_matrix = np.ones([n, n], dtype=np.float32)
# Compute the centered linear kernel:
K_linear_center = (I - one_matrix/n) @ K_linear @ (I - one_matrix/n)
#print(K_linear_center.shape)

# Step3-4. Get the eigenvalues and eigenvectors of centered linear kernel matrix:
# Calculate the eigenvectors and eigen values:
eigen_vals, eigen_vecs = np.linalg.eig(K_linear_center)
# Sort the eigen values from largest to smallest:
idx = eigen_vals.argsort()[::-1]
eigen_vals = eigen_vals[idx]
eigen_vecs = eigen_vecs[:, idx]
#print(eigen_vals)
#print(eigen_vecs)

# Step5. Compute variance for each component:
unit_eigen_vals = eigen_vals/n

# Step6-7. Compute fraction of total variance:
# Compute fraction and choose dimensionality:
sum_eig = 0
for i in range(0, len(unit_eigen_vals)):
    sum_eig = sum_eig + unit_eigen_vals[i]
    frac = sum_eig/unit_eigen_vals.sum()
    if (frac >= alpha):
        r = i
        break
print("In linear kernel PCA, the number of dimensions required to capture alpha=0.95 is:")
print("r = %d.\n" %(r+1))
print("The three dominant eigenvalues are %f, %f and %f.\n" %(unit_eigen_vals[0], unit_eigen_vals[1], unit_eigen_vals[2]))

# Step8. Get the first r unit eigenvectors that capture alpha=0.95:
unit_eigen_vecs = np.zeros([n, r+1], dtype=np.float32)
for i in range(0, r+1):
        unit_eigen_vecs[:, i] = math.sqrt(1/eigen_vals[i]) * eigen_vecs[:, i]
#print(unit_eigen_vecs)

# Step 9. Get reduced basis for the first 2 PCs:
Cr = unit_eigen_vecs[:, 0:2]
print("The two dominant PCs are:")
print("u1:")
print(Cr[:, 0])
print("u2:")
print(Cr[:, 1], "\n")

# Step10. Get the reduced dimensionality data:
A_linear = np.empty([n, 2], dtype=np.float32)
A_linear =  K_linear_center @ Cr
print("In linear kernel, the projected data onto the two dominant PCs is:")
print(A_linear)

# Created scattered plot for A_linear:
sns.scatterplot(x=A_linear[:, 0], y=A_linear[:, 1])
plt.title("Linear Kernel: Projection onto Two Dominant PCs")
plt.show()
'''
# -------------------------------------------------------------------------------------
# Apply regular PCA with covariance matrix
print("###### Regular PCA ######")
# Load in data file:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
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
print("In regular PCA, the number of dimensions required to capture alpha=0.95 is:")
print("r = %d.\n" %(r+1))
# Generate reduced basis for the first two PCs:
Ur = eigen_vecs[:, 0:2]
# Get reduced dimensionality data:
A_regular = np.empty([nrows, 2], dtype=np.float32)
for i in range(0, nrows):
    A_regular[i, :] = Ur.transpose() @ centered_data[i, :]
print("In regular PCA, the three largest eigenvalues are: %f, %f and %f.\n" %(eigen_vals[0], eigen_vals[1], eigen_vals[2]))
print("The two dominant regular PCs are:")
print(Ur, "\n")
print("In regular PCA, the projected data onto the two dominant PCs is:")
print(A_regular)

# Created scattered plot for regular PCA:
sns.scatterplot(x=A_regular[:, 0], y=A_regular[:, 1])
plt.title("Regular PCA: Projection onto Two Dominant PCs")
plt.show()
