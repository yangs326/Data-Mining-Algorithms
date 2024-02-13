import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.95
sigma = 10000

print("###### Gaussian Kernel ######")
print("The spread is sigma^2 = %d:\n" %(sigma*sigma))
# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
n = data.shape[0]
d = data.shape[1]

# Compute the Gaussian kernel matrix:
n = 5000

K_gauss = np.zeros([n, n], dtype=np.float32)
for i in range(0, n):
    for j in range(0, n):
        distance = np.linalg.norm(data[i, :] - data[j, :])
        K_gauss[i, j] = math.exp(-(distance * distance)/(2*sigma*sigma))
print("The Gaussian kernel is:")
print(K_gauss, "\n")

# Step2. Center the gaussian kernel matrix:
# Create the n*n identity matrix:
I = np.identity(n)
# Create the n*n 1's matrix:
one_matrix = np.ones([n, n], dtype=np.float32)
# Compute the centered gaussian kernel:
K_gauss_center = (I - one_matrix/n) @ K_gauss @ (I - one_matrix/n)

# Step3-4. Get the eigenvalues and eigenvectors of centered gaussian kernel matrix:
# Calculate the eigenvectors and eigen values:
eigen_vals, eigen_vecs = np.linalg.eig(K_gauss_center)
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
print("The number of dimensions required to capture alpha=0.95 is:")
print("r = %d.\n" %(r+1))
print("In Gauss Kernel, the three largest eigenvalues are: %f, %f and %f.\n" %(unit_eigen_vals[0], unit_eigen_vals[1], unit_eigen_vals[2]))

# Step8. Get the first r unit eigenvectors that capture alpha=0.95:
unit_eigen_vecs = np.zeros([n, r+1], dtype=np.float32)
for i in range(0, r+1):
        unit_eigen_vecs[:, i] = math.sqrt(1/eigen_vals[i]) * eigen_vecs[:, i]
#print(unit_eigen_vecs)

# Step 9. Get reduced basis for the first 2 PCs:
Cr = unit_eigen_vecs[:, 0:2]
print("In Gaussian kernel PCA, the two dominant PCs are:")
print("u1:")
print(Cr[:, 0])
print("u2:")
print(Cr[:, 1], "\n")

# Step10. Get the reduced dimensionality data:
A_gauss = np.empty([n, 2], dtype=np.float32)
A_gauss =  K_gauss_center @ Cr
print("In Gaussian Kernel, the projected data onto the two dominant PCs is:")
print(A_gauss)

# Created scattered plot for A_gauss:
sns.scatterplot(x=A_gauss[:, 0], y=A_gauss[:, 1])
plt.title("Gaussian Kernel: Projection onto Two Dominant PCs")
plt.show()