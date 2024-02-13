import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load in data file:
data = pd.read_csv('airfoil_self_noise.dat',sep='\t',header=None)
# Convert the matrix into a matrix:
data = np.array(data)

# Part I.a: Calculate the Mean Vector
# - The mean vector is calculated as:
mean_vector = np.sum(data, axis=0)/data.shape[0]
print("The mean vector is:")
print(mean_vector, "\n")

# - Compute the Total variance:
squared_norm = np.empty(data.shape[0], dtype=np.float64)
for i in range(0, data.shape[0]):
    norm = np.linalg.norm(data[i, :]-mean_vector)
    squared_norm[i] = norm * norm

#mean_norm = np.linalg.norm(mean_vector)
#mean_norm = mean_norm * mean_norm
total = squared_norm.sum()/data.shape[0]
print("The total variance using (1.4) is:")
print(total, "\n")

# First generate the centered data matrix:
centered_data = data - mean_vector
squared_norm = np.empty(data.shape[0], dtype=np.float32)
for i in range(0, data.shape[0]):
    norm = np.linalg.norm(centered_data[i, :])
    squared_norm[i] = norm * norm
total_var = squared_norm.sum()/data.shape[0]
print("The total variance using centered_data is:")
print(total_var, "\n")

# Part I.b: Calculate the Covariance Matrix
# - The Inner Product Form
sum_inner = np.empty([data.shape[1], data.shape[1]], dtype=np.float32)
# Use the column vector to calculate inner form covariance matrix:
for i in range(0, data.shape[1]):
    for j in range(0, data.shape[1]):
        zt = np.matrix(centered_data[:, j]).transpose()
        z = np.matrix(centered_data[:, i])
        sum_inner[i, j] = np.dot(z,zt)
cov_inner = sum_inner/data.shape[0]
print("The inner form covariance matrix is:")
print(cov_inner, "\n")

# - The Outer Product Form
sum_outer = np.empty([data.shape[1], data.shape[1]], dtype=np.float32)
# Use the row vector to calculate outer form covariance matrix:
for i in range(0, data.shape[0]):
    z = np.matrix(centered_data[i, :])
    zt = z.transpose()
    outer_z = np.dot(zt, z)
    sum_outer = sum_outer + outer_z
cov_outer = sum_outer/data.shape[0]
print("The outer form covariance matrix is:")
print(cov_outer, "\n")

# Part I.c Correlation matrix as pair-wise cosines
rho = np.empty([data.shape[1], data.shape[1]], dtype=np.float32)
# Use the column vectors and their inner products to calculate correlation matrix:
for i in range(0, data.shape[1]):
    for j in range(0, data.shape[1]):
        zt = np.array([centered_data[:, j]]).transpose()
        z = np.array([centered_data[:, i]])
        rho[i, j] = (z@zt)/(np.linalg.norm(zt) * np.linalg.norm(z))
print("The correlation matrix is:")
print(rho)
'''
print("The correlation matrix shows that Attribute 2 and Attribute 5 are the most correlated (rho = 0.753).")
print("Attribute 2 and Attribute 3 are the most anti-correlated (rho = -0.505).")
print("Attribute 3 and Attribute 4 are the least correlated (rho = 0.00378).", "\n")

# Plot the correlation matrix:
cover = np.zeros_like(rho, dtype=np.bool)
cover[np.triu_indices_from(cover)] = True
f, ax = plt.subplots(figsize=(10, 8))
colormap = sns.diverging_palette(220, 20, as_cmap=True)
# Draw a heatmap with the chosen format:
sns.heatmap(rho, mask=cover, cmap=colormap, center=-0, vmax=0.9, vmin=-0.9, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})

# Create the scattered plot for Attribute 2 and Attribute 5 (the most correlated):
#sns.scatterplot(x=data[:, 1], y=data[:, 4])

# Create the scattered plot for Attribute 2 and Attribute 3 (the most anti-correlated):
#sns.scatterplot(x=data[:, 1], y=data[:, 2])

# Create the scattered plot for Attribute 3 and Attribute 4 (the most anti-correlated):
#sns.scatterplot(x=data[:, 2], y=data[:, 3])
'''

# Part II. Compute the first two eigenvectors of the covariance matrix
# Ensure the dimension:
d = data.shape[1]
# Create the random dx2 matrix:
X0 = np.random.rand(d, 2)
# Calculate matrix X1:
X1 = np.dot(cov_inner, X0)
# Assign a and b:
a = X1[:, 0]
b = X1[:, 1]
# Orthogonalize b:
norm_a = np.linalg.norm(a)
b = b - np.dot(b.transpose(), a)/(norm_a*norm_a) * a
# Normalize a and b:
a = a/norm_a
b = b/np.linalg.norm(b)
# Reassign X1:
X1[:, 0] = a
X1[:, 1] = b
#print("The first X1 is: \n", X1, "\n")

# Calculate the distance between X1 and X0:
distance = np.linalg.norm(X1 - X0)

# Use a loop to iterate the above procedures until distance is within tolerance:
while distance > 0.001:
    X0 = X1
    X1 = np.dot(cov_inner, X1)
    a = X1[:, 0]
    b = X1[:, 1]
    norm_a = np.linalg.norm(a)
    b = b - (np.dot(b.transpose(), a) / (norm_a * norm_a)) * a
    a = a / norm_a
    b = b / np.linalg.norm(b)
    X1[:, 0] = a
    X1[:, 1] = b
    distance = np.linalg.norm(X1 - X0)

# The two eigenvectors can be be obtained now as:
u1 = X1[:, 0]
u2 = X1[:, 1]
print("The two eigenvectors are:")
print("u1 =", u1)
print("u2 =", u2, "\n")

'''
value, vector = np.linalg.eig(cov_inner)
print("v1 =", vector[:, 0])
print("v2 =", vector[:, 1], "\n")
'''

# Now project the input data matrix after centering onto the two eigenvectors:
projection = np.empty([data.shape[0], 2], dtype=np.float32)
c1 = np.array([u1]).transpose()
c2 = np.array([u2]).transpose()
eigen_matrix = np.concatenate((c1, c2), axis=1)
projection = centered_data @ eigen_matrix
#print(projection, "\n")

# Plot the projected points in the two new dimensions:
#sns.scatterplot(x=projection[:, 0], y=projection[:, 1])
#plt.show()