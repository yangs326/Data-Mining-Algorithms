import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load in data file:
data = pd.read_csv('energydata_complete.csv', sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')

# Part I.a: Calculate the Mean Vector
# - The mean vector is calculated as:
nrows = data.shape[0]
ncols = data.shape[1]
mean_vector = np.sum(data, axis=0)/nrows
print("Part I.a\nThe mean vector is:")
print(mean_vector, "\n")

# - Compute the Total variance:
centered_data = data - mean_vector # First generate the centered data matrix:
squared_norm = np.empty(data.shape[0], dtype=np.float32)
for i in range(0, data.shape[0]):
    norm = np.linalg.norm(centered_data[i, :])
    squared_norm[i] = norm * norm
total_var = squared_norm.sum()/data.shape[0]
print("The total variance is:")
print(total_var, "\n")

# Part I.b: Calculate the Covariance Matrix
# - The Inner Product Form (using attributes)
sum_inner = np.empty([ncols, ncols], dtype=np.float32)
for i in range(0, ncols):
    for j in range(0, ncols):
        zt = (centered_data[:, j]).transpose()
        z = (centered_data[:, i])
        sum_inner[i, j] = np.dot(z,zt)
cov_inner = sum_inner/nrows
print("Part I.b\nThe inner form covariance matrix is:")
print(cov_inner, "\n")

# - The Outer Product Form (using points)
sum_outer = np.empty([ncols, ncols], dtype=np.float32)
for j in range(0, nrows):
    zt = np.array([centered_data[j, :]]).transpose()
    z = np.array([centered_data[j, :]])
    sum_outer = sum_outer + zt@z
cov_outer = sum_outer/nrows
print("The outer form covariance matrix is:")
print(cov_outer, "\n")

# Part I.c Correlation matrix as pair-wise cosines
rho = np.empty([ncols, ncols], dtype=np.float32)
# Use the attribute cosines to calculate correlation matrix:
for i in range(0, ncols):
    for j in range(0, ncols):
        zt = np.array([centered_data[:, i]]).transpose()
        z = np.array([centered_data[:, j]])
        rho[i, j] = (z@zt) / np.linalg.norm(zt) / np.linalg.norm(z)
print("Part I.c\nThe correlation matrix is:")
print(rho, "\n")

# Find the max rho, least rho and min abs(rho)
max_rho = 0
min_rho = 0
abs_rho = 1
for i in range(0, ncols):
    for j in range(0, ncols):
        if (i != j):
            if (rho[i, j] > max_rho or rho[i, j] == max_rho):
                max_rho = rho[i, j]
                max_index = [i, j]
            if (rho[i, j] < min_rho or rho[i, j] == min_rho):
                min_rho = rho[i, j]
                min_index = [i, j]
            if (abs(rho[i, j]) > 0 and abs(rho[i, j]) < abs_rho):
                abs_rho = abs(rho[i, j])
                abs_index = [i, j]

# Plot the correlation matrix:
cover = np.zeros_like(rho, dtype=bool)
cover[np.triu_indices_from(cover)] = True
f, ax = plt.subplots(figsize=(10, 8))
colormap = sns.diverging_palette(220, 20, as_cmap=True)
#plt.title('Correlation Matrix', fontsize=16)

# Draw a heatmap with the chosen format:
#sns.heatmap(rho, mask=cover, cmap=colormap, center=0, vmax=1, vmin=-1, square=True, linewidths=0.5, cbar_kws={"shrink": 1.0})

# Create the scattered plot for attribute pair [20, 12] (the most correlated):
print("i) The most correlated attributes are %s and %s. Coefficient = %f.\n" % (attribute[12], attribute[20], max_rho))

#sns.scatterplot(x=data[:, 12], y=data[:, 20]).set_title("T6 and T_out")

# Create the scattered plot for attribute pair [14, 13] (the most anti-correlated):
print("ii) The most anti-correlated attributes are %s and %s. Coefficient = %f.\n" %(attribute[14], attribute[13], min_rho))
#sns.scatterplot(x=data[:, 13], y=data[:, 14]).set_title("T7 and RH_6")

# Create the scattered plot for attribute pair [26, 25] (the least correlated):
print("iii) The least correlated attributes are %s and %s. Coefficient = %f.\n" %(attribute[26], attribute[25], abs_rho))
#sns.scatterplot(x=data[:, 25], y=data[:, 26]).set_title("rv1 and Tdewpoint")

# Part II. Compute the first two eigenvectors of the covariance matrix
d = ncols
# Create the random dx2 matrix and normalize column vectors to unit length:
X0 = np.random.rand(d, 2)
X0[:, 0] = X0[:, 0]/np.linalg.norm(X0[:, 0])
X0[:, 1] = X0[:, 1]/np.linalg.norm(X0[:, 1])

# Calculate matrix X1:
X1 = np.dot(cov_inner, X0)
# Assign the column vectors a and b:
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
epsilon = np.linalg.norm(X1 - X0)

# Use a loop to iterate the above procedures until distance is within tolerance:
while epsilon > 0.0001:
    X0 = X1
    X1 = np.dot(cov_inner, X1)
    a = X1[:, 0]
    b = X1[:, 1]
    norm_a = np.linalg.norm(a)
    b = b - np.dot(b.transpose(), a) / (norm_a * norm_a) * a
    a = a / norm_a
    b = b / np.linalg.norm(b)
    X1[:, 0] = a
    X1[:, 1] = b
    epsilon = np.linalg.norm(X1 - X0)

# The two eigenvectors can be be obtained now as:
u1 = X1[:, 0]
u2 = X1[:, 1]
print("Part II.\nThe two eigenvectors are:")
print("u1 =", u1, "\n")
print("u2 =", u2, "\n")

# Now project the input data matrix after centering onto the two eigenvectors:
projection = np.empty([nrows, 2], dtype=np.float32)
c1 = np.array([u1]).transpose()
c2 = np.array([u2]).transpose()
eigen_matrix = np.concatenate((c1, c2), axis=1)
projection = centered_data @ eigen_matrix
print("The projected data points are:\n")
print(projection, "\n")

# Plot the projected points in the two new dimensions:
#sns.scatterplot(x=projection[:, 0], y=projection[:, 1])
#plt.title("Projection onto 2D")
#plt.show()