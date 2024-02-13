import math
import numpy as np
import random
import matplotlib.pyplot as plt

# Part II: Diagonals in High Dimensions
n = 100000
def pmf_plt(x, dim):
    heights, bins = np.histogram(x, bins=25)
    heights = heights/sum(heights)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    plt.bar(bins, heights, width=(max(bins) - min(bins))/len(bins), color="purple", alpha=0.5)
    plt.title("d=%d" %(dim))
    plt.show()
'''
# Generate the half-diagonals in the 10-dimensional hypercube:
X1 = np.empty([n, 1], dtype=np.float32)
for i in range(0, n):
    half_diag1 = random.choices([-1/2, 1/2], k=10)
    half_diag2 = random.choices([-1/2, 1/2], k=10)
    # Compute the angles between the diagonal pair:
    unit_diag1 = half_diag1 / np.linalg.norm(half_diag1)
    unit_diag2 = half_diag2 / np.linalg.norm(half_diag2)
    dot_product = np.dot(unit_diag1, unit_diag2)
    angle = np.arccos(dot_product)
    X1[i, 0] = angle * 180 / math.pi
# Compute the min, max, value range, mean and variance of X1:
min1 = X1.min()
max1 = X1.max()
val_range1 = max1 - min1
mean1 = X1.mean()
var1 = X1.var()
print("If d=10, min=%f, max=%f, value range=%f, mean=%f, and variance=%f.\n" %(min1,max1,val_range1,mean1,var1))
# Plot pmf:
pmf_plt(X1, 10)

# Generate the half-diagonals in the 100-dimensional hypercube:
X2 = np.empty([n, 1], dtype=np.float32)
for i in range(0, n):
    half_diag1 = random.choices([-1/2, 1/2], k=100)
    half_diag2 = random.choices([-1/2, 1/2], k=100)
    # Compute the angles between the diagonal pair:
    unit_diag1 = half_diag1 / np.linalg.norm(half_diag1)
    unit_diag2 = half_diag2 / np.linalg.norm(half_diag2)
    dot_product = np.dot(unit_diag1, unit_diag2)
    angle = np.arccos(dot_product)
    X2[i, 0] = angle * 180 / math.pi
# Compute the min, max, value range, mean and variance of X1:
min2 = X2.min()
max2 = X2.max()
val_range2 = max2 - min2
mean2 = X2.mean()
var2 = X2.var()
print("If d=100, min=%f, max=%f, value range=%f, mean=%f, and variance=%f.\n" %(min2,max2,val_range2,mean2,var2))
# Plot pmf:
pmf_plt(X2, 100)
'''

# Generate the half-diagonals in the 1000-dimensional hypercube:
X3 = np.empty([n, 1], dtype=np.float32)
for i in range(0, n):
    half_diag1 = random.choices([-1/2, 1/2], k=1000)
    half_diag2 = random.choices([-1/2, 1/2], k=1000)
    # Compute the angles between the diagonal pair:
    unit_diag1 = half_diag1 / np.linalg.norm(half_diag1)
    unit_diag2 = half_diag2 / np.linalg.norm(half_diag2)
    dot_product = np.dot(unit_diag1, unit_diag2)
    angle = np.arccos(dot_product)
    X3[i, 0] = angle * 180 / math.pi
# Compute the min, max, value range, mean and variance of X1:
min3 = X3.min()
max3 = X3.max()
val_range3 = max3 - min3
mean3 = X3.mean()
var3 = X3.var()
print("If d=1000, min=%f, max=%f, value range=%f, mean=%f, and variance=%f.\n" %(min3,max3,val_range3,mean3,var3))
# Plot pmf:
pmf_plt(X3, 1000)
