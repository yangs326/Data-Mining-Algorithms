import pandas as pd
import numpy as np
import math

# Load in entire dataset first:
data = pd.read_csv("energydata_complete.csv", sep=',', header=None)
# Convert the matrix into a matrix and drop the first and last columns:
data = np.array(data)
attribute = data[0, 1:28]
data = data[1:data.shape[0], 1:28]
data = data.astype('float32')
nrows = data.shape[0]

# Apply one-hot encoding:
y_class = np.empty([nrows, 1])
for i in range (0, nrows):
    if (10 <= data[i, 0] <= 40):
        y_class[i, 0] = 0
    if (data[i, 0] == 50):
        y_class[i, 0] = 1
    if (data[i, 0] == 60):
        y_class[i, 0] = 2
    if (70 <= data[i, 0] <= 90):
        y_class[i, 0] = 3
    if (100 <= data[i, 0] <= 160):
        y_class[i, 0] = 4
    if (170 <= data[i, 0] <= 1080):
        y_class[i, 0] = 5

# Concatenate the true cluster label to dataset:
D = np.hstack((data[:, 1:(data.shape[1])], y_class))
Y = D[:, -1]

for i in range(0,100):
    D = pd.DataFrame(D).sample(frac=1)
D = np.array(D)

D = D[0:100, :]
n_row = D.shape[0]
d_col = D.shape[1]

# Define parameters:
eps = 0.0001
maxiter = 1
spread = 10
k = 6
t = 0

# Define Gaussian kernel mapping function:
def gaussian(x, y, spread):
    norm = np.linalg.norm(x-y)
    feature = math.exp(-norm/(2 * spread))
    return feature

# Define function to calculate fraction of points assigned:
def frac(df1, df2, k, n):
    sum = 0
    for i in range(0, k):
        list1 = df1[i].index
        list2 = df2[i].index
        inter = len(set(list1).intersection((list2)))
        sum = sum + inter
    fraction = 1 - sum/n
    return fraction

# Randomly partition the points into k clusters:
shuffled = pd.DataFrame(D).sample(frac=1)
partition = np.array_split(shuffled, k) #Partition is of type "list"

# Useful codes:
#df = partition[0]
#print(df)
#x = np.array(df.iloc[0]) #Transform the first row of df to array vector
#y = np.array(df.iloc[1]) #Transform the second row of df to array vector
#df.loc[7] = list(D[0, 0:d_col]) #Add the first row of D to df and index it as 12
#if 6 not in set(list(df.index)):
    #df = df.drop(7)  #Delete the row with index 7 of df
#print(len(df))
#df1 = partition[0]
#df2 = partition[1]
#list1 = list(df1.index) #Transform the row index of df1 to list
#list2 = list(df2.index) #Transform the row index of df1 to list
#Num_CommonElements = len(set(list1).intersection(list2)) #Show the number of common elements in two lists using set()
#cluster = df.iloc[0][26] #Show the 26-th column element in the first row of df

# Repeat:
while (t < maxiter):
    print("Iteration %d\n" %(t))
    t = t + 1
    partition_t = list.copy(partition)
    # Compute squared norm of cluster means:
    sqnorm = np.empty(k)
    for i in range(0, k):
        size = len(partition_t[i])
        sum1 = 0
        for m in range(0, size):
            for n in range(0, size):
                x = np.array(partition_t[i].iloc[m])[0:d_col-1]
                y = np.array(partition_t[i].iloc[n])[0:d_col-1]
                kernel_map1 = gaussian(x, y, spread)
                sum1 = sum1 + kernel_map1
        sqnorm[i] = sum1 / (size * size)
    # Computer average kernel value for every points in D and Ci:
    sum2 = 0
    for j in range(0, n_row):
        #print(j)
        for i in range(0, k):
            distance = np.empty(k)
            size = len(partition_t[i])
            for a in range(0, size):
                x = np.array(partition_t[i].iloc[a])[0:d_col-1]
                y = D[j, 0:(d_col-1)]
                kernel_map2 = gaussian(x, y, spread)
                sum2 = sum2 + kernel_map2
                avg = sum2 / size
            distance[i] = sqnorm[i] - 2 * avg
        # Calculate cluster label for Xj:
        label = np.argmin(distance)
        print(label)
        # If no need to reassign:
        if j in set(list(partition_t[label].index)):
            continue
        elif j not in set(list(partition_t[label].index)):
            # Reassign point Xj to the new cluster:
            partition_t[label].loc[j] = list(D[j, 0:d_col])
            # Remove point Xj from its original cluster:
            for q in range(0, k):
                if (j in set(list(partition_t[q].index)) and q != label):
                    partition_t[q] = partition_t[q].drop(j)

    # Calculate fraction of points assigned:
    fraction = frac(partition, partition_t, k, n_row)
    if (fraction <= eps):
        break

    # Update cluster for next iteration
    partition = list.copy(partition_t)
    
# Compute purity score:
count = 0
for i in range(0, k):
    size = len(partition_t[i])
    for j in range (0, size):
        true_cluster = partition_t[i].iloc[j][d_col-1]
        if (true_cluster == i):
            count = count + 1
        else:
            continue
purity = count/n_row

# Print purity score:
print ("####### Kernel K-Means #######")
print("The purity score is: %f.\n" %(purity))

# Print the size of each cluster:
print("The size of cluster C1 is: %d." %(len(partition_t[0])))
print("The size of cluster C2 is: %d." %(len(partition_t[1])))
print("The size of cluster C3 is: %d." %(len(partition_t[2])))
print("The size of cluster C4 is: %d." %(len(partition_t[3])))
print("The size of cluster C5 is: %d." %(len(partition_t[4])))
print("The size of cluster C6 is: %d.\n" %(len(partition_t[5])))

# Print some parameters:
print("eps = %f" %(eps))
print("spread = %d." %(spread))