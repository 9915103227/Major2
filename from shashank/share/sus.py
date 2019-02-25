import numpy as np

my_data = np.genfromtxt("data.txt", delimiter=",", skip_header=True)
print (my_data.min(axis=0)[1])
print (my_data.max(axis=0)[2])
