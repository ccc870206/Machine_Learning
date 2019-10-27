import numpy as np


def generator_gaussian(mean, var):
    return np.random.normal(mean, var**(1/2))


def output_result(data, mean, var):
    print("Add data point:", data)
    print("Mean =", mean)
    print("Variance =", var)


m = float(input("m:"))
s = float(input("s:"))

n = 0
old_var = 1
old_mean = 0

print("Data point source function: N("+str(m)+", "+str(s))
print()
datapoint = generator_gaussian(m, s)

new_mean = datapoint
new_var = 0
output_result(datapoint, new_mean, new_var)

i = 1

while(True):
    datapoint = generator_gaussian(m, s)

    new_mean = (i * old_mean + datapoint)/(i+1)
    new_var = (i * (old_var + old_mean**2) + datapoint**2)/(i+1) - new_mean**2

    output_result(datapoint, new_mean, new_var)
    if abs(new_mean - old_mean) < 10e-5 and abs(new_var - old_var) < 10e-5:
        break

    old_mean = new_mean
    old_var = new_var
    i += 1
