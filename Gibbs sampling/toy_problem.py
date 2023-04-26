import numpy as np


def joint_distribution(x, y):
    return np.exp(-(x**2 + y**2 - 2*x*y)/2) * (x**2 + y**2 <= 1)

# Samples
x = np.random.uniform(-1, 1)
y = np.random.uniform(-1, 1)
print("Initial x:", x)
print("Initial y:", y)

# Run the Gibbs sampler
num_iterations = 100000
samples = []

for i in range(num_iterations):
    # Sample from the conditional distribution of X given the current value of Y
    x = np.random.normal(y, 0.5)
    while abs(x) > 1:
        x = np.random.normal(y, 0.5)
    
    # Sample from the conditional distribution of Y given the current value of X
    y = np.random.normal(x, 0.5)
    while abs(y) > 1:
        y = np.random.normal(x, 0.5)
    
    # add pair (x, y) to samples
    samples.append((x, y))
    
# Compute the mean of the samples as an estimate of the expected value of X and Y
mean_x = np.mean([sample[0] for sample in samples])
var_x = np.var([sample[0] for sample in samples])
mean_y = np.mean([sample[1] for sample in samples])
var_y = np.var([sample[1] for sample in samples])

print("Estimate mean of x: ", mean_x, ", Estimate var of x: ", var_x)
print("Estimate mean of y: ", mean_y, ", Estimate var of y: ", var_y)
