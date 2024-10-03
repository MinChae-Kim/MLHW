import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import norm

def gaussian_kernel(x, h):
    """
    Gaussian kernel function.
    
    Parameters:
    x (float or array-like): Input value(s).
    h (float): Bandwidth parameter.
    
    Returns:
    float or array-like: Kernel density estimate.
    """
    return (1 / (h * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / h) ** 2)

proc_samples = []
for i in range(1000):
    uniform_sample = np.random.uniform(0, 1, 12)
    proc_samples.append(sum([sample-0.5 for sample in uniform_sample]))

gaussian_samples = np.random.randn(1000)
proc_samples = np.array(proc_samples)

def make_folds(data, k):
    """
    Create K folds for cross-validation.
    
    Parameters:
    data (numpy array): The data to be split into K folds.
    k (int): The number of folds.
    
    Returns:
    list of tuples: Each tuple contains the training and validation sets for one fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    
    for train_index, val_index in kf.split(data):
        train_set, val_set = data[train_index], data[val_index]
        folds.append((train_set, val_set))
    
    return folds

def evaluate(input_value, train_set, kernel, bandwidth):
    """
    Evaluate the density estimate at the input value using KDE.
    
    Parameters:
    input_value (float): The point at which to evaluate the density.
    train_set (numpy array): The training set for KDE.
    kernel (function): The kernel function to use.
    bandwidth (float): The bandwidth parameter for the kernel.
    
    Returns:
    float: The density estimate at the input value.
    """
    n = len(train_set)
    density = np.sum(kernel(input_value - train_set, bandwidth)) / n 
    return density

def select_optimal_bandwidth(data, kernel, k, bandwidths):
    """
    Select the optimal bandwidth that maximizes the sum of log estimated densities over the validation set.
    
    Parameters:
    data (numpy array): The data to be used for cross-validation.
    kernel (function): The kernel function to use.
    k (int): The number of folds for cross-validation.
    bandwidths (numpy array): The range of bandwidth values to try.
    
    Returns:
    float: The optimal bandwidth value.
    """
    folds = make_folds(data, k)
    best_bandwidth = None
    best_log_likelihood = -np.inf
    epsilon = 1e-10  # Small positive value to prevent log(0)

    for h in bandwidths:
        log_likelihood = 0
        for train_set, val_set in folds:
            for val in val_set:
                density = evaluate(val, train_set, kernel, h)
                log_likelihood += np.log(density + epsilon)  # Add epsilon to prevent log(0)
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_bandwidth = h

    return best_bandwidth

k=5
bandwidths = np.linspace(0.05, 5.0, 99)
h = select_optimal_bandwidth(proc_samples, gaussian_kernel, k, bandwidths)
x = np.linspace(min(proc_samples), max(proc_samples), 1000)
y = [evaluate(xi, proc_samples, gaussian_kernel, h) for xi in x]
plt.plot(x, y, label=f'Gaussian KDE on Procedure Samples h={round(h, 2)}')
plt.hist(proc_samples, bins=20, density=True, alpha=0.5, label='Histogram of Procedure Samples')

h = select_optimal_bandwidth(gaussian_samples, gaussian_kernel, k, bandwidths)
x = np.linspace(min(gaussian_samples), max(gaussian_samples), 1000)
y = [evaluate(xi, gaussian_samples, gaussian_kernel, h) for xi in x]
plt.plot(x, y, label=f'Gaussian KDE on Randn Samples h={round(h, 2)}')
plt.hist(gaussian_samples, bins=20, density=True, alpha=0.5, label='Histogram of Randn Samples')    

x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
plt.plot(x, y, label='Standard Normal Distribution')

plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Gaussian Kernel Density Estimation')
plt.legend()
plt.show()
