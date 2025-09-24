import numpy as np
import scipy.linalg as sla
from utils.waveguide_main_genetic import executeCore

## UTILS FUNCTIONS ##

def change_of_basis(n, modes, inte, steps, len_mat):
    U = []
    for j in range(n):  
        U.append([np.trapz(modes[j][steps//2-len_mat//2+inte[i]:steps//2-len_mat//2+inte[i+1]]) for i in range(len(inte) - 1)])
    return np.array(U)

def expand_lambda(lambdas):
    lambdas_expanded = []
    for i in range(len(lambdas)):
        lambdas_expanded.append(np.exp(2 * lambdas[i] * (-1) ** i))  # x^2
        lambdas_expanded.append(np.exp(-2 * lambdas[i] * (-1) ** i)) # p^2
    return lambdas_expanded

def expand_matrix(U):
    n = U.shape[0]
    expanded_matrix = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            expanded_matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2] = np.diag([U[i, j], U[i, j]])
    return expanded_matrix
