import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.linalg as spla
import matplotlib as mpl
import control as ctrl
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Define the problem data...
n = 2 # 2 states
m = 1 # 1 input

# Computation of matrices P and K...
N = 5
R = 25
Pf = np.eye(2)
Q = np.eye(2)
A = [[0.9,1.5],[1.3,-0.7]]
B = [[0.5],[0.2]]
A = np.array(A)
B = np.array(B)

N = 50
P = np.zeros((n, n, N+1)) # tensor
K = np.zeros((m, n, N)) # tensor (3D array)
P[:,:,N] = Pf

P, _, K = ctrl.dare(A, B, Q, R)
K = -K
print('For 3i)')
print("K =")
print(K)
print("P =")
print(P)
