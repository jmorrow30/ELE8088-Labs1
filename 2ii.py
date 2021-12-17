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

PN11 = []
PN12 = []
PN22 = []
i_values = []

for i in range(N):
  Pcurr = P[:,:,N-i]
  K[:,:,N-i-1] = -sp.linalg.solve(R+B.T@Pcurr@B, B.T@Pcurr@A)
  P[:,:,N-i-1] = Q + A.T @ Pcurr @ A + A.T @ Pcurr @ B @ K[:,:,N-i-1]
  error = Pcurr - P[:,:,N-i-1]
  #PN12 = PN 21
  PN11.append(Pcurr[0,0])
  PN12.append(Pcurr[0,1])
  PN22.append(Pcurr[1,1])
  i_values.append(i)



N_values = []
loop=0
while loop<N:
  N_values.append(N-i_values[loop])
  loop += 1


print('For 2ii)\n','for N =',N,'PN11 =',PN11[0],'PN12 =',PN12[0],'PN21 =',PN12[0],'PN22 =',PN22[0],'Therefore as seen fro the graph in 2ii the values go towards the matrix P that solves DARE as N increases and are approximately equal to it at N = 50')

#Graph for 2ii
#2ii) Graph shows that the values of PN converge to a value at higher N values

plt.plot(N_values,PN11,N_values,PN12,N_values,PN22)
plt.legend(['PN 1,1','PN 1,2','PN 2,2'])
plt.xlabel('N')
plt.ylabel('P')
plt.savefig('2ii) P against N')
plt.show()
