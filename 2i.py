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

# Computation of matrices P and K…
# Setting Variables
N = 5
R = 25
Pf = np.eye(2)
Q = np.eye(2)
A = [[0.9,1.5],[1.3,-0.7]]
B = [[0.5],[0.2]]
A = np.array(A)
B = np.array(B)

N = 5
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
  PN11.append(Pcurr[0,0])
  PN12.append(Pcurr[0,1])
  PN22.append(Pcurr[1,1])
  i_values.append(i)

N_values = []
loop=0
while loop<N:
  N_values.append(N-i_values[loop])
  loop += 1

#print('For 2i)\n','N =',N,'\nPN11',PN11,'PN12',PN12,'PN22',PN22,'\n')
Ploop = 0
while Ploop <= 4:
  print('P',Ploop+1,'=',[[[P[0,0,Ploop]],[P[0,1,Ploop]]],[[P[1,0,Ploop]],[[P[1,1,Ploop]]]]])
  Ploop += 1
#Graph for 2i
plt.plot(N_values,P[0,0,0:5],N_values,P[1,0,0:5],N_values,P[1,1,0:5])
plt.legend(['PN 1,1','PN 1,2','PN 2,2'])
plt.ylabel('N')
plt.ylabel('P')
plt.savefig('2i) P against N')
plt.show()
exit()
