import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.linalg as spla
import matplotlib as mpl
import control as ctrl
import matplotlib.pyplot as plt
from numpy import linalg as LA

n = 1000 # Dimension of x (n=1000)
m = 100 # Rows of A (m=100)

# Random data
x_var = cp.Variable(n)
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

#A transpose = A_T
A_T = A.T
C = A_T@b
mu_max = LA.norm(C)
print('mu max =',mu_max)
z_increment = 1
#z is mu from 0 to mu max
z = np.arange(0,mu_max,z_increment)
counter=0
sp_x = []
while counter < mu_max:
  mu = z[counter]
  counter+=1
  #Part i
  objective = cp.Minimize(0.5*cp.norm2(A@x_var - b)**2 + mu * cp.norm1(x_var))
  
  #Part ii
  #objective = cp.Minimize(0.5*cp.norm2((A@x_var*mu) - b)**2)

  problem = cp.Problem(objective)
  problem.solve()
  w = x_var.value
  non_0_x = sum(i > (10**-5) for i in w)
  sp_x_value = non_0_x/len(w)
  sp_x.append(sp_x_value)
  
print('z length=',len(z),', sp(x) length =',len(sp_x))

plt.suptitle('1i) sp(x) against mu')
plt.ylabel('sp(x)')
plt.xlabel('mu')
plt.plot(z,sp_x)
plt.savefig('sp(x) against mu')
plt.show()
