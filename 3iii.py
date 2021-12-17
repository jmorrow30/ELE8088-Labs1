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

A = np.array([[.95, 1.5],[1.3, -0.7]])
B = np.array([[0.5], [0.2]])
Q = np.eye(2)
R = 25
Pf = np.eye(2)
N = 50

# Construct Qbar and Rbar
Qbar = np.kron(np.eye(N-1), Q)
Qbar = sp.linalg.block_diag(Qbar, Pf)
Rbar = np.kron(np.eye(N), R)

# Construct Abar
Atemp = Abar = A
for i in range(N-1):
  Atemp = A @ Atemp
  Abar = np.vstack((Abar, Atemp))

# Construct Bbar
Bbar = np.hstack((B, np.zeros((n, (N-1)*m))))
for i in range(N-1):
  Btemp = B
  B_current_line = B
  for j in range(i+1):
    Btemp = A @ Btemp
    B_current_line = np.hstack((Btemp, B_current_line))
  B_current_line = np.hstack((B_current_line, np.zeros((n, (N-j-2)*m))))
  Bbar = np.vstack((Bbar, B_current_line))


N = 50
P = np.zeros((n, n, N+1)) # tensor
K = np.zeros((m, n, N)) # tensor (3D array)
P[:,:,N] = Pf

xmin = np.array([-1,-1])
xmax = np.array([1, 1])
umin = np.array([-1])
umax = np.array([1])

# Problem statement
# -----------------------------
x0 = cp.Parameter(n) # <--- x is a parameter of the optimisation problem P_N(x)
u_seq = cp.Variable((m, N)) # <--- sequence of control actions
x_seq = cp.Variable((n, N+1))

cost = 0
constraints = [x_seq[:, 0] == x0] # x_0 = x

for t in range(N-1):
  xt_var = x_seq[:, t] # x_t
  ut_var = u_seq[:, t] # u_t
  cost += 0.5*(cp.quad_form(xt_var, Q) + R*ut_var**2)
  # dynamics, xmin <= xt <= xmax, umin <= ut <= umax
  constraints += [x_seq[:, t+1] == A@xt_var + B@ut_var, 
                  xmin <= xt_var,
                  xt_var <= xmax,
                  umin <= ut_var,
                  ut_var <= umax]


xN = x_seq[:, N-1]
cost += 0.5*cp.quad_form(xN, Q) # terminal cost
constraints += [xmin <= xN, xN <= xmax] # terminal constraints (xmin <= xN <= xmax)

# Solution
# -----------------------------
x0.value = np.array([0.02, 0.05])
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()


# Plotting of solution
# -----------------------------
x_star = x_seq.value
plt.plot(x_star.T)
plt.rcParams['font.size'] = '14'
plt.xlabel('Time, t')
plt.ylabel('States')
plt.savefig(‘States against time')
plt.cla()
plt.clf()

u_star = u_seq.value
plt.plot(u_star.T)
plt.xlabel('Time, t')
plt.ylabel('Input')
plt.savefig('Input against Time')
plt.cla()
plt.clf()


P, _, K = ctrl.dare(A, B, Q, R)
K = - K

x_star=np.array([[1],[1]])

V_star_inf = (1/2)*(x_star.T)@P@x_star

print(V_star_inf[0,0])

V_values = []
x = np.array([[1],[2]])
Acl = A + B@K
for t in range(100):
  x = Acl @ x
  V_star_inf_x = 0.5*(x.T @ P @ x)
  V_values.append(V_star_inf_x[0,0])
print(V_values)

plt.plot(V_values)
plt.xlabel('Time, t')
plt.ylabel('V inf star')
plt.savefig('V inf star against Time')
plt.cla()
plt.clf()

