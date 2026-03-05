#from sympy import symbols, cos, sin, Matrix, pi, sqrt, eye, zeros, simplify
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = 'Times New Roman'


## Defining homogeneous matrix for ball joint model [LM2025]
# Defining variables
theta = sp.symbols('theta')
phi = sp.symbols('phi')
w = sp.symbols('w')
d = sp.symbols('d')
h = sp.symbols('h')
# x, y, z = symbols('x y z')

# Rotation vector R_y(theta) * R_z(phi)
R = sp.Matrix([
    [sp.cos(theta) * sp.cos(phi), -sp.cos(theta) * sp.sin(phi), sp.sin(theta)],
    [sp.sin(phi), sp.cos(phi), 0],
    [-sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)]
])

# translation vector
t = sp.Matrix([0, 0, 0])

# homogeneous matrix
T = R.row_join(t)
T = T.col_join(sp.Matrix([[0, 0, 0, 1]]))


#Ligament definitions
#Local coordinates
#Cruciate ligaments
p_acl_f = sp.Matrix([-d/2, 0,  0])
p_acl_t = sp.Matrix([ d/2, 0, -h])
p_pcl_f = sp.Matrix([ d/2, 0,  0])
p_pcl_t = sp.Matrix([-d/2, 0, -h])

#Collateral ligaments
p_lcl_f = sp.Matrix([12, 33,  35])
p_lcl_t = sp.Matrix([-5, 40,  -30])
p_mcl_f = sp.Matrix([12, -33,  35])
p_mcl_t = sp.Matrix([-5, -40,  -30])


p_acl_f_hg = p_acl_f.col_join(sp.Matrix([1]))
p_acl_t_hg = p_acl_t.col_join(sp.Matrix([1]))
p_pcl_f_hg = p_pcl_f.col_join(sp.Matrix([1]))
p_pcl_t_hg = p_pcl_t.col_join(sp.Matrix([1]))

p_lcl_f_hg = p_lcl_f.col_join(sp.Matrix([1]))
p_lcl_t_hg = p_lcl_t.col_join(sp.Matrix([1]))
p_mcl_f_hg = p_mcl_f.col_join(sp.Matrix([1]))
p_mcl_t_hg = p_mcl_t.col_join(sp.Matrix([1]))


P_ACL_matrix = sp.Matrix([p_acl_f_hg]).row_join(sp.Matrix([p_acl_t_hg])).row_join(sp.Matrix([p_pcl_f_hg]).row_join(sp.Matrix([p_pcl_t_hg])))
P_LCL_matrix = sp.Matrix([p_lcl_f_hg]).row_join(sp.Matrix([p_lcl_t_hg])).row_join(sp.Matrix([p_lcl_f_hg]).row_join(sp.Matrix([p_lcl_t_hg])))


#global coordinates
p_acl_t_hg = T*p_acl_t_hg
p_pcl_t_hg = T*p_pcl_t_hg
p_lcl_t_hg = T*p_lcl_t_hg
p_mcl_t_hg = T*p_mcl_t_hg


#Ligament vector computation
v_acl = p_acl_t_hg - p_acl_f_hg
v_pcl = p_pcl_t_hg - p_pcl_f_hg

v_lcl = p_lcl_t_hg - p_lcl_f_hg
v_mcl = p_mcl_t_hg - p_mcl_f_hg

L_acl = v_acl.norm()
L_pcl = v_pcl.norm()
L_lcl = v_lcl.norm()
L_mcl = v_mcl.norm()


#Plot ligament length
ligament_list = []
ligament_list.append(v_acl)
ligament_list.append(v_pcl)

# theta の値を指定してノルムを計算
theta_vals = np.linspace(0.0001, 1.0/2.0 *np.pi, 10)
phi_vals = np.linspace(-1.0/2.0 *np.pi, 1.0/2.0 *np.pi, 10)

norm_list = []
norm_list.append(L_acl)
norm_list.append(L_pcl)

Theta, Phi = np.meshgrid(theta_vals, phi_vals)

norm_vals_list = []


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for i_num in range(len(ligament_list)):
    Z = np.array([
        #[norm_list[i_num].subs({d: 20, h: 25, theta: th, phi: ph}).evalf()
        [norm_list[i_num].subs({d: 36, h: 36, theta: th, phi: ph}).evalf()
         for th in theta_vals]
        for ph in phi_vals
    ], dtype=float)
    ax.plot_surface(Theta, Phi, Z, label=f'#{i_num}', alpha=0.7)

# ax.set_xlabel(r'$\theta~\left[\mathrm{rad} \right]$', fontsize=24)
# ax.set_ylabel(r'$\phi~\left[\mathrm{rad} \right]$', fontsize=24)
# ax.set_zlabel(r'Ligament length $\mathrm{[mm]}$', fontsize=24)
ax.set_xlabel(r'Pitch angle $\left[\mathrm{rad} \right]$')
ax.set_ylabel(r'Yaw angle $\left[\mathrm{rad} \right]$')
ax.set_zlabel(r'Ligament length $\mathrm{[mm]}$')

# tick labels
ax.tick_params(axis='both', labelsize=18)
ax.tick_params(axis='z', which='major', labelsize=18)

plt.show()