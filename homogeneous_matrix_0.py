#!/bin/env python3
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 22})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# Defining variables
theta = sp.symbols('theta')
phi = sp.symbols('phi')
L = sp.symbols('L')
# x, y, z = symbols('x y z')

# Defining matrices
T = sp.Matrix([
            [sp.cos(phi) * sp.cos(theta), -sp.sin(phi), sp.cos(phi) * sp.sin(theta), L / theta * sp.cos(phi) * (1 -sp.cos(theta))], 
            [sp.sin(phi) * sp.cos(theta),  sp.cos(phi), sp.sin(phi) * sp.sin(theta), L / theta * sp.sin(phi) * (1 -sp.cos(theta))], 
            [-sp.sin(theta), 0, sp.cos(theta), L / theta * sp.sin(theta)], 
            [0, 0, 0, 1]
            ])


T_supp = sp.Matrix([
            [sp.cos(phi)**2 * (sp.cos(theta)-1) +1, sp.sin(phi) * sp.cos(phi) * (sp.cos(theta)-1), sp.cos(phi) * sp.sin(theta), L / theta * sp.cos(phi) * (1 -sp.cos(theta))], 
            [sp.sin(phi)* sp.cos(phi) * (sp.cos(theta)-1), sp.cos(phi)**2 *(1 -sp.cos(theta)) + sp.cos(theta), sp.sin(phi) * sp.sin(theta), L / theta * sp.sin(phi) * (1 -sp.cos(theta))], 
            [-sp.cos(phi) * sp.sin(theta), -sp.sin(phi) * sp.sin(theta), sp.cos(theta), L / theta * sp.sin(theta)], 
            [0, 0, 0, 1]
            ])


i = sp.symbols('i')
a = sp.symbols('a')
##for pitch and yaw
#pv_l = sp.Matrix([a * sp.cos(sp.pi/2*i +sp.pi/4), a * sp.sin(sp.pi/2*i +sp.pi/4), 0])
#for roll
pv_l = sp.Matrix([a * sp.cos(sp.pi/2*i -sp.pi/4), a * sp.sin(sp.pi/2*i -sp.pi/4), 0])


pv_l_ilist = []
for i_num in range(4):
    pv_l_i = pv_l.subs({i: i_num})
    pv_l_ilist.append(pv_l_i)


pv_h_hg = T * pv_l.col_join(sp.Matrix([1]))
pv_h = sp.Matrix(pv_h_hg[:-1])

pv_h_ilist = []
for i_num in range(4):
    pv_h_i = pv_h.subs({i: i_num})
    pv_h_ilist.append(pv_h_i)


ligament_list = []
for i_num in range(4):
    ligament = pv_h_ilist[i_num] - pv_l_ilist[i_num]
    ligament_list.append(ligament)

ligament_list.append(pv_h_ilist[1] - pv_l_ilist[0])
ligament_list.append(pv_h_ilist[0] - pv_l_ilist[1])
ligament_list.append(pv_h_ilist[3] - pv_l_ilist[2])
ligament_list.append(pv_h_ilist[2] - pv_l_ilist[3])


#display(p_parallel_list[3])
#norm = sp.sqrt(sum(comp**2 for comp in p_parallel_list[0]))
norm_list = []
for i_num in range(len(ligament_list)):
    norm = ligament_list[i_num].norm()#+50*np.random.normal(0, 0.01)
    norm_list.append(norm)


# theta の値を指定してノルムを計算
theta_vals = np.linspace(-1.0 *np.pi, 1.0 *np.pi, 100)

norm_vals_list = []
for i_num in range(len(ligament_list)):
    #Pitch rotation
    norm_vals = [norm_list[i_num].subs({L: 20, a: 5, theta: val, phi: 0.0}).evalf() for val in theta_vals]
    ##Yaw rotation
    #norm_vals = [norm_list[i_num].subs({L: 20, a: 5, theta: 0.000001, phi: val}).evalf() for val in theta_vals]


    norm_vals_list.append(norm_vals)
    #pv_l_i = pv_l.subs({L: 20, a: 5, i: i_num})

# グラフのプロット
for i_num in range(len(ligament_list)):
    linestyle = '-' if i_num < 4 else '--'  # #0-#3は実線、#4-#7は点線
    plt.plot(theta_vals, norm_vals_list[i_num], label=f'#{i_num}', linestyle=linestyle)

plt.axvline(x=0, color='black', linestyle='--')

#plt.title('Norm of the vector y as a function of theta')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel('Ligament length [mm]')

plt.xlim(-np.pi, np.pi)
plt.ylim(0, 25)
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()