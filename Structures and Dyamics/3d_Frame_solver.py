import numpy as np
import pandas as pd
from extra_functions import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQAiVvlK4HosRkGM0ZQuDVm-QdFwMwEuV1b277YvFMGsL7DvRBW9PdKyF-ZxUXCAtv_MPBNc37SHaCz/pub?output=xlsx"

# file = "StructureData.xlsx"
sheet_name = "LanderStrut1"
df = pd.read_excel(file, sheet_name=sheet_name)

nodes = df[['x', 'y', 'z']].to_numpy()
nodes = nodes[~np.isnan(nodes).any(axis=1)]


elements = df[['start', 'end']].dropna().to_numpy().astype(int) - 1  # index corrected elements

element_data = {}
element_data["Iy"] = df['Iy'].dropna().to_numpy()
element_data["Iz"] = df['Iz'].dropna().to_numpy()
element_data["A"] = df['A'].dropna().to_numpy()
element_data["E"] = df['E'].dropna().to_numpy()
element_data["G"] = df['G'].dropna().to_numpy()
element_data["J"] = df['J'].dropna().to_numpy()

fixed_nodes = df[['Fix_x', "Fix_y", "Fix_z", "Fix_rx", "Fix_ry", "Fix_rz"]].to_numpy()
fixed_nodes = fixed_nodes[~np.isnan(fixed_nodes).any(axis=1)]
fixed_nodes = fixed_nodes.astype(bool).flatten()


force_nodes = df[["Fx", "Fy", "Fz", "Mx", "My", "Mz"]].to_numpy()
force_nodes = force_nodes[~np.isnan(force_nodes).any(axis=1)]
F = force_nodes.flatten()


n_dof = len(nodes) * 6
assert(n_dof == len(F)), "Degrees of Freedom are not consistent"

K = np.zeros((n_dof, n_dof))


fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_title(f"Truss Structure Case {sheet_name[-1]}")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axis('equal')
ax.view_init(elev=33.264, azim=45)

plt_structure_3d(elements, nodes, ax, label="Original", color="-k")
# plt_structure(elements, nodes+1, ax, label="Deformed", color='-r')
# plt.show(block=True)

element_results = []

for i, (n1, n2) in enumerate(elements):
    n1 = int(n1)
    n2 = int(n2)
    x1, y1, z1 = nodes[n1]
    x2, y2, z2 = nodes[n2]

    Iy = element_data['Iy'][i]
    Iz = element_data['Iz'][i]
    A = element_data['A'][i]
    E = element_data['E'][i]
    G = element_data['G'][i]
    J = element_data['J'][i]

    L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    cx = (x2-x1)/L
    cy = (y2-y1)/L
    cz = (z2-z1)/L

    ex = np.array([cx,cy,cz])

    # Reference vector
    ref = np.array([0,0,1])

    # parallel vector check
    if np.allclose(np.abs(np.dot(ex,ref)), 1.0):
        ref = np.array([0,1,0])

    # local y-axis
    ey = ref - np.dot(ref,ex) * ex
    ey = ey/np.linalg.norm(ey)

    # Local z-axis
    ez = np.cross(ex,ey)

    # Rotation matrix
    R = np.vstack((ex,ey,ez))

    k_local = frame3d_local_stiffness(E=E, G=G, A=A, J=J, L=L, Iy=Iy, Iz=Iz)

    # Transformation matrix (12x12)
    T = np.zeros((12,12))
    for j in range(4):
        T[3 * j:3 * j + 3, 3 * j:3 * j + 3] = R

    k_global = T.T @ k_local @ T

    dof = [
        6*n1, 6*n1+1, 6*n1+2, 6*n1+3, 6*n1+4, 6*n1+5,
        6*n2, 6*n2+1, 6*n2+2, 6*n2+3, 6*n2+4, 6*n2+5
    ]

    # for i in range(len(k)):
    #     for j in range(len(k)):
    #         K[dof[i], dof[j]] += k[i, j]

    K[np.ix_(dof, dof)] += k_global


# Reduced global matrix
free_dofs = ~fixed_nodes
K_reduced = K[np.ix_(free_dofs, free_dofs)]

F_reduced = F[free_dofs]

u_reduced = np.linalg.solve(K_reduced, F_reduced)

# Full displacement vector
U = np.zeros(n_dof)
U[free_dofs] = u_reduced
U_nodes = U.reshape(len(nodes), 6)

nodes_deformed = nodes + U_nodes[:, :3]

plt_structure_3d(elements, nodes_deformed, ax, label="Deformed", color="-r")
plt.savefig(f"../images/3dDeformations.png",
                   dpi=300,
                   bbox_inches="tight")
plt.show(block=True)


# Full elemental forces array
R_full = K @ U - F

reactions = np.zeros_like(R_full)
reactions[fixed_nodes] = R_full[fixed_nodes]

reaction_nodes = reactions.reshape(len(nodes), 6)


for i, (n1, n2) in enumerate(elements):
    n1 = int(n1)
    n2 = int(n2)
    x1, y1, z1 = nodes[n1]
    x2, y2, z2 = nodes[n2]

    Iy = element_data['Iy'][i]
    Iz = element_data['Iz'][i]
    A = element_data['A'][i]
    E = element_data['E'][i]
    G = element_data['G'][i]
    J = element_data['J'][i]

    L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    cx = (x2-x1)/L
    cy = (y2-y1)/L
    cz = (z2-z1)/L

    ex = np.array([cx,cy,cz])

    # Reference vector
    ref = np.array([0,0,1])

    # parallel vector check
    if np.allclose(np.abs(np.dot(ex,ref)), 1.0):
        ref = np.array([0,1,0])

    # local y-axis
    ey = ref - np.dot(ref,ex) * ex
    ey = ey/np.linalg.norm(ey)

    # Local z-axis
    ez = np.cross(ex,ey)

    # Rotation matrix
    R = np.vstack((ex,ey,ez))

    k_local = frame3d_local_stiffness(E=E, G=G, A=A, J=J, L=L, Iy=Iy, Iz=Iz)

    # Transformation matrix (12x12)
    T = np.zeros((12,12))
    for j in range(4):
        T[3 * j:3 * j + 3, 3 * j:3 * j + 3] = R


    dof = [
        6*n1, 6*n1+1, 6*n1+2, 6*n1+3, 6*n1+4, 6*n1+5,
        6*n2, 6*n2+1, 6*n2+2, 6*n2+3, 6*n2+4, 6*n2+5
    ]

    u_global_elem = U[dof]
    u_local_elem = T @ u_global_elem

    f_local = k_local @ u_local_elem

    Fx1, Fy1, Fz1, Mx1, My1, Mz1 = f_local[0:6]
    Fx2, Fy2, Fz2, Mx2, My2, Mz2 = f_local[6:12]

    b = 0.01
    h = 0.01

    cy = h/2
    cz = b/2


    # ---- normal stress at section corners ----
    # sigma = N/A - Mz*y/Iz + My*z/Iy
    def corner_stresses(N, My, Mz, A, Iy, Iz, cy, cz):
        pts = [
            (+cy, +cz),
            (+cy, -cz),
            (-cy, +cz),
            (-cy, -cz),
        ]
        vals = []
        for y, z in pts:
            sigma = N / A - Mz * y / Iz + My * z / Iy
            vals.append(sigma)
        return vals


    stresses_end1 = corner_stresses(Fx1, My1, Mz1, A, Iy, Iz, cy, cz)
    stresses_end2 = corner_stresses(Fx2, My2, Mz2, A, Iy, Iz, cy, cz)

    sigma_max = max(stresses_end1 + stresses_end2)
    sigma_min = min(stresses_end1 + stresses_end2)
    sigma_abs_max = max(abs(sigma_max), abs(sigma_min))

    element_results.append({
        "element": i+1,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "sigma_abs_max": sigma_abs_max,
    })

element_results = pd.DataFrame(element_results)

print("Elements Stresses")
print(element_results)

sigma_abs_max_list = element_results['sigma_abs_max']
print(f"Maximum Stress: {np.max(sigma_abs_max_list)/1e6:.3f} MPa")
print(f"Maximum Deformation: {np.max(U_nodes)*1e3:3f} mm")

# print(f"Max Stress: {np.max(element_results['sigma_abs_max'])}")