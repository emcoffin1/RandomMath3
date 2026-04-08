import numpy as np
import pandas as pd
from extra_functions import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pint import UnitRegistry

unit = UnitRegistry()

# add cost function and weight funciton
file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQAiVvlK4HosRkGM0ZQuDVm-QdFwMwEuV1b277YvFMGsL7DvRBW9PdKyF-ZxUXCAtv_MPBNc37SHaCz/pub?output=xlsx"

def run(sheet_name="LanderStrut1", iterating=False, density=0.0, outer_d_input=None, inner_d_input=None):

    df = pd.read_excel(file, sheet_name=sheet_name)

    # ================== #
    # == EXTRACT DATA == #
    # ================== #

    nodes = df[['x', 'y', 'z']].to_numpy()
    nodes = nodes[~np.isnan(nodes).any(axis=1)]


    elements = df[['start', 'end']].dropna().to_numpy().astype(int) - 1  # index corrected elements

    element_data = {}
    # These must be input into the spreadsheet

    element_data["E"] = df['E'].dropna().to_numpy()
    element_data["G"] = df['G'].dropna().to_numpy()
    if outer_d_input is not None:
        outer_d = np.full((len(element_data["E"]),), outer_d_input)
        inner_d = np.full((len(element_data["E"]),), inner_d_input)

    else:
        outer_d = df["D_outer"].dropna().to_numpy()
        outer_h = df["H_outer"].dropna().to_numpy()
        inner_d = df["D_inner"].dropna().to_numpy()
        inner_h = df["H_inner"].dropna().to_numpy()

    struct = df["Shape"].dropna().to_numpy()

    element_data['Iy'] = np.zeros((len(struct)))
    element_data['Iz'] = np.zeros((len(struct)))
    element_data['A'] = np.zeros((len(struct)))
    element_data['J'] = np.zeros((len(struct)))

    # The following are computed and dependent on only geometry
    for i in range(len(struct)):
        Iy = Iz = J = A = np.nan
        if struct[i].title() == "Circle":
            try:
                Iy = Iz = np.pi/ 64 * (outer_d[i]**4 - inner_d[i]**4)
                J = Iy + Iz
                A = np.pi/4 * (outer_d[i]**2 - inner_d[i]**2)
            except Exception as e:
                print(print(np.size(outer_d)))

        if struct[i].title() == "Rectangle":
            bo = outer_d[i]
            ho = outer_h[i]
            bi = inner_d[i]
            hi = inner_h[i]

            # Bending
            Iy = 1 / 12 * (bo * ho ** 3 - bi * hi ** 3)
            Iz = 1 / 12 * (ho * bo ** 3 - hi * bi ** 3)

            # Thickness (assumes uniform wall)
            t = (bo - bi) / 2

            # Median dimensions
            bm = bo - t
            hm = ho - t

            # Torsion (thin-wall approximation)
            J = (2 * t * bm ** 2 * hm ** 2) / (bm + hm)

            # Area
            A_o = bo * ho
            A_i = bi * hi
            A = A_o - A_i

        if struct[i].title() == "Square":
            ao = outer_d[i]
            ai = inner_d[i]

            # Bending (same about both axes)
            Iy = 1 / 12 * (ao ** 4 - ai ** 4)
            Iz = Iy

            # Thickness (uniform wall)
            t = (ao - ai) / 2

            # Median dimension
            am = ao - t

            # Torsion (thin-wall approximation)
            J = t * am ** 3

            # Area
            A_o = ao**2
            A_i = ai**2
            A = A_o - A_i

        # Now update the values
        element_data["Iy"][i] = Iy
        element_data["Iz"][i] = Iz
        element_data["A"][i] = A
        element_data["J"][i] = J

    # element_data["Iy"] = df['Iy'].dropna().to_numpy()
    # element_data["Iz"] = df['Iz'].dropna().to_numpy()
    # element_data["A"] = df['A'].dropna().to_numpy()
    # element_data["J"] = df['J'].dropna().to_numpy()

    fixed_nodes = df[['Fix_x', "Fix_y", "Fix_z", "Fix_rx", "Fix_ry", "Fix_rz"]].to_numpy()
    fixed_nodes = fixed_nodes[~np.isnan(fixed_nodes).any(axis=1)]
    fixed_nodes = fixed_nodes.astype(bool).flatten()


    force_nodes = df[["Fx", "Fy", "Fz", "Mx", "My", "Mz"]].to_numpy()
    force_nodes = force_nodes[~np.isnan(force_nodes).any(axis=1)]
    F = force_nodes.flatten()


    n_dof = len(nodes) * 6
    assert(n_dof == len(F)), "Degrees of Freedom are not consistent"

    K = np.zeros((n_dof, n_dof))

    if not iterating:
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

    mass = 0.0
    if density != 0.0:
        print("="*50)
        print("COMPUTING VEHICLE WEIGHT")

        for i, (n1, n2) in enumerate(elements):
            n1 = int(n1)
            n2 = int(n2)
            x1, y1, z1 = nodes[n1]
            x2, y2, z2 = nodes[n2]

            area = element_data["A"][i]
            L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            mass += density * area * L
    mass = mass * unit.kg



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

    if not iterating:
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
        Ux = U_nodes[0]
        Uy = U_nodes[2]
        Uz = U_nodes[4]
        U_max = np.sqrt(Ux**2 + Uy**2 + Uz**2)

        element_results.append({
            "element": i+1,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "sigma_abs_max": sigma_abs_max,
            'deform_max': U_max
        })

    element_results = pd.DataFrame(element_results)

    # print("Elements Stresses")
    # print(element_results)

    sigma_abs_max_list = element_results['sigma_abs_max']
    U_maxs = list(element_results['deform_max'])


    print(f"Maximum Stress: {np.max(sigma_abs_max_list)/1e6:.3f} MPa")
    print(f"Maximum Deformation: {np.max(U_maxs)*1e3:3f} mm")

    if iterating:
        return [np.max(sigma_abs_max_list), np.max(U_maxs), mass]
    else:
        return mass


def iterate():
    thicknesses = [0.049, 0.065, 1/8]
    outer_options = np.array([1/4, 3/8, 1/2, 3/4]) * 0.0254
    inner_options1 = outer_options - 2*0.049*0.0254
    inner_options2 = outer_options - 2*0.035*0.0254
    inner_options = np.array([inner_options1, inner_options2])
    # outer_options = np.array([1/4])*0.0254
    # inner_options = np.array([0.152])*0.0254

    fig, ax = plt.subplots(2, 1, sharex=True)
    for j in range(len(inner_options)):
        results = {"stress": [],
                   "deformation": [], }

        for i in range(len(outer_options)):
            if i == 0:
                # res = run(iterating=False, outer_d_input=outer_options[i], inner_d_input=inner_options[j, i])
                res = run(iterating=True, outer_d_input=outer_options[i], inner_d_input=inner_options[j, i])
            else:
                res = run(iterating=True, outer_d_input=outer_options[i], inner_d_input=inner_options[j, i])
            results['stress'].append(res[0])
            results['deformation'].append(res[1])
            results['mass'].append(res[2])

        stresses = np.array(results["stress"]) * 1e-6
        deforms = np.array(results["deformation"]) * 1e3
        ax[0].plot(outer_options, stresses, label=f"{(outer_options[0]-inner_options[j,0])*2*0.0254*1e3:.3f}")
        ax[1].plot(outer_options, deforms, label=f"{(outer_options[0]-inner_options[j,0])*2*0.0254*1e3:.3f}")

    ax[0].set_ylabel("Stress [MPa]")
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.25', color='black', alpha=0.5)

    ax[1].set_ylabel("Deformation [mm]")
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax[1].grid(which='minor', linestyle=':', linewidth='0.25', color='black', alpha=0.5)

    fig.supxlabel("Outer Diameter [m]")
    fig.suptitle(f"Study of Different Diameters of 6061 t6 Aluminum")
    ax[0].legend(title='Thickness [m]')
    ax[1].legend(title='Thickness [m]')
    plt.show()




    fig, ax = plt.subplots()




if __name__ == "__main__":
    mass = run(iterating=False, density=2700)
    print(f"{mass.to(unit.lb):.2f}")
    # iterate()

