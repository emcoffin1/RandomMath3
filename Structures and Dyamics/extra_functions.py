import matplotlib.pyplot as plt
import numpy as np

def plt_structure(elements, nodes, ax, label, color):


    first = True

    for n1, n2 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x = [x1, x2]
        y = [y1, y2]

        if first:
            ax.plot(x, y, color, label=label)
            first = False
        else:
            ax.plot(x, y, color)

    ax.legend()



def plt_structure_3d(elements, nodes, ax, label, color):

    first = True
    for n1, n2 in elements:
        n1 = int(n1)
        n2 = int(n2)
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        if first:
            ax.plot(x, y, z, color, label=label)
            first = False
        else:
            ax.plot(x, y, z, color)

    ax.legend()


def axial_rigidity(force_nodes, L, d_reshaped):
    force_nodes = force_nodes.reshape(np.shape(d_reshaped))
    avg_tip_deflection = (d_reshaped[-1][0] + d_reshaped[-2][0])/2
    print(f"Average Tip Deflection (u): {avg_tip_deflection}")
    F = force_nodes[-1][0]+force_nodes[-2][0]

    EA = F*L / avg_tip_deflection
    print(f"Axial Rigidity (EA)eq: {EA:2.3e}")


def flexural_rigidity_couple(force_nodes, L, d_reshaped, nodes):
    force_nodes = force_nodes.reshape(np.shape(d_reshaped))

    avg_tip_deflection = (d_reshaped[-1][1] + d_reshaped[-2][1]) / 2
    print(f"Average Tip Deflection (v): {avg_tip_deflection}")

    x0 = (nodes[-1][0] + nodes[-2][0]) / 2
    y0 = (nodes[-1][1] + nodes[-2][1]) / 2
    Mi = []
    for i in range(len(force_nodes)):
        if i != (0,1):
            mx = (nodes[i][0] - x0)*force_nodes[i][1]
            my = (nodes[i][1] - y0)*force_nodes[i][0]
            Mi.append(mx-my)


    M = np.sum(Mi)
    print(f"Couple for equivalency {M:2.3e} Nm")

    EI = M*L**2/(2*avg_tip_deflection)
    print(f"Flexural Rigidity (EI)eq: {EI:2.3e}")


def transverse_load(force_nodes, L, d_reshaped, EI=0):

    force_nodes = force_nodes.reshape(np.shape(d_reshaped))

    avg_tip_deflection = (d_reshaped[-1][1] + d_reshaped[-2][1]) / 2
    print(f"Average Tip Deflection (v): {avg_tip_deflection}")
    F = force_nodes[-1][1] + force_nodes[-2][1]

    if EI == 0:
        EI = F*L**3/(3*avg_tip_deflection)
        print(f"Flexural Rigidity (EI)eq: {EI:2.3e}")
    else:
        term_big = avg_tip_deflection - (F*L**3/(3*EI))
        GA = F*L / term_big
        print(f"Shear rigidity (GA)eq: {GA:2.3e}")


def format_sci(x):
    mantissa, exponent = f"{x:2.1e}".split('e')
    exponent = int(exponent)
    return f"{mantissa}e{exponent}"


def frame3d_local_stiffness(E, G, A, Iy, Iz, J, L):

    k = np.zeros((12,12))

    EA = E*A/L
    GJ = G*J/L

    EIy = E*Iy
    EIz = E*Iz

    k[0,0] = EA
    k[0,6] = -EA
    k[6,0] = -EA
    k[6,6] = EA

    k[3,3] = GJ
    k[3,9] = -GJ
    k[9,3] = -GJ
    k[9,9] = GJ

    k[1,1] = 12*EIz/L**3
    k[1,5] = 6*EIz/L**2
    k[1,7] = -12*EIz/L**3
    k[1,11] = 6*EIz/L**2

    k[5,1] = 6*EIz/L**2
    k[5,5] = 4*EIz/L
    k[5,7] = -6*EIz/L**2
    k[5,11] = 2*EIz/L

    k[7,1] = -12*EIz/L**3
    k[7,5] = -6*EIz/L**2
    k[7,7] = 12*EIz/L**3
    k[7,11] = -6*EIz/L**2

    k[11,1] = 6*EIz/L**2
    k[11,5] = 2*EIz/L
    k[11,7] = -6*EIz/L**2
    k[11,11] = 4*EIz/L

    k[2,2] = 12*EIy/L**3
    k[2,4] = -6*EIy/L**2
    k[2,8] = -12*EIy/L**3
    k[2,10] = -6*EIy/L**2

    k[4,2] = -6*EIy/L**2
    k[4,4] = 4*EIy/L
    k[4,8] = 6*EIy/L**2
    k[4,10] = 2*EIy/L

    k[8,2] = -12*EIy/L**3
    k[8,4] = 6*EIy/L**2
    k[8,8] = 12*EIy/L**3
    k[8,10] = 6*EIy/L**2

    k[10,2] = -6*EIy/L**2
    k[10,4] = 2*EIy/L
    k[10,8] = 6*EIy/L**2
    k[10,10] = 4*EIy/L

    return k

