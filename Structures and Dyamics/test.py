import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# --- constants ---
g = 386.4  # in/s^2

# --- given ---
W = 400  # lbf
m = W / g  # slugs

I_cg = np.array([
    [18941.672, 1088.376, 63.619],
    [1088.376, 10548.603, -73.062],
    [63.619, -73.062, 19320.02]
])

cg = np.array([-0.28, 0.269, 10.118 + 8])

leg1 = np.array([-9.843, 17.048, 0])
leg2 = np.array([-19.682, 0, 0])
leg3 = np.array([-9.843, -17.048, 0])

# --- helper function ---
def tipping_velocity(legA, legB, cg, I_cg, m, ax):
    # axis unit vector
    e = (legB - legA) / np.linalg.norm(legB - legA)

    # vector from axis point to CG
    r = cg - legA

    # parallel axis theorem
    I_O = I_cg + m * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

    # inertia about axis
    I_axis = e.T @ I_O @ e

    # projection
    r_parallel = np.dot(r, e) * e
    r_perp = r - r_parallel

    d = np.linalg.norm(r_perp)
    h = cg[2] - legA[2]
    h_vec = np.array([0, 0, h])
    dh = d - h

    # --- plotting ---

    # axis line
    ax.plot(
        [legA[0], legB[0]],
        [legA[1], legB[1]],
        [legA[2], legB[2]],
        'k-', linewidth=2, label='axis'
    )

    # r vector (blue)
    ax.quiver(
        legA[0], legA[1], legA[2],
        r[0], r[1], r[2],
        color='blue', label='r'
    )

    # r_parallel (green)
    ax.quiver(
        legA[0], legA[1], legA[2],
        r_parallel[0], r_parallel[1], r_parallel[2],
        color='green', label='r_parallel'
    )

    # r_perp (red) — start at end of r_parallel
    base = legA + r_parallel
    ax.quiver(
        base[0], base[1], base[2],
        r_perp[0], r_perp[1], r_perp[2],
        color='red', label='r_perp'
    )

    # CG point
    ax.scatter(cg[0], cg[1], cg[2], color='purple', s=50, label='CG')

    return I_axis, d, h, dh


# --- run for one edge (you can extend later) ---
I_axis, d, h, dh = tipping_velocity(leg1, leg2, cg, I_cg, m, ax)

# --- plot legs ---
for leg in [leg1, leg2, leg3]:
    ax.scatter(leg[0], leg[1], leg[2], color='black', s=40)

# --- formatting ---
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title("Tipping Geometry Visualization")

# equal aspect ratio
ax.set_box_aspect([1,1,1])

plt.legend()
plt.show()