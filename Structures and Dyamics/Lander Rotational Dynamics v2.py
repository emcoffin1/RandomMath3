import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

# --- constants ---
g = 386.4  # in/s^2

# --- given ---
W = 400  # lbf
m = W / g  # slugs

I_cg = np.array([
    [18941.672, 1088.376, 63.619],
    [1088.376, 10548.603, -73.062],
    [63.619, -73.062, 19320.02]
])  # slug*in^2

cg = np.array([-0.28, 0.269, 10.118 + 8])

leg1 = np.array([-9.843, 17.048, -8+8])
leg2 = np.array([-19.682, 0, -8+8])
leg3 = np.array([-9.843, -17.048, -8+8])

legs = [leg1, leg2, leg3]

# --- helper function ---
def tipping_velocity(legA, legB, cg, I_cg, m, ax):
    # axis unit vector
    e = (legB - legA) / np.linalg.norm(legB - legA)

    # vector from axis to CG
    r = cg - legA

    # parallel axis theorem
    I_O = I_cg + m * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

    # inertia about axis
    I_axis = e.T @ I_O @ e

    # perpendicular distance from CG to axis
    r_parallel = np.dot(r, e) * e
    r_perp = r - r_parallel
    d = np.linalg.norm(r_perp)

    # vertical height of CG above ground
    h = cg[2] - legA[2]

    # potential barrier
    dh = d - h

    # if dh <= 0 → already unstable (CG outside polygon)
    if dh <= 0:
        return 0.0, I_axis, d, h, dh

    # critical velocity
    v = np.sqrt((2 * I_axis * g * dh) / (m * h**2))

    ax.plot([r[0],r_parallel[0]],[r[1],r_parallel[1]],[r[2],r_parallel[2]], label='r_par')
    # ax.plot([],[],[], label='')

    return v, I_axis, d, h, dh


# --- evaluate all edges ---
edges = [
    (leg1, leg2, "leg1-leg2"),
    # (leg2, leg3, "leg2-leg3"),
    # (leg3, leg1, "leg3-leg1")
]

results = []

for A, B, name in edges:
    v, I_axis, d, h, dh = tipping_velocity(A, B, cg, I_cg, m, ax)
    results.append((name, v, I_axis, d, h, dh))

# --- print results ---
for name, v, I_axis, d, h, dh in results:
    print(f"\nEdge: {name}")
    print(f"  I_axis = {I_axis:.3f} slug*in^2")
    print(f"  d      = {d:.3f} in")
    print(f"  h      = {h:.3f} in")
    print(f"  d-h    = {dh:.3f} in")
    print(f"  v_crit = {v:.3f} in/s")

# --- worst case ---
worst = min(results, key=lambda x: x[1])

print("\n--- Worst Case ---")
print(f"Edge: {worst[0]}")
print(f"Minimum tipping velocity: {worst[1]:.3f} in/s")
print(f"Minimum tipping velocity: {worst[1]*0.0254:.3f} m/s")

ax.plot([cg[0], leg1[0]], [cg[1], leg1[1]], [cg[2], leg1[2]], label='rA')
ax.plot([cg[0], leg2[0]], [cg[1], leg2[1]], [cg[2], leg2[2]], label='rB')
plt.legend()
plt.show()


