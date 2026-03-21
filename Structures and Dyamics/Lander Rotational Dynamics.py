import numpy as np

W = 400
g = 386.4
m = W/g

I = np.array([[18941.672, 1088.376, 63.619],
             [1088.376, 10548.603, -73.062],
             [63.619, -73.062, 19320.02]])

cg = np.array([-0.28, 0.269, 10.118+8])

leg1 = np.array([-9.843, 17.048, -8+8])
leg2 = np.array([-19.682, 0, -8+8])
leg3 = np.array([-9.843, -17.048, -8+8])

rA = leg1
rB = leg2

e = (rB - rA) / (np.linalg.norm(rB-rA))

r = cg - rA
I_0 = I + m*((np.dot(r,r) * np.eye(3)) - np.outer(r,r))
I_axis = e.T @ I_0 @ e

r_parallel = np.dot(r,e) * e
r_perp = r - r_parallel
d = np.linalg.norm(r_perp)

h = cg[2] - rA[2]

dh = d-h

v = np.sqrt((2* I_axis * g * dh) / (m * h**2))

print("I_axis =", I_axis)
print("d =", d)
print("h =", h)
print("dh =", dh)
print("v_crit (in/s) =", v)
print("v_crit (mph) =", v*0.0568182)
print("v_crit (m/s) =", v*0.0254)