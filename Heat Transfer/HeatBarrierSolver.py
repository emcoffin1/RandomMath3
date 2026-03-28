import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

k = 16.3
lambd = 5.6703e-8
emis1 = 0.5
emis2 = 0.3

h1 = 100
h3 = 20
h4 = 20

Thg = 300
Ttank = 305 * 1.05
Teng = 60+273


Tmelt = 1375+273
t = np.linspace(0.001, 0.1, 1000)

# x0 = [700, 700, 1e6] # Initial guess
# def residuals(x):
#     Twh, Twc, q = x
#     A = emis1*lambd*(Thg**4 - Twh**4) - q
#     B = (k/t)*(Twh - Twc) - q
#     C = emis3*lambd*(Twc**4 - Ttank**4) - q
#     return [A, B, C]


x0 = [700, 700, 350] # Initial guess
def residuals(x, t):
    Twh, Twc, Tcg = x
    A = h1*(Thg - Twh) + emis1*lambd*(Teng**4 - Twh**4) - (k/t)*(Twh - Twc)
    B = (k/t)*(Twh - Twc) - h3*(Twc-Tcg) - emis2*lambd*(Twc**4 - Ttank**4)
    C = h3*(Twc - Tcg) - h4*(Tcg-Ttank)
    return [A,B,C]





C = np.zeros((len(t), 3))
for ind,i in enumerate(t):
    sol = fsolve(residuals, x0, args=(i,))
    C[ind, :] = sol

fig, ax = plt.subplots(layout='constrained')
ax.plot(t, C[:,1], label="Cold Wall", color="b")
ax.plot(t, C[:,0], label="Hot Wall", color="g")
ax.plot(t, C[:,2], label="Cold Gas", color='y')
ax.set_xlabel("Thickness [m]")
ax.set_ylabel("Temperature [K]")
ax.set_title("Heat Barrier Maximum Temp (Steady State)")
# ax.axhline(Tmelt, color="r", label='Melting Point')
ax.legend(bbox_to_anchor=(1.25, 0.5), loc="center right")
ax.grid(which='major', linestyle='-')
ax.grid(which='minor', linestyle=':', alpha=0.65)
ax.minorticks_on()
plt.show()