import numpy as np
from astropy.constants import G, M_earth as m_E, R_earth as r_E, M_sun as m_S, R_sun as r_S
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -- Constants --
G = G.to_value("km3 / (kg s2)")  # km^3/(kg s^2)


# name, mass (kg), semimajor axis (km), eccentricity, plot color
# Orbital elements are Sun-relative, except the Moon, which is Earth-relative.
# The first body is the animation center; reorder rows to follow another body.
planets = [ 
    ("Sun", m_S.value, 0.0, 0.0, "orange"),
    ("Saturn",  5.6832e26, 1432.040e6, 0.0520, "darkkhaki"),
    ("Earth",   m_E.value, 149.6e6, 0.0167, "blue"),
    ("Mercury", 3.3010e23, 57.909e6, 0.2056, "gray"),
    ("Venus",   4.8673e24, 108.210e6, 0.0068, "goldenrod"),
    ("Moon",   7.34767309e22, 0.3844e6, 0.0549, "lightgray"),
    ("Mars",    6.419e23, 227.9e6, 0.0935, "red"),
    ("Sat1", 1000, (23.9345*3600*np.sqrt(398600.4418)/(2*np.pi))**(2/3), 0, "green"),  # Geostationary satellite
    ("Jupiter", 1.89813e27, 778.479e6, 0.0487, "saddlebrown"),
    ("Uranus",  8.6811e25, 2867.040e6, 0.0469, "cyan"),
    ("Neptune", 1.02409e26, 4514.950e6, 0.0097, "purple")
]
mu_S = G * m_S.value  # km^3/s^2


# -- Time Setup --
dt = 86400
t_max = 365 * 3 * dt
t = np.arange(0, t_max+dt, dt)
N = len(t)

# -- Initial Conditions --
# Each state stores [mass, x, y, vx, vy] in kg, km, and km/s.
sun_mass = next(mass for name, mass, *_ in planets if name == "Sun")
X_S = np.array([sun_mass, 0, 0, 0, 0])
states_by_name = {"Sun": X_S}

# Init planets, skip the moon
for name, mass, a, e, color in planets:
    if name in ("Sun", "Moon", "Sat1"):
        continue  # Initialize after Earth, regardless of the list order.
    rp = a * (1 - e)
    h = np.sqrt(a * mu_S * (1 - e**2))
    vp = h / rp
    states_by_name[name] = np.array([mass, rp, 0, 0, vp])


# Init moon, planets already handled
for name, mass, a, e, color in planets:
    if name not in ("Moon", "Sat1"):
        continue
    earth = states_by_name["Earth"]
    # Start at lunar perigee, outward from Earth, in a prograde orbit.
    rp = a * (1 - e)
    mu_earth_moon = G * (earth[0] + mass)
    vp = np.sqrt(mu_earth_moon * (1 + e) / rp)
    moon = np.array([mass, rp, 0, 0, vp])
    moon[1:] += earth[1:]  # Convert both position and velocity to the shared frame.
    states_by_name[name] = moon


# Preserve the list order used by the animation's labels and colors.
initial_states = [states_by_name[name] for name, *_ in planets]

X = np.concatenate(initial_states)
N_i = 5


# -- Dynamics --
def dynamics(t, X):
    N = int(len(X) // N_i)
    states = X.reshape(N,N_i)

    derivatives = np.zeros_like(states)
    for i, (m,x,y,vx,vy) in enumerate(states):
        xi_new = vx
        yi_new = vy
        ax = 0
        ay = 0
        for j, (mj,xj,yj,vxj,vyj) in enumerate(states):
            if i == j:
                continue
            dx = x - xj
            dy = y - yj
            r = np.sqrt(dx**2 + dy**2)
            ax += -G*mj* dx / r**3
            ay += -G*mj* dy / r**3
        derivatives[i] = [0, xi_new, yi_new, ax, ay]
    
    return derivatives.ravel()



sol = solve_ivp(
    fun=dynamics,
    t_span=[0, t_max],
    y0=X,
    method='RK45',
    t_eval=t,
    rtol=1e-9,
    atol=1e-9
)


if not sol.success:
    raise RuntimeError(sol.message)




# -- Animation --
center_name = planets[0][0]
# None fits all bodies. For Earth/Moon viewing, put Earth first and try 500000.
view_radius_km = None  # Distance from the center to each edge of the view.(none for all)
fig, ax = plt.subplots()

# Subtract the center body's position at each timestamp, including for trails.
# Gravity is still integrated in the original shared inertial frame.
x_positions = sol.y[1::N_i] - sol.y[1]
y_positions = sol.y[2::N_i] - sol.y[2]

if view_radius_km is None:
    half_span = 1.1 * max(np.abs(x_positions).max(), np.abs(y_positions).max(), 1.0)
else:
    if view_radius_km <= 0:
        raise ValueError("view_radius_km must be positive or None")
    half_span = view_radius_km
ax.set_xlim(-half_span, half_span)
ax.set_ylim(-half_span, half_span)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(f"x relative to {center_name} (km)")
ax.set_ylabel(f"y relative to {center_name} (km)")
ax.set_title(f"Orbits relative to {center_name}")

names = [planet[0] for planet in planets]
colors = [planet[4] for planet in planets]

trail_duration = 75 * 86400  # seconds of history shown behind each body
trails = []
markers = []
for name, color in zip(names, colors):
    marker, = ax.plot([], [], marker='o', color=color, linestyle="none", markersize=8 if name=="Sun" else 4)
    markers.append(marker)
    trail, = ax.plot([], [], linestyle="--", color=color, linewidth=1, alpha=0.7)
    trails.append(trail)

time_text = ax.text(0.02,0.95, '', transform=ax.transAxes)

def update(frame):
    start = np.searchsorted(sol.t, sol.t[frame] - trail_duration, side="left")
    for i, marker in enumerate(markers):
        x = x_positions[i, frame]
        y = y_positions[i, frame]
        marker.set_data([x], [y])
        trails[i].set_data(
            x_positions[i, start:frame+1],
            y_positions[i, start:frame+1],
        )
    time_text.set_text(f"Day {sol.t[frame] / 86400:.1f}")
    return [*trails, *markers, time_text]  

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=0.02, blit=True, repeat=True)

plt.show()