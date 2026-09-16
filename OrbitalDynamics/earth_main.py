import numpy as np
from astropy.constants import G, M_earth as m_E, R_earth as r_E, M_sun as m_S, R_sun as r_S
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# -- Constants --
G = G.to_value("km3 / (kg s2)")  # km^3/(kg s^2)


# name, mass (kg), semimajor axis (km), eccentricity, plot color, radius (km), is_satellite
# Orbital elements are Sun-relative, except the Moon, which is Earth-relative.
# The first body is the animation center; reorder rows to follow another body.
planets = [ 
    ("Earth",   m_E.value, 149.6e6, 0.0167, "blue", r_E.to_value("km"), False),
    ("Moon",   7.34767309e22, 0.3844e6, 0.0549, "lightgray", 1737.4, False),
    ("Sat1", 1000, (23.9345*3600*np.sqrt(398600.4418)/(2*np.pi))**(2/3), 0, "green", 0.001, True),  # Geostationary satellite; assumed 1 m radius,
    ("Sat2", 2000, 400000, 0.75, "purple", 0.001, True),  
]
mu_S = G * m_S.value  # km^3/s^2


# -- Time Setup --
dt = 3600 * 2
t_max = 365 * 1 * dt
t = np.arange(0, t_max+dt, dt)
N = len(t)

# -- Initial Conditions --
# Each state stores [mass, x, y, vx, vy] in kg, km, and km/s.
states_by_name = {}

# Init planets, skip the moon
for name, mass, a, e, color, radius_km, is_satellite in planets:
    if name == "Moon" or is_satellite:
        continue  # Initialize after Earth, regardless of the list order.
    rp = a * (1 - e)
    h = np.sqrt(a * mu_S * (1 - e**2))
    vp = h / rp
    states_by_name[name] = np.array([mass, rp, 0, 0, vp])


# Init moon, planets already handled
for name, mass, a, e, color, radius_km, is_satellite in planets:
    if name != "Moon" and not is_satellite:
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

radii_km = np.array([planet[5] for planet in planets])
if view_radius_km is None:
    half_span = 1.1 * max(
        (np.abs(x_positions) + radii_km[:, None]).max(),
        (np.abs(y_positions) + radii_km[:, None]).max(),
        1.0,
    )
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

trail_duration = 3 * 86400  # seconds of history shown behind each body
trails = []
markers = []
# Circles use kilometers in data coordinates, so body sizes follow the zoom.
satellite_flags = [planet[6] for planet in planets]
for name, color, radius_km, is_satellite in zip(names, colors, radii_km, satellite_flags):
    if is_satellite:
        marker, = ax.plot([], [], marker="x", linestyle="none", color=color,
                          markersize=6, markeredgewidth=1.5, zorder=3)
    else:
        marker = Circle((0, 0), radius=4*radius_km, facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(marker)
    markers.append(marker)
    trail, = ax.plot([], [], linestyle="--", color=color, linewidth=1, alpha=0.7)
    trails.append(trail)

earth_index = names.index("Earth")
earth_rotation_period = 23.9345 * 3600  # sidereal day in seconds
# Follow the displayed Earth radius, including any visual size scaling.
earth_rotation_line, = ax.plot([], [], color="white", linewidth=2, zorder=4)

time_text = ax.text(0.02,0.95, '', transform=ax.transAxes)

def update(frame):
    start = np.searchsorted(sol.t, sol.t[frame] - trail_duration, side="left")
    for i, marker in enumerate(markers):
        x = x_positions[i, frame]
        y = y_positions[i, frame]
        if satellite_flags[i]:
            marker.set_data([x], [y])
        else:
            marker.center = (x, y)
        trails[i].set_data(
            x_positions[i, start:frame+1],
            y_positions[i, start:frame+1],
        )
    angle = 2 * np.pi * (sol.t[frame] % earth_rotation_period) / earth_rotation_period
    earth_x = x_positions[earth_index, frame]
    earth_y = y_positions[earth_index, frame]
    display_radius = markers[earth_index].radius
    earth_rotation_line.set_data(
        [earth_x, earth_x + display_radius * np.cos(angle)],
        [earth_y, earth_y + display_radius * np.sin(angle)],
    )
    time_text.set_text(f"Day {sol.t[frame] / 86400:.1f}")
    return [*trails, *markers, earth_rotation_line, time_text]  

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=50, blit=True, repeat=True)

plt.show()