"""
Cardiac dynamical 'physics' model (McSharry et al., 2003, IEEE TBME)

Each heartbeat is represented by five Gaussian 'events' (P, Q, R, S, T).
The phase theta(t) travels around the cardiac cycle, and z(t) is the ECG voltage.

    dz/dt = - sum_i  a_i * dtheta_i * omega * exp(-dtheta_i^2 / (2 b_i^2))  - (z - z0)

    theta_i : position of each wave (P, Q, R, S, T) in rad, with R = 0
    a_i     : amplitude of each wave
    b_i     : width of each wave
    omega   : angular frequency = 2*pi / RR interval (s)

The PINN uses this equation as its 'physics loss' and estimates a_i, b_i, theta_i from real ECGs.
"""
import numpy as np
from scipy.integrate import solve_ivp

WAVES = ["P", "Q", "R", "S", "T"]

# Default values from the McSharry paper
DEFAULT_THETA = np.array([-np.pi / 3, -np.pi / 12, 0.0, np.pi / 12, np.pi / 2])
DEFAULT_A = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
DEFAULT_B = np.array([0.25, 0.1, 0.1, 0.1, 0.4])


def wrap(x):
    """Wrap an angle into the range (-pi, pi] (works for numpy and torch)."""
    pi = np.pi
    return (x + pi) % (2 * pi) - pi


def phase(t, omega, t_r):
    """Phase at time t, with theta = 0 at t_r (time of the first R-peak)."""
    return wrap(omega * (t - t_r))


def simulate(t, omega, t_r=0.0, a=DEFAULT_A, b=DEFAULT_B, theta=DEFAULT_THETA, z0=0.0):
    """Generate a synthetic ECG from the physics model (numerical ODE integration)."""
    def rhs(tt, z):
        dth = wrap(phase(tt, omega, t_r) - theta)
        return [-np.sum(a * dth * omega * np.exp(-dth**2 / (2 * b**2))) - (z[0] - z0)]

    sol = solve_ivp(rhs, (t[0], t[-1]), [z0], t_eval=t, max_step=1e-3, rtol=1e-6)
    return sol.y[0]
