"""
심장 전기생리 '물리 모델' (McSharry et al., 2003, IEEE TBME)

ECG 한 박동을 P, Q, R, S, T 다섯 개의 가우시안 '사건'으로 표현하는 동역학 모델입니다.
위상 theta(t)가 심장 주기를 따라 돌고, z(t)가 ECG 전압입니다.

    dz/dt = - sum_i  a_i * dtheta_i * omega * exp(-dtheta_i^2 / (2 b_i^2))  - (z - z0)

    theta_i : 각 파형(P,Q,R,S,T)의 위치 (rad, R=0)
    a_i     : 각 파형의 크기
    b_i     : 각 파형의 폭
    omega   : 각속도 = 2*pi / RR간격(초)

PINN은 이 식을 '물리 손실'로 사용해서, 실제 ECG로부터 a_i, b_i, theta_i를 추정합니다.
"""
import numpy as np
from scipy.integrate import solve_ivp

WAVES = ["P", "Q", "R", "S", "T"]

# McSharry 논문의 기본값
DEFAULT_THETA = np.array([-np.pi / 3, -np.pi / 12, 0.0, np.pi / 12, np.pi / 2])
DEFAULT_A = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
DEFAULT_B = np.array([0.25, 0.1, 0.1, 0.1, 0.4])


def wrap(x):
    """각도를 (-pi, pi] 범위로 감기 (numpy / torch 둘 다 동작)."""
    pi = np.pi
    return (x + pi) % (2 * pi) - pi


def phase(t, omega, t_r):
    """시간 t에서의 위상. t_r(첫 R-peak 시각)에서 theta=0."""
    return wrap(omega * (t - t_r))


def simulate(t, omega, t_r=0.0, a=DEFAULT_A, b=DEFAULT_B, theta=DEFAULT_THETA, z0=0.0):
    """물리 모델로 합성 ECG 생성 (ODE 수치적분)."""
    def rhs(tt, z):
        dth = wrap(phase(tt, omega, t_r) - theta)
        return [-np.sum(a * dth * omega * np.exp(-dth**2 / (2 * b**2))) - (z[0] - z0)]

    sol = solve_ivp(rhs, (t[0], t[-1]), [z0], t_eval=t, max_step=1e-3, rtol=1e-6)
    return sol.y[0]
