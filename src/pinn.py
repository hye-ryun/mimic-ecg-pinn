"""
Physics-Informed Neural Network (PINN) for ECG

Input     : time t (s)
Output    : ECG voltage z(t)
Loss      : (1) data loss    = mismatch with the measured ECG
            (2) physics loss = how well the McSharry ODE is satisfied (dz/dt via autograd)
Trainable : network weights + physical parameters (a_i, b_i, theta_i, z0)  <- inverse problem
"""
import numpy as np
import torch
import torch.nn as nn

from physics import DEFAULT_A, DEFAULT_B, DEFAULT_THETA, WAVES


def torch_wrap(x):
    return torch.remainder(x + np.pi, 2 * np.pi) - np.pi


class ECGPINN(nn.Module):
    def __init__(self, omega, t_r, n_harmonics=12, hidden=64, a_init=None):
        super().__init__()
        self.omega, self.t_r = float(omega), float(t_r)
        self.k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)

        # Neural network: sin/cos features of the phase -> z
        self.net = nn.Sequential(
            nn.Linear(2 * n_harmonics, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Trainable 'physical' parameters (physiologically interpretable)
        a0 = DEFAULT_A if a_init is None else a_init
        self.a = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        self.log_b = nn.Parameter(torch.tensor(np.log(DEFAULT_B), dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(DEFAULT_THETA, dtype=torch.float32))
        self.z0 = nn.Parameter(torch.zeros(1))

    def phase(self, t):
        return torch_wrap(self.omega * (t - self.t_r))

    def forward(self, t):
        th = self.phase(t)                          # (N,1)
        feats = torch.cat([torch.sin(self.k * th), torch.cos(self.k * th)], dim=1)
        return self.net(feats)

    def physics_residual(self, t):
        """McSharry ODE residual: dz/dt - f(z, theta). Closer to 0 means the physics is better satisfied."""
        t = t.clone().requires_grad_(True)
        z = self(t)
        dz_dt = torch.autograd.grad(z, t, torch.ones_like(z), create_graph=True)[0]
        b = torch.exp(self.log_b)
        dth = torch_wrap(self.phase(t) - self.theta)            # (N,5)
        f = -(self.a * dth * self.omega * torch.exp(-dth**2 / (2 * b**2))).sum(1, keepdim=True) - (z - self.z0)
        return dz_dt - f

    def params_dict(self):
        a = self.a.detach().numpy()
        b = torch.exp(self.log_b).detach().numpy()
        th = self.theta.detach().numpy()
        return {
            w: {"a": float(a[i]), "b": float(b[i]), "theta_rad": float(th[i])}
            for i, w in enumerate(WAVES)
        } | {"z0": float(self.z0.detach())}


class BeatPINN(nn.Module):
    """
    One PINN per record, with physical parameters estimated separately for every beat.

    The phase is piecewise linear between detected R-peaks r_0 < r_1 < ... < r_{K-1}:
        Phi(t) = 2*pi * (k + (t - r_k) / RR_k)   for r_k <= t < r_{k+1}
    Beat index j = round(Phi / 2pi), local phase theta = Phi - 2*pi*j (R-peak at theta = 0),
    and omega(t) = 2*pi / RR_k is the instantaneous angular frequency.

    Network : [sin(k*theta), cos(k*theta), t_norm] -> z   (t_norm lets the shape vary slowly)
    Physics : per-beat a, b, theta (K x 5) + shared baseline z0
    """

    def __init__(self, r_peaks, t_span, n_harmonics=12, hidden=64):
        super().__init__()
        r = np.asarray(r_peaks, dtype=np.float64)
        rr = np.diff(r)
        self.n_beats = len(r)
        # Interval boundaries and lengths; extend RR to the first/last half-beat
        self.register_buffer("r", torch.tensor(r, dtype=torch.float32))
        self.register_buffer("rr", torch.tensor(np.r_[rr, rr[-1]], dtype=torch.float32))
        self.t0, self.t1 = float(t_span[0]), float(t_span[1])
        self.k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(2 * n_harmonics + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        K = self.n_beats
        self.a = nn.Parameter(torch.tensor(np.tile(DEFAULT_A, (K, 1)), dtype=torch.float32))
        self.log_b = nn.Parameter(torch.tensor(np.tile(np.log(DEFAULT_B), (K, 1)), dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(np.tile(DEFAULT_THETA, (K, 1)), dtype=torch.float32))
        self.z0 = nn.Parameter(torch.zeros(1))

    def phase(self, t):
        """Return (local phase theta, omega, beat index) for each time point. t: (N,1)."""
        tf = t.view(-1)
        # Interval index k such that r_k <= t < r_{k+1} (clamped to the first/last interval)
        k = torch.clamp(torch.bucketize(tf.detach(), self.r, right=True) - 1, 0, self.n_beats - 1)
        Phi = 2 * np.pi * (k + (tf - self.r[k]) / self.rr[k])
        j = torch.clamp(torch.round(Phi.detach() / (2 * np.pi)).long(), 0, self.n_beats - 1)
        theta = Phi - 2 * np.pi * j
        omega = 2 * np.pi / self.rr[k]
        return theta.view(-1, 1), omega.view(-1, 1), j

    def forward(self, t):
        th, _, _ = self.phase(t)
        t_norm = 2 * (t - self.t0) / (self.t1 - self.t0) - 1
        feats = torch.cat([torch.sin(self.k * th), torch.cos(self.k * th), t_norm], dim=1)
        return self.net(feats)

    def physics_residual(self, t):
        t = t.clone().requires_grad_(True)
        z = self(t)
        dz_dt = torch.autograd.grad(z, t, torch.ones_like(z), create_graph=True)[0]
        th, omega, j = self.phase(t)
        a, b, th_i = self.a[j], torch.exp(self.log_b[j]), self.theta[j]     # (N,5) each
        dth = torch_wrap(th - th_i)
        f = -(a * dth * omega * torch.exp(-dth**2 / (2 * b**2))).sum(1, keepdim=True) - (z - self.z0)
        return dz_dt - f

    def beat_params(self):
        a = self.a.detach().numpy()
        b = torch.exp(self.log_b).detach().numpy()
        th = self.theta.detach().numpy()
        return a, b, th


def train(model, t_data, z_data, t_col, epochs=4000, lam=1.0, lr=1e-3, log_every=500):
    """Minimize data loss + lam * physics loss."""
    t_d = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    z_d = torch.tensor(z_data, dtype=torch.float32).view(-1, 1)
    t_c = torch.tensor(t_col, dtype=torch.float32).view(-1, 1)
    phys_params = [model.a, model.log_b, model.theta, model.z0]
    opt = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": lr},
        {"params": phys_params, "lr": lr * 10},   # physical parameters learn faster
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss_data = ((model(t_d) - z_d) ** 2).mean()
        loss_phys = (model.physics_residual(t_c) ** 2).mean()
        loss = loss_data + lam * loss_phys
        loss.backward()
        opt.step()
        sched.step()
        history.append((loss_data.item(), loss_phys.item()))
        if ep % log_every == 0 or ep == 1:
            print(f"[{ep:5d}] data={loss_data.item():.5f}  physics={loss_phys.item():.5f}")
    return history
