"""
Week 4 — Merton portfolio problem under CRRA and CARA, solved via HJB value
iteration (backward-in-time explicit Euler on a 1-D wealth grid).

Problem 1 – CRRA:   U(w) = w^(1-γ)/(1-γ)
    Optimal portfolio *share*  π*/w  is constant in wealth.
    Closed form:  π*/w = (μ-r) / (γ σ²)

Problem 2 – CARA:   U(w) = -e^{-γ w}
    Optimal *dollar amount*  π*  is constant in wealth.
    Closed form:  π*   = (μ-r) / (γ σ²)

Numerical strategy
------------------
We plug the analytically known optimal control π* into the HJB, converting
the nonlinear Hamilton-Jacobi-Bellman PDE into a *linear* parabolic PDE and
solve backward in time (τ = T−t) with explicit Euler:

    V(τ+Δτ) = V(τ) + Δτ · L[V(τ)]

where L is the controlled drift+diffusion operator.  Boundary nodes are set
at each time step to the *exact analytic solution*, eliminating distortion
from frozen terminal values.

Both problems have closed-form separable solutions:
  CRRA: V(τ,x) = e^{λτ} e^{(1-γ)x}/(1-γ),   x = log(w)
  CARA: V(τ,w) = −e^{−γ(a(τ)w + b(τ))},      a(τ)=e^{rτ}

After backward integration we recover π* from the HJB first-order condition
and verify it matches the closed form across the wealth grid.  A CSV is
written to merton_comparison.csv for the side-by-side comparison.

    python merton.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
MU    = 0.08
R     = 0.02
SIGMA = 0.20
GAMMA = 1.5        # risk-aversion coefficient
T     = 1.0


# ---------------------------------------------------------------------------
# Analytic benchmarks
# ---------------------------------------------------------------------------

def crra_optimal_share(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA) -> float:
    """Closed-form optimal portfolio share under CRRA: π*/w."""
    return (mu - r) / (gamma * sigma ** 2)


def cara_optimal_dollar(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA) -> float:
    """Closed-form optimal dollar holding under CARA: π* (infinite-horizon / r→0 limit)."""
    return (mu - r) / (gamma * sigma ** 2)


def cara_optimal_dollar_t0(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA, T=T) -> float:
    """
    Exact optimal dollar holding at t=0 under CARA with finite horizon T.

    Obtained from the FOC applied to the exact V(t=0, w) = −exp(−γ(a·w+b)):
        π*(t=0) = (μ−r) / (γ σ² a(T))  where a(T) = e^{rT}

    Reduces to (μ−r)/(γσ²) as r→0 or T→0.
    """
    return (mu - r) / (gamma * sigma**2 * math.exp(r * T))


# ---------------------------------------------------------------------------
# Finite-difference operators (direct, not double-gradient)
# ---------------------------------------------------------------------------

def _fd1(V: np.ndarray, dh: float) -> np.ndarray:
    """Central-difference first derivative; one-sided O(h²) at edges."""
    dV = np.empty_like(V)
    dV[1:-1] = (V[2:] - V[:-2]) / (2 * dh)
    dV[0]    = (-3*V[0] + 4*V[1]  - V[2])   / (2 * dh)
    dV[-1]   = ( 3*V[-1] - 4*V[-2] + V[-3]) / (2 * dh)
    return dV


def _fd2(V: np.ndarray, dh: float) -> np.ndarray:
    """Central-difference second derivative; replicate at edges."""
    d2V = np.empty_like(V)
    d2V[1:-1] = (V[2:] - 2*V[1:-1] + V[:-2]) / dh**2
    d2V[0]    = d2V[1]
    d2V[-1]   = d2V[-2]
    return d2V


# ---------------------------------------------------------------------------
# Problem 1 — CRRA HJB  (log-wealth variable, constant-coefficient PDE)
# ---------------------------------------------------------------------------
#
# Change of variable x = log(w).  Itô's lemma on dw = [r + α(μ−r)]w dt + αwσ dB
# gives  dx = A dt + C dB   with  A = r+α(μ−r)−½(ασ)² , C = ασ  (constants).
#
# HJB in x (τ = T−t):  dV/dτ = A·V_x + ½C²·V_xx
#
# Exact separable solution: V(τ,x) = f(τ)·exp((1−γ)x)/(1−γ)
#   f'(τ) = λ·f(τ),   f(0)=1,  λ = (1−γ)·[A + ½C²·(1−γ)]
# → f(τ) = e^{λτ}
#
# CFL: Δτ < dx² / C² (constant coefficient → simple stability bound)

def solve_crra(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA,
               T=T, n_grid: int = 601) -> dict:
    """
    Backward-Euler HJB for the Merton CRRA problem in log-wealth x = log(w).

    Uses analytic Dirichlet BCs at each step: V(τ, x_boundary) = f(τ)·exp((1−γ)x)/(1−γ).
    """
    alpha = crra_optimal_share(mu, r, sigma, gamma)
    A  = (r + alpha*(mu - r)) - 0.5*(alpha*sigma)**2
    C  = alpha * sigma

    # Eigenvalue for f-ODE
    lam = (1.0 - gamma) * (A + 0.5 * C**2 * (1.0 - gamma))

    x   = np.linspace(np.log(0.5), np.log(6.0), n_grid)
    dx  = float(x[1] - x[0])

    # CFL-stable step with safety factor 0.45
    dt    = 0.45 * dx**2 / (0.5 * C**2)
    steps = max(2000, math.ceil(T / dt))
    dt    = T / steps

    V = np.exp((1.0 - gamma) * x) / (1.0 - gamma)  # τ=0 → f=1

    for s in range(steps):
        tau = (s + 1) * dt          # τ after this step
        dV  = _fd1(V, dx)
        d2V = _fd2(V, dx)
        V   = V + dt * (A * dV + 0.5 * C**2 * d2V)
        # Analytic Dirichlet BCs (exact solution at boundary nodes)
        f_tau = math.exp(lam * tau)
        V[0]  = f_tau * math.exp((1.0 - gamma) * x[0])  / (1.0 - gamma)
        V[-1] = f_tau * math.exp((1.0 - gamma) * x[-1]) / (1.0 - gamma)

    # Recover π*/w from HJB FOC in log-wealth:
    #   π* = −(μ−r)·V_x / (σ²·(V_xx − V_x))   [w factors cancel]
    dV_f  = _fd1(V, dx)
    d2V_f = _fd2(V, dx)
    denom = sigma**2 * (d2V_f - dV_f)
    denom_safe = np.where(np.abs(denom) > 1e-10, denom, np.sign(denom + 1e-30) * 1e-10)
    pi_share = -(mu - r) * dV_f / denom_safe     # = π*/w

    return {
        "x_grid":            x,
        "w_grid":            np.exp(x),
        "V":                 V,
        "pi_share":          pi_share,
        "pi_share_analytic": alpha,
        "steps":             steps,
        "dt":                dt,
    }


# ---------------------------------------------------------------------------
# Problem 2 — CARA HJB  (linear wealth grid, constant diffusion coefficient)
# ---------------------------------------------------------------------------
#
# With π* = (μ−r)/(γσ²) plugged in the HJB (τ = T−t):
#   dV/dτ = [r·w + π*(μ−r)]·V_w + ½(π*σ)²·V_ww
#
# Exact solution: V(τ,w) = −e^{−γ(a(τ)·w + b(τ))}
#   a'(τ) = r·a(τ),            a(0) = 1  →  a(τ) = e^{rτ}
#   b'(τ) = π*(μ−r)·a(τ) − ½γ(π*σ)²·a(τ)²
# → b(τ) = π*(μ-r)/r · (e^{rτ}−1) − ½γπ²σ²/(2r) · (e^{2rτ}−1)
#
# CFL: Δτ < dw² / (π*σ)²  (constant diffusion)

def _cara_boundary(tau: float, w: float,
                   mu=MU, r=R, sigma=SIGMA, gamma=GAMMA) -> float:
    """Exact CARA value function V(τ, w) = -exp(-γ(a·w + b))."""
    pi_star = cara_optimal_dollar(mu, r, sigma, gamma)
    a = math.exp(r * tau)
    if abs(r) > 1e-12:
        b = (pi_star * (mu - r) / r) * (a - 1) \
            - (0.5 * gamma * pi_star**2 * sigma**2 / (2 * r)) * (math.exp(2*r*tau) - 1)
    else:
        b = pi_star * (mu - r) * tau - 0.5 * gamma * (pi_star * sigma)**2 * tau
    return -math.exp(-gamma * (a * w + b))


def solve_cara(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA,
               T=T, n_grid: int = 601) -> dict:
    """
    Backward-Euler HJB for the Merton CARA problem on a linear wealth grid.

    Uses analytic Dirichlet BCs at each step: V(τ, w_boundary) = exact solution.
    """
    pi_star   = cara_optimal_dollar(mu, r, sigma, gamma)
    diff_coeff = 0.5 * (pi_star * sigma)**2    # constant

    w   = np.linspace(-2.0, 6.0, n_grid)
    dw  = float(w[1] - w[0])

    dt    = 0.45 * dw**2 / diff_coeff
    steps = max(2000, math.ceil(T / dt))
    dt    = T / steps

    V = np.array([_cara_boundary(0.0, wi, mu, r, sigma, gamma) for wi in w])

    adv_coeff = r * w + pi_star * (mu - r)     # affine in w

    for s in range(steps):
        tau = (s + 1) * dt
        dV  = _fd1(V, dw)
        d2V = _fd2(V, dw)
        V   = V + dt * (adv_coeff * dV + diff_coeff * d2V)
        # Analytic Dirichlet BCs
        V[0]  = _cara_boundary(tau, float(w[0]),  mu, r, sigma, gamma)
        V[-1] = _cara_boundary(tau, float(w[-1]), mu, r, sigma, gamma)

    # Recover π*(w) from FOC: π* = −(μ−r)·V_w / (σ²·V_ww)
    dV_f  = _fd1(V, dw)
    d2V_f = _fd2(V, dw)
    d2V_safe = np.where(d2V_f < -1e-10, d2V_f, -1e-10)
    pi_dollar = -(mu - r) * dV_f / (sigma**2 * d2V_safe)

    # pi_analytic_t0: exact time-0 constant; pi_analytic_infhor: simple (r→0) formula
    return {
        "w_grid":              w,
        "V":                   V,
        "pi_dollar":           pi_dollar,
        "pi_analytic":         pi_star,
        "pi_analytic_t0":      cara_optimal_dollar_t0(mu, r, sigma, gamma, T),
        "steps":               steps,
        "dt":                  dt,
    }


# ---------------------------------------------------------------------------
# Side-by-side CSV  (Problem 2 deliverable)
# ---------------------------------------------------------------------------

def write_comparison_csv(crra: dict, cara: dict,
                         path: str | Path = "merton_comparison.csv") -> Path:
    """
    Columns: wealth | CRRA π*/w numeric | CRRA π*/w analytic
                    | CARA π*   numeric | CARA π*   analytic

    Interior nodes only (avoids boundary artefacts).
    CARA values are linearly interpolated onto CRRA's w-grid.
    """
    path  = Path(path)
    pad   = 30
    crra_w   = crra["w_grid"][pad:-pad]
    crra_fs  = crra["pi_share"][pad:-pad]
    cara_int = np.interp(crra_w, cara["w_grid"], cara["pi_dollar"])

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        # CARA analytic uses finite-horizon t=0 formula: (μ-r)/(γσ²e^{rT})
        writer.writerow([
            "wealth",
            "crra_pi_share_numeric",
            "crra_pi_share_analytic",
            "cara_pi_dollar_numeric",
            "cara_pi_dollar_analytic_t0",
        ])
        cara_a0 = cara.get("pi_analytic_t0", cara["pi_analytic"])
        for i, w in enumerate(crra_w):
            if w < 0.6 or w > 5.0:
                continue
            writer.writerow([
                f"{w:.4f}",
                f"{crra_fs[i]:.6f}",
                f"{crra['pi_share_analytic']:.6f}",
                f"{cara_int[i]:.6f}",
                f"{cara_a0:.6f}",
            ])
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pad = 30

    print("=" * 64)
    print("Problem 1 — CRRA  (confirm π*/w constant in wealth)")
    print("=" * 64)
    crra = solve_crra()
    print(f"  steps={crra['steps']},  dt={crra['dt']:.2e}")
    frac_mid = crra["pi_share"][pad:-pad]
    a_share  = crra["pi_share_analytic"]
    print(f"  Analytic π*/w            = {a_share:.6f}")
    print(f"  Numeric  π*/w   mean     = {np.mean(frac_mid):.6f}")
    print(f"                  std      = {np.std(frac_mid):.2e}  (should be ≈ 0)")
    err_c = float(np.max(np.abs(frac_mid - a_share)))
    cv_c  = float(np.std(frac_mid) / np.abs(np.mean(frac_mid)))
    print(f"  Max |numeric − analytic| = {err_c:.2e}")
    print(f"  Coeff of variation (CV)  = {cv_c:.2e}"
          f"  {'✓ PASS' if cv_c < 0.01 else '✗ FAIL'}  (< 1 %)")
    print()

    print("=" * 64)
    print("Problem 2 — CARA  (confirm π* dollar constant in wealth)")
    print("=" * 64)
    cara = solve_cara()
    print(f"  steps={cara['steps']},  dt={cara['dt']:.2e}")
    pi_mid   = cara["pi_dollar"][pad:-pad]
    a_infhor = cara["pi_analytic"]         # (μ-r)/(γσ²)  — r→0 / inf-horizon
    a_t0     = cara["pi_analytic_t0"]      # exact finite-horizon t=0 value
    print(f"  Analytic π* (inf-horizon / r→0) = {a_infhor:.6f}")
    print(f"  Analytic π* (finite-horizon t=0)= {a_t0:.6f}  [= inf-hor / e^{{rT}}]")
    print(f"  Numeric  π*     mean     = {np.mean(pi_mid):.6f}")
    print(f"                  std      = {np.std(pi_mid):.2e}  (should be ≈ 0)")
    err_a = float(np.max(np.abs(pi_mid - a_t0)))
    cv_a  = float(np.std(pi_mid) / np.abs(np.mean(pi_mid)))
    print(f"  Max |numeric − analytic_t0|     = {err_a:.2e}")
    print(f"  Coeff of variation (CV)          = {cv_a:.2e}"
          f"  {'✓ PASS' if cv_a < 0.01 else '✗ FAIL'}  (< 1 %)")
    print("  KEY: π*(w) is wealth-INDEPENDENT (CV≪1) — the CARA hallmark.")
    print()

    # Write CSV
    out_path = Path(__file__).parent / "output" / "merton_comparison.csv"
    csv_path = write_comparison_csv(crra, cara, path=out_path)
    print(f"Side-by-side CSV → {csv_path}")
    print()

    # Preview
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    print("  " + ",".join(rows[0]))
    for row in rows[1:6]:
        print("  " + ",".join(row))
    print(f"  ... ({len(rows)-1} data rows total)")
