#!/usr/bin/env python3
"""
kappa_m0_bound.py

Compute:
  • Alignment κ (consistent Selberg/Dirichlet geometry)
  • Paley–Zygmund baseline R0 (from moments or manual)
  • ρ² lower bound (control–variate formula)
  • Optional normalized-tilt improvement δ (Euler-product report + proxy)
  • Resulting M_new and even-rounded m0 bound

Also supports solving for the optimal A (by degree) that maximizes κ for a chosen bump b'(t).

USAGE
-----
python3 kappa_m0_bound.py
Edit the CONFIG section at the bottom as needed.

NOTES (matches paper §5)
------------------------
Geometry (Selberg coefficient space, modeled via Dirichlet inner products):
  Q[A]     = E[T_A^2]  (here realized as ∫_0^1 (A'(t))^2 dt)
  B[A]     = E[T_A Z_c] (here ∫_0^1 A'(t) b'(t) dt)
  ||b||^2  = ∫_0^1 (b'(t))^2 dt
  κ        = (B[A])^2 / ( Q[A] * ||b||^2 ) ∈ [0,1]

ρ² lower bound (Prop. 4.6 / eq. (8) in the paper):
  ρ² ≥ |U| · R0 · κ² · (Σ c_p/(p(p-1)))² / (Σ c_p²/p)

Tilt:
  We report EP_ratio = E[W_{2τ}]/(E[W_{τ}])^2 from the exact Euler product over p ≤ k (log-domain),
  and also provide a conservative proxy δ_proxy ≈ c0 · |U| · τ².
  Use either δ=0 (no tilt), or δ=δ_proxy (recommended), or set δ manually.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    from sympy import primerange
except Exception:
    primerange = None


# ------------------------------
# Dirichlet / Selberg geometry
# ------------------------------
def poly_deriv(coeffs: List[float]) -> np.ndarray:
    """Given A(t) = sum c_i t^i, return coefficients of A'(t): [1*c1, 2*c2, ...]."""
    return np.array([i*c for i, c in enumerate(coeffs)][1:], dtype=float)

def gram_dirichlet(dim: int) -> np.ndarray:
    """
    Gram matrix on the derivative monomial basis {t^{m-1}}_{m=1..dim}:
    ⟨t^{i-1}, t^{j-1}⟩ = ∫_0^1 t^{i+j-2} dt = 1 / (i + j - 1).
    """
    G = np.empty((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            G[i, j] = 1.0 / ((i+1) + (j+1) - 1)  # 1/(i+j+1)
    return G

def Q_of_A(coeffs: List[float]) -> float:
    """Q[A] = ∫ (A')^2 dt via Gram matrix."""
    Ader = poly_deriv(coeffs)
    if Ader.size == 0:
        return 0.0
    G = gram_dirichlet(len(Ader))
    return float(Ader @ G @ Ader)

def norm_bprime_sq(b_deriv_coeffs: List[float]) -> float:
    """||b||^2 = ∫ (b')^2 dt, with b'(t) = Σ b_j t^j."""
    s = 0.0
    L = len(b_deriv_coeffs)
    for i in range(L):
        for j in range(L):
            s += b_deriv_coeffs[i] * b_deriv_coeffs[j] / (i + j + 1.0)
    return max(s, 1e-16)

def moment_vector_bprime(b_deriv_coeffs: List[float], dim: int) -> np.ndarray:
    """h_m = ∫_0^1 t^{m-1} b'(t) dt = Σ_j b_j / (m + j),  m=1..dim."""
    h = np.zeros(dim, dtype=float)
    for m in range(1, dim+1):
        h[m-1] = sum(bj / (m + j) for j, bj in enumerate(b_deriv_coeffs))
    return h

def B_of_A_direct(coeffs: List[float], b_deriv_coeffs: List[float]) -> float:
    """Direct ∫ A'(t)b'(t) dt using monomial integrals: Σ_i Σ_j a_i b_j /(i+j)."""
    Ader = poly_deriv(coeffs)
    if Ader.size == 0:
        return 0.0
    s = 0.0
    for i, ai in enumerate(Ader, start=1):
        for j, bj in enumerate(b_deriv_coeffs, start=0):
            s += ai * bj / (i + j)
    return float(s)

def kappa_of_A(coeffs: List[float], b_deriv_coeffs: List[float]) -> float:
    """κ = B[A]^2 / ( Q[A] * ||b||^2 ), clamped to [0,1]."""
    Q = Q_of_A(coeffs)
    B = B_of_A_direct(coeffs, b_deriv_coeffs)
    Nb = norm_bprime_sq(b_deriv_coeffs)
    if Q <= 0 or Nb <= 0:
        return 0.0
    val = (B * B) / (Q * Nb)
    return max(0.0, min(1.0, val))

def optimal_kappa_and_Aprime(b_deriv_coeffs: List[float], deg_A: int) -> Tuple[float, np.ndarray]:
    """
    Maximize κ over degree-deg_A polynomials.
    In derivative basis a ∈ R^d, κ = ((a·h)^2 / ||b||^2) / (a^T G a).
    Maximizer a ∝ G^{-1} h, κ_max = (h^T G^{-1} h) / ||b||^2.
    Robust solve with tiny Tikhonov if needed.
    """
    d = deg_A
    G = gram_dirichlet(d)
    h = moment_vector_bprime(b_deriv_coeffs, d)
    Nb = norm_bprime_sq(b_deriv_coeffs)
    try:
        invGh = np.linalg.solve(G, h)
        a = invGh.copy()
    except np.linalg.LinAlgError:
        reg = 1e-12
        invGh = np.linalg.solve(G + reg*np.eye(d), h)
        a = invGh.copy()
    kappa_max = float(h @ invGh / Nb)
    kappa_max = max(0.0, min(1.0, kappa_max))
    return kappa_max, a

def integrate_ader_to_Acoeffs(a_der: np.ndarray, A0: float = 0.0) -> List[float]:
    """Given A'(t)=Σ_{m=1..d} a_m t^{m-1}, return A(t)=A0 + Σ_{m=1..d} a_m t^m / m."""
    d = len(a_der)
    coeffs = [A0] + [0.0]*d
    for m in range(1, d+1):
        coeffs[m] = a_der[m-1] / m
    return coeffs


# ------------------------------
# Prime sums and μ_k
# ------------------------------
def primes_up_to(n: int) -> List[int]:
    if n < 2:
        return []
    if primerange is not None:
        return list(primerange(2, n+1))
    # fallback simple sieve
    sieve = bytearray(b"\x01")*(n+1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5)+1):
        if sieve[p]:
            sieve[p*p : n+1 : p] = b"\x00" * (((n - p*p)//p) + 1)
    return [i for i in range(2, n+1) if sieve[i]]

def mu_k(k: int) -> float:
    """μ_k = Σ_{p≤k} 1/p."""
    return sum(1.0/p for p in primes_up_to(k))

def prime_weight(p: int, typ: str) -> float:
    if typ == "flat":
        return 1.0
    if typ == "optimized":
        return 1.0/(p-1)
    raise ValueError("weight_type must be 'flat' or 'optimized'")

def prime_sums(k: int, weight_type: str = "optimized") -> Tuple[float, float]:
    """
    S1 = Σ c_p / (p(p-1)),   S2 = Σ c_p^2 / p,  over p ≤ k.
    """
    S1 = 0.0
    S2 = 0.0
    for p in primes_up_to(k):
        c = prime_weight(p, weight_type)
        S1 += c / (p * (p - 1))
        S2 += (c*c) / p
    return S1, max(S2, 1e-16)


# ------------------------------
# ρ² and M_new
# ------------------------------
def rho_squared(kappa: float, R0: float, U_size: float, k: int, weight_type: str = "optimized") -> float:
    S1, S2 = prime_sums(k, weight_type)
    rho2 = float(U_size) * max(0.0, R0) * (kappa**2) * (S1*S1) / S2
    return max(0.0, min(1.0, rho2))

def M_new_from_rho2(rho2: float, M_base: float = 246.0) -> float:
    return (1.0 - rho2) * M_base

def even_round_down(x: float) -> int:
    return int(math.floor(x/2.0))*2


# ------------------------------
# R0 calibration
# ------------------------------
def R0_from_moments(E_T: float, Var_T: float) -> float:
    """R0 = (E[T])^2 / Var(T) from baseline (unmodified) measured moments."""
    if Var_T <= 0:
        return 0.0
    return (E_T * E_T) / Var_T


# ------------------------------
# Normalized tilt: Euler product + proxy δ
# ------------------------------
def euler_prod_EW_tau(tau: float, U_size: float, k: int) -> float:
    """
    E[W_{τ,k}] = ∏_{p≤k} (1 - (1 - e^{-τ})/p)^{|U|}
    Use log-domain to avoid underflow.
    """
    if tau == 0.0:
        return 1.0
    factor = math.exp(-tau)
    log_prod = 0.0
    for p in primes_up_to(k):
        term = 1.0 - (1.0 - factor)/p
        if term <= 0.0:
            return 0.0
        log_prod += U_size * math.log(term)
    return math.exp(log_prod)

def euler_prod_ratio(tau: float, U_size: float, k: int) -> float:
    """
    EP_ratio = E[W_{2τ}] / (E[W_{τ}])^2  (≥ 1).
    """
    EW_t = euler_prod_EW_tau(tau, U_size, k)
    EW_2t = euler_prod_EW_tau(2.0*tau, U_size, k)
    if EW_t <= 0:
        return 1.0
    return EW_2t / (EW_t*EW_t)

def delta_proxy(U_size: float, tau: float, c0: float = 0.50) -> float:
    """
    Conservative proxy: δ ≈ c0 * |U| * τ^2 (positive). Clamp to [0,1).
    """
    val = max(0.0, c0 * U_size * (tau*tau))
    return min(val, 0.99)

def M_new_with_tilt(rho2: float, delta: float, M_base: float = 246.0) -> float:
    """
    Combine projection (rho2) and normalized-tilt (delta): effective factor ≈ (1 - rho2)*(1 - delta).
    """
    delta = max(0.0, min(0.99, delta))
    return (1.0 - rho2) * (1.0 - delta) * M_base


# ------------------------------
# Helpers
# ------------------------------
@dataclass
class Scenario:
    name: str
    use_optimal_A: bool         # if True, ignore A_coeffs and solve optimal degree
    deg_for_opt: int            # degree used for optimal-A search (len(A') = deg)
    A_coeffs: Optional[List[float]]  # used if use_optimal_A=False
    bump_deriv: List[float]     # b'(t) coefficients
    R0_mode: str                # "manual" | "moments"
    R0_manual: float            # used if R0_mode="manual"
    E_T_base: float             # used if R0_mode="moments"
    Var_T_base: float           # used if R0_mode="moments"
    U_size: float
    k_cut: int
    weight_type: str = "optimized"
    tau: float = 0.0            # normalized tilt parameter (0 = off)
    delta_mode: str = "proxy"   # "none" | "proxy" | "manual"
    delta_manual: float = 0.0   # used if delta_mode="manual"
    c0_proxy: float = 0.50      # δ ≈ c0 |U| τ^2

def scan_kappa(b_deriv_coeffs: List[float], max_deg: int = 6) -> List[Tuple[int,float]]:
    """Convenience: κ vs polynomial degree."""
    out = []
    for d in range(1, max_deg+1):
        kmax, _ = optimal_kappa_and_Aprime(b_deriv_coeffs, d)
        out.append((d, kmax))
    return out

def run_scenario(sc: Scenario, M_base: float = 246.0) -> None:
    if sc.k_cut < 2:
        raise ValueError("k_cut must be ≥ 2.")
    if sc.U_size < 0:
        raise ValueError("|U| must be nonnegative.")
    if sc.R0_mode not in ("manual", "moments"):
        raise ValueError("R0_mode must be 'manual' or 'moments'.")

    # κ
    if sc.use_optimal_A:
        kappa_max, a_der = optimal_kappa_and_Aprime(sc.bump_deriv, sc.deg_for_opt)
        A_used = integrate_ader_to_Acoeffs(a_der, A0=0.0)
        kappa_val = kappa_max
        tagA = f"optimal (deg={sc.deg_for_opt})"
    else:
        A_used = sc.A_coeffs or [1.0, -1.0]
        kappa_val = kappa_of_A(A_used, sc.bump_deriv)
        tagA = "given"

    # R0
    if sc.R0_mode == "moments":
        R0_val = R0_from_moments(sc.E_T_base, sc.Var_T_base)
        tagR0 = f"moments (E={sc.E_T_base:.4g}, Var={sc.Var_T_base:.4g})"
    else:
        R0_val = sc.R0_manual
        tagR0 = "manual"

    # rho^2 and M_new
    rho2 = rho_squared(kappa_val, R0_val, sc.U_size, sc.k_cut, sc.weight_type)
    Mproj = M_new_from_rho2(rho2, M_base)

    # Tilt reporting
    EP_ratio = euler_prod_ratio(sc.tau, sc.U_size, sc.k_cut) if sc.tau != 0 else 1.0
    if sc.delta_mode == "none":
        delta = 0.0
        tag_delta = "none"
    elif sc.delta_mode == "manual":
        delta = max(0.0, sc.delta_manual)
        tag_delta = f"manual ({delta:.4f})"
    else:
        delta = delta_proxy(sc.U_size, sc.tau, sc.c0_proxy)
        tag_delta = f"proxy (c0={sc.c0_proxy:.2f})"

    Mtilt = M_new_with_tilt(rho2, delta, M_base)

    # Print
    print(f"\n=== {sc.name} ===")
    print(f"A(t)                 : {tagA} | coeffs={A_used}")
    print(f"b'(t)                : {sc.bump_deriv}")
    print(f"κ                    : {kappa_val:.6f}  (clamped to [0,1])")
    print(f"R0                   : {R0_val:.6f}  [{tagR0}]")
    print(f"|U|                  : {sc.U_size:.3f}")
    print(f"k cutoff             : {sc.k_cut}, weight_type={sc.weight_type}")
    print(f"ρ² (lower bound)     : {rho2:.6f}  (clamped to [0,1])")
    print(f"M_new (projection)   : {Mproj:.3f}  => m0 ≤ {even_round_down(Mproj)}")
    print(f"τ (tilt)             : {sc.tau:.4f}  | EP_ratio={EP_ratio:.6f}")
    print(f"δ (tilt choice)      : {delta:.6f}   [{tag_delta}]")
    print(f"M_new (with tilt)    : {Mtilt:.3f}  => m0 ≤ {even_round_down(Mtilt)}")



# ------------------------------
# CONFIG / MAIN
# ------------------------------
if __name__ == "__main__":
    M_BASE = 246.0

    # Choose a bump not orthogonal to A. A safe default:
    #   b'(t) = 1 - t (mean-zero-ish over [0,1], works with A having a linear part)
    bump_default = [1.0, -1.0]

    # Quick κ-by-degree scan (optional)
    print("κ by degree (b'(t)=1 - t):", scan_kappa(bump_default, 5))

    # -- Example 1: Two-window, optimal A degree-2, R0 from moments, with a small tilt
    sc_two_opt_moments = Scenario(
        name="Two-Window (optimal A degree-2) + moments-calibrated R0 + small tilt",
        use_optimal_A=True,
        deg_for_opt=2,
        A_coeffs=None,
        bump_deriv=bump_default,
        R0_mode="moments",
        R0_manual=0.0,          # ignored in "moments" mode
        E_T_base=1.00,          # <-- put your measured baseline E[T] here
        Var_T_base=40.0,        # <-- put your measured baseline Var(T) here
        U_size=3.0,
        k_cut=100000,
        weight_type="optimized",
        tau=0.12 / math.sqrt(max(1.0, math.log(math.log(100000)))) ,  # small normalized tilt
        delta_mode="proxy",     # use proxy δ ≈ c0 |U| τ²
        delta_manual=0.0,
        c0_proxy=0.50,
    )

    # -- Example 2: Two-window, optimal A degree-2, manual R0, no tilt (δ=0)
    sc_two_opt_manual = Scenario(
        name="Two-Window (optimal A degree-2) + manual R0, no tilt",
        use_optimal_A=True,
        deg_for_opt=2,
        A_coeffs=None,
        bump_deriv=bump_default,
        R0_mode="manual",
        R0_manual=0.0185,       # <-- set to your calibrated sieve value
        E_T_base=0.0, Var_T_base=0.0,
        U_size=3.0,
        k_cut=100000,
        weight_type="optimized",
        tau=0.0,
        delta_mode="none",
        delta_manual=0.0,
        c0_proxy=0.50,
    )

    # -- Example 3: Single-window, optimal A degree-1, manual R0, with manual tilt δ
    sc_single_opt_manual_tilt = Scenario(
        name="Single-Window (optimal A degree-1) + manual R0 + manual tilt δ",
        use_optimal_A=True,
        deg_for_opt=1,
        A_coeffs=None,
        bump_deriv=bump_default,
        R0_mode="manual",
        R0_manual=0.0150,       # example
        E_T_base=0.0, Var_T_base=0.0,
        U_size=2.0,
        k_cut=100000,
        weight_type="optimized",
        tau=0.10 / math.sqrt(max(1.0, math.log(math.log(100000)))),
        delta_mode="manual",
        delta_manual=0.014,     # e.g., proven-level illustration
        c0_proxy=0.50,
    )

    for sc in (sc_two_opt_moments, sc_two_opt_manual, sc_single_opt_manual_tilt):
        run_scenario(sc, M_BASE)
