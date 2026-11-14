"""Numba-compiled hot loop for the flight simulator (flight.Sim). Pure numerics; no DroneBench types.

One call advances the aircraft by dt with RK4, runs the servo, powertrain and gust filters, and returns the
load factor. About 15 microseconds per step once compiled, which is what makes RL training practical.

Parameter vector layout is built in flight.pack_params(); tables are passed as separate arrays.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

G = 9.80665
RHO0 = 1.225

# parameter vector indices
(P_MASS, P_S, P_B, P_C, P_REF, P_PROP, P_I, P_IINV, P_ASTALL, P_CD0F, P_CDAX, P_INC, P_NTR, P_ESC, P_CAP, P_RBAT, P_CELLS,
 P_USABLE, P_DEFL, P_RATE, P_TAU, P_TAUP) = (0, 1, 2, 3, 4, 7, 10, 19, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41)
NP = 48

# derivative vector indices
(D_CLQ, D_CLDE, D_CYB, D_CYP, D_CYR, D_CYDR, D_CLB, D_CLP, D_CLR, D_CLDA, D_CLDR, D_CMQ, D_CMDE, D_CNB, D_CNP, D_CNR,
 D_CNDA, D_CNDR, D_CDDE, D_CDDA) = range(20)
ND = 20


@njit(cache=True, fastmath=False)
def _dcm(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (y * y + z * z); R[0, 1] = 2 * (x * y - w * z); R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z); R[1, 1] = 1 - 2 * (x * x + z * z); R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y); R[2, 1] = 2 * (y * z + w * x); R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


@njit(cache=True)
def _interp1(xs, ys, x):
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1 - t) + ys[hi] * t


@njit(cache=True)
def _tab_lookup(tab, alphas, lspeeds, alpha, V):
    n = alphas.shape[0]
    a = min(max(alpha, alphas[0]), alphas[n - 1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if alphas[mid] <= a:
            lo = mid
        else:
            hi = mid
    t = (a - alphas[lo]) / (alphas[hi] - alphas[lo])
    m = lspeeds.shape[0]
    lv = math.log(min(max(V, math.exp(lspeeds[0])), math.exp(lspeeds[m - 1])))
    # interpolate along speed
    k = 0
    while k < m - 2 and lspeeds[k + 1] <= lv:
        k += 1
    s = (lv - lspeeds[k]) / (lspeeds[k + 1] - lspeeds[k])
    r0 = tab[k, lo] * (1 - t) + tab[k, hi] * t
    r1 = tab[k + 1, lo] * (1 - t) + tab[k + 1, hi] * t
    return r0 * (1 - s) + r1 * s


@njit(cache=True)
def _bilin(grid, xs, ys, x, y):
    nx, ny = xs.shape[0], ys.shape[0]
    x = min(max(x, xs[0]), xs[nx - 1]); y = min(max(y, ys[0]), ys[ny - 1])
    i = 0
    while i < nx - 2 and xs[i + 1] <= x:
        i += 1
    j = 0
    while j < ny - 2 and ys[j + 1] <= y:
        j += 1
    tx = (x - xs[i]) / (xs[i + 1] - xs[i]); ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    return (grid[i, j] * (1 - tx) * (1 - ty) + grid[i + 1, j] * tx * (1 - ty) + grid[i, j + 1] * (1 - tx) * ty
            + grid[i + 1, j + 1] * tx * ty)


@njit(cache=True)
def _forces(vb, w, q, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm, alt):
    R = _dcm(q)
    va = vb - R.T @ wind
    V = max(math.sqrt(va[0] ** 2 + va[1] ** 2 + va[2] ** 2), 0.1)
    alpha = math.atan2(va[2], max(va[0], 1e-3))
    beta = math.asin(max(-1.0, min(1.0, va[1] / V)))
    rho = RHO0 * math.exp(-alt / 9300.0)
    qbar = 0.5 * rho * V * V
    S, b, c = P[P_S], P[P_B], P[P_C]
    vv = max(V, 1.0)
    ph, qh, rh = w[0] * b / (2 * vv), w[1] * c / (2 * vv), w[2] * b / (2 * vv)
    da, de, dr = surf[0], surf[1] + P[P_INC], surf[2]
    cl = _tab_lookup(CL, alphas, lspeeds, alpha, V) + D[D_CLQ] * qh + D[D_CLDE] * de
    cd = (_tab_lookup(CD, alphas, lspeeds, alpha, V) + P[P_CDAX] / S) * P[P_CD0F] + D[D_CDDE] * (abs(surf[1]) + 0.4 * abs(P[P_INC])) + D[D_CDDA] * abs(da)
    f = 1.0
    ast = P[P_ASTALL]
    if alpha > ast:
        f = min(max(1.0 - (alpha - ast) / 0.18, 0.25), 1.0)
    cy = f * (D[D_CYB] * beta + D[D_CYP] * ph + D[D_CYR] * rh + D[D_CYDR] * dr)
    cll = f * (D[D_CLB] * beta + D[D_CLP] * ph + D[D_CLR] * rh + D[D_CLDA] * da + D[D_CLDR] * dr)
    cm = _tab_lookup(Cm, alphas, lspeeds, alpha, V) + D[D_CMQ] * qh + D[D_CMDE] * de
    cn = f * (D[D_CNB] * beta + D[D_CNP] * ph + D[D_CNR] * rh + D[D_CNDA] * da + D[D_CNDR] * dr)
    L, Dg, Y = qbar * S * cl, qbar * S * cd, qbar * S * cy
    ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
    Fw0, Fw1, Fw2 = -Dg, Y, -L
    Fa = np.empty(3)
    Fa[0] = ca * cb * Fw0 - ca * sb * Fw1 - sa * Fw2
    Fa[1] = sb * Fw0 + cb * Fw1
    Fa[2] = sa * cb * Fw0 - sa * sb * Fw1 + ca * Fw2
    refb = P[P_REF:P_REF + 3]
    Ma = np.empty(3)
    Ma[0] = qbar * S * b * cll + refb[1] * Fa[2] - refb[2] * Fa[1]
    Ma[1] = qbar * S * c * cm + refb[2] * Fa[0] - refb[0] * Fa[2]
    Ma[2] = qbar * S * b * cn + refb[0] * Fa[1] - refb[1] * Fa[0]
    pp = P[P_PROP:P_PROP + 3]
    Mp = np.empty(3)
    Mp[0] = 0.0
    Mp[1] = pp[2] * thrust                 # r x (T,0,0): y-component = r_z * T
    Mp[2] = -pp[1] * thrust
    g_b = R.T @ np.array([0.0, 0.0, P[P_MASS] * G])
    F = Fa + g_b
    F[0] += thrust
    return F, Ma + Mp, alpha, V


@njit(cache=True)
def _deriv(x, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm):
    q = x[6:10]
    vb = x[3:6]
    w = x[10:13]
    F, M, alpha, V = _forces(vb, w, q, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm, -x[2])
    R = _dcm(q)
    out = np.empty(13)
    dpos = R @ vb
    out[0], out[1], out[2] = dpos[0], dpos[1], dpos[2]
    m = P[P_MASS]
    out[3] = F[0] / m - (w[1] * vb[2] - w[2] * vb[1])
    out[4] = F[1] / m - (w[2] * vb[0] - w[0] * vb[2])
    out[5] = F[2] / m - (w[0] * vb[1] - w[1] * vb[0])
    wx, wy, wz = w[0], w[1], w[2]
    out[6] = 0.5 * (-q[1] * wx - q[2] * wy - q[3] * wz)
    out[7] = 0.5 * (q[0] * wx + q[2] * wz - q[3] * wy)
    out[8] = 0.5 * (q[0] * wy - q[1] * wz + q[3] * wx)
    out[9] = 0.5 * (q[0] * wz + q[1] * wy - q[2] * wx)
    I = P[P_I:P_I + 9].reshape(3, 3)
    Iinv = P[P_IINV:P_IINV + 9].reshape(3, 3)
    Iw = I @ w
    gyro = np.array([w[1] * Iw[2] - w[2] * Iw[1], w[2] * Iw[0] - w[0] * Iw[2], w[0] * Iw[1] - w[1] * Iw[0]])
    dw = Iinv @ (M - gyro)
    out[10], out[11], out[12] = dw[0], dw[1], dw[2]
    return out


@njit(cache=True)
def step(x, surf, pt, gust_state, cmd, noise, dt, P, D, alphas, lspeeds, CL, CD, Cm, wind_mean,
         pv_grid, pv_vm, pv_va, pv_T, pv_I, batt_soc, batt_v, gust_s, gust_L):
    """Advance one step. Mutates x (13), surf (3), pt (4: thrust state, current state, used_wh, i_batt),
    gust_state (3). cmd = (da, de, dr, throttle). Returns load factor n and airspeed."""
    # servos
    defl, rate, tau = P[P_DEFL], P[P_RATE], P[P_TAU]
    for i in range(3):
        c = min(max(cmd[i], -defl), defl)
        d = (c - surf[i]) * (1.0 - math.exp(-dt / tau))
        d = min(max(d, -rate * dt), rate * dt)
        surf[i] += d
    # airspeed (for gust filter and powertrain)
    q0 = x[6:10]
    R = _dcm(q0)
    wind = wind_mean + gust_state
    va = x[3:6] - R.T @ wind
    V = max(math.sqrt(va[0] ** 2 + va[1] ** 2 + va[2] ** 2), 0.1)
    Vg = max(V, 3.0)
    for k in range(3):
        a = math.exp(-Vg / gust_L[k] * dt)
        gust_state[k] = a * gust_state[k] + gust_s[k] * math.sqrt(2.0 * Vg / gust_L[k] * dt) * noise[k]
    wind = wind_mean + gust_state
    # powertrain (table lookup, first-order spin-up)
    soc = max(0.0, 1.0 - pt[2] / max(P[P_CAP], 1e-9))
    voc = P[P_CELLS] * _interp1(batt_soc, batt_v, soc)
    vterm = max(voc - pt[3] * P[P_RBAT], 0.5 * voc)
    thr = min(max(cmd[3], 0.0), 1.0)
    vm = thr * 0.98 * vterm
    rho = RHO0 * math.exp(-(-x[2]) / 9300.0)
    T_t = _bilin(pv_T, pv_vm, pv_va, vm, V) * rho / RHO0
    I_t = _bilin(pv_I, pv_vm, pv_va, vm, V) * rho / RHO0
    k_sp = 1.0 - math.exp(-dt / P[P_TAUP])
    pt[0] += (T_t - pt[0]) * k_sp
    pt[1] += (I_t - pt[1]) * k_sp
    ntr = P[P_NTR]
    thrust = ntr * max(pt[0], 0.0)
    p_motor = vm * max(pt[1], 0.0)
    p_batt = ntr * p_motor / P[P_ESC]
    pt[3] = p_batt / max(vterm, 1.0)
    pt[2] += voc * pt[3] * dt / 3600.0
    # RK4
    k1 = _deriv(x, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm)
    k2 = _deriv(x + 0.5 * dt * k1, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm)
    k3 = _deriv(x + 0.5 * dt * k2, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm)
    k4 = _deriv(x + dt * k3, surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm)
    xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    nq = math.sqrt(xn[6] ** 2 + xn[7] ** 2 + xn[8] ** 2 + xn[9] ** 2)
    for i in range(13):
        x[i] = xn[i] / nq if 6 <= i <= 9 else xn[i]
    # load factor
    F, M, alpha, V2 = _forces(x[3:6], x[10:13], x[6:10], surf, thrust, wind, P, D, alphas, lspeeds, CL, CD, Cm, -x[2])
    Rn = _dcm(x[6:10])
    gz = (Rn.T @ np.array([0.0, 0.0, P[P_MASS] * G]))[2]
    n = -(F[2] - gz) / (P[P_MASS] * G)
    return n, V2, alpha, thrust, p_batt, soc
