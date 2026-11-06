
import numpy as np
from robot_model import wrap, f_uni, F_uni

def ekf_predict(xh, P, u, Qd):
    F_t = F_uni(xh, u)
    P = F_t @ P @ F_t.T+Qd
    return f_uni(xh, u), (P+P.T)/2


def ekf_update(xh, P, z, Rn):
    if z is None:
        return xh, P, np.nan, False
    H = np.eye(3)
    nu = z-xh
    nu[2] = wrap(nu[2])
    S = H @ P @ H.T+Rn
    nis = float(nu @ np.linalg.solve(S, nu))
    if nis > 16.27:
        return xh, P, nis, False
    K = np.linalg.solve(S, H @ P).T
    xh = xh+K @ nu
    xh[2] = wrap(xh[2])

    I_KH = np.eye(3)-K @ H
    P = I_KH @ P @ I_KH.T+K @ Rn @ K.T
    return xh, (P+P.T)/2, nis, True
