
import control as ct
import numpy as np
from robot import CONTROL_DT, WHEEL_RADIUS, WHEEL_SEPARATION

T = CONTROL_DT
X0_NOMINAL = np.array([-2., -2., np.pi/4])
ODOMETRY_SCALE = np.ones(2)
CAMERA_OFFSET = .082
LASER_OFFSET = -.032


def wrap(a):
    return (np.asarray(a)+np.pi) % (2*np.pi)-np.pi


def error_model(v, kappa, Ts):
    A = np.array([[0., v], [-v*kappa**2, 0.]])
    B = np.array([[0.], [1.]])
    d = ct.ss(A, B, np.eye(2), np.zeros((2, 1))).sample(Ts, method='zoh')
    return d.A, d.B


def f_uni(xh, u):
    v, w = u
    x, y, th = xh
    if abs(w) < 1e-6:
        return np.array([x+v*T*np.cos(th), y+v*T*np.sin(th), wrap(th)])
    th_n = th+w*T
    return np.array([x+v/w*(np.sin(th_n)-np.sin(th)),
                     y+v/w*(-np.cos(th_n)+np.cos(th)), wrap(th_n)])


def F_uni(xh, u):
    v, w = u
    th = xh[2]
    F_t = np.eye(3)
    if abs(w) < 1e-6:
        F_t[0, 2], F_t[1, 2] = -v*T*np.sin(th), v*T*np.cos(th)
    else:
        F_t[0, 2] = v/w*(np.cos(th+w*T)-np.cos(th))
        F_t[1, 2] = v/w*(np.sin(th+w*T)-np.sin(th))
    return F_t


def encoder_input(previous, current):
    dl, dr = WHEEL_RADIUS*(current-previous)
    return ODOMETRY_SCALE*np.array([(dl+dr)/(2*T), (dr-dl)/(WHEEL_SEPARATION*T)])
