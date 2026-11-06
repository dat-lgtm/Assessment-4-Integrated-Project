
import control as ct
import numpy as np
from robot_model import T, wrap, error_model
from navigation import DOCK_GOAL
from sensors import MARKER_POSITION

V = .10
OMEGA_MAX = .8


def design_controller(quiet=False):
    Ad, Bd = error_model(V, 0., T)

    Q_lqr = np.diag([1/.08**2, 1/np.deg2rad(15)**2])
    R_lqr = np.array([[1/.4**2]])
    K_lqr, P_lqr, E_lqr = ct.dlqr(Ad, Bd, Q_lqr, R_lqr)
    if not quiet:
        print('Error-model controllability rank:', np.linalg.matrix_rank(ct.ctrb(Ad, Bd)))
        print('K_lqr:', K_lqr, 'pole magnitudes:', np.abs(E_lqr))
        A_pose = np.array([[1, 0, 0], [0, 1, V*T], [0, 0, 1.]])
        print('Visible camera-pose observability rank:',
              np.linalg.matrix_rank(ct.obsv(A_pose, np.eye(3))))
        print('Encoders alone do not observe global drift when the camera is absent.')
    return K_lqr


class LQRController:
    def __init__(self, K):
        self.K = np.asarray(K).reshape(1, 2)

    def step(self, xh, path):
        if len(path) < 2:
            return 0., 0.
        points = np.array(path)
        i = np.argmin(np.linalg.norm(points-xh[:2], axis=1))
        target = points[min(i+3, len(points)-1)]
        delta = target-xh[:2]
        desired = np.arctan2(delta[1], delta[0])
        eth = float(wrap(xh[2]-desired))
        if abs(eth) > .55:
            return 0., float(np.clip(-1.2*eth, -OMEGA_MAX, OMEGA_MAX))
        start = points[i]
        ey = -np.sin(desired)*(xh[0]-start[0])+np.cos(desired)*(xh[1]-start[1])
        omega = -(self.K @ np.array([ey, eth])).item()
        return V*max(.35, np.cos(eth)), float(np.clip(omega, -OMEGA_MAX, OMEGA_MAX))


def docking_command(xh, marker_recent):
    if not marker_recent:
        desired = np.arctan2(MARKER_POSITION[1]-xh[1], MARKER_POSITION[0]-xh[0])
        return 0., float(np.clip(wrap(desired-xh[2]), -.35, .35))
    delta = DOCK_GOAL[:2]-xh[:2]
    distance = np.linalg.norm(delta)
    if distance < .055:
        return 0., float(np.clip(-1.2*wrap(xh[2]), -.35, .35))
    eth = float(wrap(np.arctan2(delta[1], delta[0])-xh[2]))
    v = min(.06, .5*distance) if abs(eth) < .5 else 0.
    return v, float(np.clip(1.5*eth, -.5, .5))


def safety_command(v, omega, angles, ranges, docking):
    front = np.abs(wrap(angles)) < np.deg2rad(25)
    valid = ranges[front & np.isfinite(ranges)]
    if v > 0 and len(valid) and np.min(valid) < (.20 if docking else .30):
        return 0., omega
    return v, omega
