
import numpy as np
from robot_model import X0_NOMINAL, wrap
from navigation import MAP_BOXES, MAP_PILLAR, DOCK_GOAL

def obstacle_contact(robot):
    for contact in robot.data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        ba, bb = robot.model.geom_bodyid[[a, b]]
        if (ba == 0) != (bb == 0):
            world_geom = a if ba == 0 else b
            if robot.model.geom(world_geom).name != 'floor':
                return True
    return False


def evaluate(log):
    truth, estimate, PP = map(np.asarray, (log['truth'], log['estimate'], log['P']))
    err = truth-estimate
    err[:, 2] = wrap(err[:, 2])
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(err, PP)])
    dock_error = np.linalg.norm(truth[-1, :2]-DOCK_GOAL[:2])
    yaw_error = abs(float(wrap(truth[-1, 2])))
    tracking = np.asarray(log['cross_track'])
    tracking = tracking[np.isfinite(tracking)]
    result = {
        'success': bool(log['completed'] and dock_error < .12 and
                        yaw_error < np.deg2rad(15) and log['contact_steps'] == 0),
        'completion_time_s': float(log['time'][-1]),
        'position_rmse_m': float(np.sqrt(np.mean(np.sum(err[:, :2]**2, axis=1)))),
        'heading_rmse_deg': float(np.rad2deg(np.sqrt(np.mean(err[:, 2]**2)))),
        'tracking_rmse_m': float(np.sqrt(np.mean(tracking**2))) if len(tracking) else np.nan,
        'dock_position_error_m': float(dock_error),
        'dock_heading_error_deg': float(np.rad2deg(yaw_error)),
        'rms_steering_rad_s': float(np.sqrt(np.mean(np.asarray(log['u'])[:, 1]**2))),
        'anees': float(np.mean(nees)), 'nees_inside_95': float(np.mean(nees <= 7.815)),
        'camera_updates': log['camera_updates'], 'lidar_updates': log['lidar_updates'],
        'obstacle_contact_steps': log['contact_steps']}
    return result, err, nees


def cross_track_error(position, path):

    if len(path) < 2:
        return np.nan
    points = np.asarray(path)
    segments = points[1:]-points[:-1]
    lengths2 = np.sum(segments**2, axis=1)
    valid = lengths2 > 1e-10
    if not np.any(valid):
        return np.nan
    start, segments, lengths2 = points[:-1][valid], segments[valid], lengths2[valid]
    fraction = np.clip(np.sum((position-start)*segments, axis=1)/lengths2, 0, 1)
    nearest = start+fraction[:, None]*segments
    distance = np.linalg.norm(position-nearest, axis=1)
    i = np.argmin(distance)
    side = np.sign(segments[i, 0]*(position[1]-start[i, 1])-
                   segments[i, 1]*(position[0]-start[i, 0]))
    return float(distance[i]*(side if side else 1))


def plot_results(log, err, nees):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    plt.rcParams['figure.dpi'] = 100
    L = log
    t = np.asarray(L['time'])
    X = np.asarray(L['truth'])
    XH = np.asarray(L['estimate'])
    P = np.asarray(L['P'])
    trial = f"Seed {L['seed']}"


    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Navigation trajectory', fontsize=16, fontweight='bold', y=0.98)
    ax.add_patch(Rectangle((-2.5, -2.5), 5, 5, fill=False, lw=2, label='Room'))
    for i, (cx, cy, sx, sy) in enumerate(MAP_BOXES):
        ax.add_patch(Rectangle((cx-sx, cy-sy), 2*sx, 2*sy,
                               color='0.65', alpha=0.7,
                               label='Obstacles' if i == 0 else None))
    ax.add_patch(Circle(MAP_PILLAR[:2], MAP_PILLAR[2], color='0.65', alpha=0.7))
    path = np.asarray(L['initial_path'])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'g--', lw=1.2, label='Initial planned path')
    ax.plot(X[:, 0], X[:, 1], 'k-', lw=1.2, label='True path')
    ax.plot(XH[:, 0], XH[:, 1], 'r-', lw=0.8, alpha=0.7, label='EKF estimate')
    ax.scatter(*X0_NOMINAL[:2], c='tab:blue', s=50, label='Start')
    ax.scatter(*DOCK_GOAL[:2], c='tab:green', marker='x', s=90, label='Docking target')
    ax.set_aspect('equal')
    ax.set(xlim=(-2.65, 2.65), ylim=(-2.65, 2.65), xlabel='x [m]', ylabel='y [m]')
    ax.set_title(trial + ': planned route, travelled path and estimated pose', fontweight='bold')
    ax.legend(loc='upper center', ncol=3, fontsize=8)
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])


    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('EKF consistency', fontsize=16, fontweight='bold', y=0.98)
    labels = ['x estimation error [cm]', 'y estimation error [cm]', 'Heading estimation error [deg]']
    for i, ax in enumerate(axs.flat):
        if i < 3:
            scale = 180 / np.pi if i == 2 else 100
            sigma = np.sqrt(np.maximum(P[:, i, i], 0)) * scale
            ax.fill_between(t, -2*sigma, 2*sigma, color='tab:blue', alpha=0.2,
                            label=r'$\pm 2\sigma$ from $P$')
            ax.plot(t, err[:, i] * scale, color='tab:blue', lw=1, label='True minus estimated')
            ax.set_title(labels[i], fontweight='bold')
            ax.set_ylabel('[deg]' if i == 2 else '[cm]')
        else:
            ax.semilogy(t, np.maximum(nees, 1e-8), 'tab:purple', lw=0.8, label='NEES')
            ax.axhline(7.815, color='k', ls='--', label=r'95% $\chi^2_3$ upper point')
            ax.axhline(3, color='tab:green', ls=':', label='Expected mean, 3')
            ax.set_title(f'ANEES {np.mean(nees):.2f}, {100*np.mean(nees <= 7.815):.1f}% below bound',
                         fontweight='bold')
            ax.set_ylabel('NEES')
        ax.set_xlabel('time [s]')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])


    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Tracking and docking', fontsize=16, fontweight='bold', y=0.98)
    e_y_true = np.asarray(L['cross_track'])
    e_y_est = np.asarray(L['cross_track_est'])
    valid = np.isfinite(e_y_true)
    rms_ey = np.sqrt(np.mean(e_y_true[valid]**2)) if np.any(valid) else np.nan
    ax = axs[0]
    ax.plot(t, 100*e_y_true, 'k-', lw=1, label='True cross-track error')
    ax.plot(t, 100*e_y_est, 'r-', lw=0.8, alpha=0.7, label='Estimated cross-track error')
    ax.axhline(0, color='0.5', lw=0.7)
    ax.set_title(f'Navigation tracking (RMS {100*rms_ey:.1f} cm; active replanned reference)',
                 fontweight='bold')
    ax.set_ylabel('cross-track [cm]')
    ax = axs[1]
    distance_true = np.linalg.norm(X[:, :2]-DOCK_GOAL[:2], axis=1)
    distance_est = np.linalg.norm(XH[:, :2]-DOCK_GOAL[:2], axis=1)
    ax.plot(t, distance_true, 'k-', lw=1, label='True distance')
    ax.plot(t, distance_est, 'r-', lw=0.8, alpha=0.7, label='Estimated distance')
    ax.axhline(.12, color='tab:green', ls=':', label='Prototype position tolerance')
    ax.set_title(f'Distance to docking pose (final {100*distance_true[-1]:.1f} cm)', fontweight='bold')
    ax.set_ylabel('distance [m]')
    ax = axs[2]
    heading_true = np.rad2deg(wrap(X[:, 2]-DOCK_GOAL[2]))
    heading_est = np.rad2deg(wrap(XH[:, 2]-DOCK_GOAL[2]))
    ax.plot(t, heading_true, 'k-', lw=1, label='True heading difference')
    ax.plot(t, heading_est, 'r-', lw=0.8, alpha=0.7, label='Estimated heading difference')
    ax.set_title(f'Heading relative to final docking pose (final {heading_true[-1]:+.1f} deg)',
                 fontweight='bold')
    ax.set_ylabel('heading [deg]')
    ax.set_xlabel('time [s]')
    docking = np.flatnonzero(np.asarray(L['phase']) == 'DOCK')
    for ax in axs:
        if len(docking):
            ax.axvspan(t[docking[0]], t[-1], color='0.9', label='Docking phase')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
