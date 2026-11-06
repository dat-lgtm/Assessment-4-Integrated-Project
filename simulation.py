
import argparse
import time
import cv2
import numpy as np
from robot import TurtleBot3
from compatible_viewer import CompatibleViewer
from robot_model import T, X0_NOMINAL, wrap, encoder_input
from noise import P0, Q_D, R_LIDAR, ENCODER_SIGMA, TRIAL_CONDITIONS
from sensors import CameraSensor, calibration_check, calibrate_odometry, lidar_pose_measurement
from ekf import ekf_predict, ekf_update
from navigation import Navigator, STAGING, DOCK_GOAL, display_map
from controller import LQRController, design_controller, docking_command, safety_command
from evaluation import obstacle_contact, evaluate, cross_track_error, plot_results

DEFAULT_TRIALS = 1


def run(seconds=200, headless=False, seed=1, noise_scale=1., camera_dropout=0.,
        wheel_bias=0., plots=False, quiet=False):
    robot = TurtleBot3(seed)
    rng = np.random.default_rng(seed+100)
    xh, P = X0_NOMINAL.copy(), P0.copy()
    controller = LQRController(design_controller(quiet))
    navigator, camera = Navigator(), CameraSensor()
    viewer = None if headless else CompatibleViewer(robot.model, robot.data)
    previous = robot.encoders()
    path, state = [], 'NAVIGATE'
    last_marker, dock_count = -np.inf, 0
    frame = np.zeros((480, 640, 3), np.uint8)
    log = {'time': [], 'truth': [], 'estimate': [], 'P': [], 'u': [], 'nis': [],
           'completed': False, 'contact_steps': 0, 'camera_updates': 0, 'lidar_updates': 0,
           'cross_track': [], 'cross_track_est': [], 'phase': [], 'reprojection': [], 'marker_frame': None,
           'initial_path': [], 'last_path': [], 'seed': seed,
           'noise_scale': noise_scale, 'wheel_bias': wheel_bias}
    try:
        for k in range(round(seconds/T)):
            wall_start, now = time.perf_counter(), k*T
            if viewer is not None and not viewer.is_running():
                break
            ranges = robot.scan()
            if k % 10 == 0:
                z_lidar = lidar_pose_measurement(xh, robot.angles, ranges)
                R_lidar = R_LIDAR
                xh, P, _, accepted = ekf_update(xh, P, z_lidar, R_lidar)
                log['lidar_updates'] += int(accepted)
            if state == 'NAVIGATE':
                if k % 20 == 0 or (not path and k % 5 == 0):
                    path = navigator.plan(xh, robot.angles, ranges)
                    if len(path) > 1:
                        if not log['initial_path']:
                            log['initial_path'] = np.array(path).tolist()
                        log['last_path'] = np.array(path).tolist()
                if np.linalg.norm(xh[:2]-STAGING) < .16:
                    state = 'DOCK'
                v, omega = controller.step(xh, path)
            else:
                v, omega = docking_command(xh, now-last_marker < 1.)
            v, omega = safety_command(v, omega, robot.angles, ranges, state == 'DOCK')

            robot.command(v, omega+wheel_bias)
            robot.step()
            current = robot.encoders()+rng.normal(0, ENCODER_SIGMA*noise_scale, 2)
            u_odom = encoder_input(previous, current)
            previous = current
            xh, P = ekf_predict(xh, P, u_odom, Q_D*noise_scale**2)
            nis = np.nan
            reprojection = np.nan
            if k % 4 == 0:
                z, frame = camera.measure(robot.camera(), rng, noise_scale, camera_dropout)
                reprojection = camera.reprojection_error
                if z is not None:
                    log['marker_frame'] = frame.copy()
                xh, P, nis, accepted = ekf_update(xh, P, z, camera.R*noise_scale**2)
                if accepted:
                    last_marker = now
                    log['camera_updates'] += 1
            if state == 'DOCK' and now-last_marker < 1 and \
                    np.linalg.norm(xh[:2]-DOCK_GOAL[:2]) < .06 and abs(wrap(xh[2])) < .12:
                dock_count += 1
            else:
                dock_count = 0
            log['time'].append((k+1)*T)
            truth = robot.truth_for_evaluation()
            log['truth'].append(truth)

            log['cross_track'].append(cross_track_error(truth[:2], path)
                                      if state == 'NAVIGATE' else np.nan)
            log['cross_track_est'].append(cross_track_error(xh[:2], path)
                                          if state == 'NAVIGATE' else np.nan)
            log['phase'].append(state)
            log['reprojection'].append(reprojection)
            log['estimate'].append(xh.copy())
            log['P'].append(P.copy())
            log['u'].append([v, omega])
            log['nis'].append(nis)
            log['contact_steps'] += int(obstacle_contact(robot))
            if not quiet and k % 100 == 0:
                print(f't={now:5.1f}s {state:8s} estimate={np.round(xh, 2)} '
                      f'camera={log["camera_updates"]} lidar={log["lidar_updates"]} '
                      f'path={len(path)}', flush=True)
            if viewer is not None:
                viewer.sync()
                if k % 4 == 0:
                    cv2.imshow('ArUco camera - measured pixels', frame)
                    display_state = 'NO ROUTE - STOPPED' if state == 'NAVIGATE' and not path else state
                    cv2.imshow('Navigation - estimated state only', display_map(xh, path, display_state))
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
                time.sleep(max(0, T-(time.perf_counter()-wall_start)))
            if dock_count >= 10:
                log['completed'] = True
                break
    finally:
        robot.command(0, 0)
        robot.close()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()
    if not log['time']:
        return None
    metrics, err, nees = evaluate(log)
    print('\nTRIAL RESULTS (truth used for evaluation only)')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}' if isinstance(value, float) else f'  {name}: {value}')
    if plots:
        plot_results(log, err, nees)
    return metrics


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=200)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Show result figures after the simulation (default)')
    parser.add_argument('--no-plots', action='store_false', dest='plots')
    parser.add_argument('--calibration-check', action='store_true')
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.trials < 1:
        parser.error('seconds and trials must be positive')
    if args.calibration_check:
        calibration_check(args.plots)
    calibrate_odometry()
    results = []
    for trial in range(args.trials):
        conditions = TRIAL_CONDITIONS
        noise, dropout, bias = conditions[trial % 3]
        print(f'\nTRIAL {trial+1}: seed={args.seed+trial}, noise={noise}, '
              f'camera dropout={dropout}, physical yaw bias={bias}')
        result = run(args.seconds, args.headless, args.seed+trial,
                     noise, dropout, bias, args.plots)
        if result is not None:
            result['condition'] = trial % 3
            results.append(result)
    if len(results) > 1:
        print('Success rate:', np.mean([r['success'] for r in results]))
        print('Small-sample test only, not a reliability guarantee.')
    if args.plots:
        import matplotlib.pyplot as plt
        plt.show()
