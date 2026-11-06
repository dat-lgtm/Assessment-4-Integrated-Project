
import cv2
import numpy as np
from scipy.optimize import least_squares
from robot import TurtleBot3, WHEEL_RADIUS, WHEEL_SEPARATION
from robot_model import T, X0_NOMINAL, ODOMETRY_SCALE, CAMERA_OFFSET, LASER_OFFSET, wrap, f_uni
from noise import R_N, CORNER_SIGMA
from navigation import MAP_BOXES, MAP_PILLAR

MARKER_POSITION = np.array([1.899, 2., .24])
MARKER_SIZE = .16


F = 480/(2*np.tan(np.deg2rad(60)/2))
CAMERA_MATRIX = np.array([[F, 0, 319.5], [0, F, 239.5], [0, 0, 1.]])
DISTORTION = np.zeros(5)


class CameraSensor:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        h = MARKER_SIZE/2
        self.points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0.]])

        self.R_wm = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
        self.R = R_N.copy()
        self.reprojection_error = np.nan

    def measure(self, rgb, rng, noise_scale=1., dropout=0.):
        self.reprojection_error = np.nan
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        corners, ids, _ = self.detector.detectMarkers(rgb)
        if ids is None or rng.random() < dropout:
            return None, image
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for points, marker_id in zip(corners, ids.flatten()):
            if marker_id != 0:
                continue
            pixels = points.reshape(4, 2).astype(np.float64)
            if np.min(np.linalg.norm(np.roll(pixels, -1, axis=0)-pixels, axis=1)) < 14:
                continue
            pixels += rng.normal(0, CORNER_SIGMA*noise_scale, pixels.shape)
            ok, rvec, tvec = cv2.solvePnP(self.points, pixels, CAMERA_MATRIX,
                                         DISTORTION, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or tvec[2, 0] <= 0:
                continue
            predicted, _ = cv2.projectPoints(self.points, rvec, tvec, CAMERA_MATRIX, DISTORTION)
            rms = np.sqrt(np.mean((predicted.reshape(4, 2)-pixels)**2))
            if rms > 2:
                continue
            R_cm, _ = cv2.Rodrigues(rvec)
            R_wc = self.R_wm @ R_cm.T
            p_wc = MARKER_POSITION-R_wc @ tvec[:, 0]
            heading = np.arctan2(R_wc[1, 2], R_wc[0, 2])
            position = p_wc[:2]-CAMERA_OFFSET*np.array([np.cos(heading), np.sin(heading)])
            if abs(p_wc[2]-.151) > .20 or np.any(np.abs(position) > 3):
                continue
            z = np.array([position[0], position[1], wrap(heading)])


            distance = np.linalg.norm(tvec)
            sigma_pos = .03 + .04*distance**2
            sigma_heading = np.deg2rad(3 + 3*distance)
            self.R = np.diag([sigma_pos**2, sigma_pos**2, sigma_heading**2])
            self.reprojection_error = float(rms)
            cv2.putText(image, f'ID 0 | reprojection RMS {rms:.2f}px', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            return z, image
        return None, image


def calibrate_odometry():
    scales = []
    for v, omega in [(.1, 0.), (0., .5)]:
        robot = TurtleBot3(99)
        try:
            initial = lidar_pose_measurement(X0_NOMINAL, robot.angles, robot.scan())
            if initial is None:
                raise RuntimeError('LiDAR could not localise the calibration start.')
            before = robot.encoders()
            previous = before.copy()
            guess = initial.copy()
            robot.command(v, omega)
            for _ in range(round(3/T)):
                robot.step()
                current = robot.encoders()
                dl, dr = WHEEL_RADIUS*(current-previous)
                raw_input = np.array([(dl+dr)/(2*T), (dr-dl)/(WHEEL_SEPARATION*T)])
                guess = f_uni(guess, raw_input)
                previous = current
            final = lidar_pose_measurement(guess, robot.angles, robot.scan())
            if final is None:
                raise RuntimeError('LiDAR could not localise the calibration finish.')
            dl, dr = WHEEL_RADIUS*(robot.encoders()-before)
            if v:
                measured = np.linalg.norm(final[:2]-initial[:2])
                predicted = abs((dl+dr)/2)
            else:
                measured = abs(float(wrap(final[2]-initial[2])))
                predicted = abs((dr-dl)/WHEEL_SEPARATION)
            scale = measured/max(predicted, 1e-8)
            if not .2 < scale < 2.:
                raise RuntimeError('Unreasonable encoder scale; check LiDAR and map alignment.')
            scales.append(scale)
        finally:
            robot.command(0, 0)
            robot.close()
    ODOMETRY_SCALE[:] = scales
    print('LiDAR-based odometry calibration [distance, yaw]:', ODOMETRY_SCALE)
    return ODOMETRY_SCALE.copy()


def calibration_check(plots=False):

    rng = np.random.default_rng(10)
    board = np.zeros((42, 3), np.float32)
    board[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)*.025
    objects, images = [], []
    for _ in range(20):
        pixels, _ = cv2.projectPoints(board, rng.uniform(-.4, .4, 3),
                                     np.array([-.075, -.06, rng.uniform(.5, .9)]),
                                     CAMERA_MATRIX, DISTORTION)
        pixels += rng.normal(0, .15, pixels.shape)
        objects.append(board.copy())
        images.append(pixels.astype(np.float32))
    rms, K, distortion, rotations, translations = cv2.calibrateCamera(
        objects, images, (640, 480), None, None)
    print('SYNTHETIC calibration check; does not replace experimental calibration.')
    print('Reprojection RMS [pixels]:', rms, '\nNominal K:\n', CAMERA_MATRIX,
          '\nRecovered K:\n', K, '\nDistortion:', distortion.flatten())


def lidar_pose_measurement(xh, angles, ranges):
    valid = np.flatnonzero(np.isfinite(ranges))[::5]
    if len(valid) < 20:
        return None
    a, distance = angles[valid], ranges[valid]
    local = np.column_stack([LASER_OFFSET+distance*np.cos(a), distance*np.sin(a)])

    def residual(pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        points = local @ rotation.T+pose[:2]

        errors = [np.abs(np.abs(points[:, 0])-2.5),
                  np.abs(np.abs(points[:, 1])-2.5)]
        for cx, cy, sx, sy in MAP_BOXES:
            q = np.abs(points-np.array([cx, cy]))-np.array([sx, sy])
            signed = np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(np.max(q, axis=1), 0)
            errors.append(np.abs(signed))
        cx, cy, radius = MAP_PILLAR
        errors.append(np.abs(np.linalg.norm(points-[cx, cy], axis=1)-radius))
        return np.min(np.column_stack(errors), axis=1)

    bounds = np.array([.18, .18, np.deg2rad(18)])
    fit = least_squares(residual, xh, bounds=(xh-bounds, xh+bounds),
                        loss='soft_l1', f_scale=.02, max_nfev=30)
    if np.sqrt(np.mean(residual(fit.x)**2)) > .055:
        return None
    z = fit.x.copy()
    z[2] = wrap(z[2])
    return z
