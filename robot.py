from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160
CONTROL_DT = 0.05


class TurtleBot3:
    def __init__(self, seed=1):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.laser_id = self.model.site('laser').id
        self.base_id = self.model.body('base_link').id
        self.angles = np.deg2rad(np.arange(360))
        self.renderer = None
        mujoco.mj_forward(self.model, self.data)

        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def command(self, v, omega):

        v = np.clip(v, -0.22, 0.22)
        omega = np.clip(omega, -2.0, 2.0)
        left = (v - omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        right = (v + omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS

        scale = max(1.0, abs(left) / 8, abs(right) / 8)
        self.data.ctrl[:] = [left / scale, right / scale]

    def step(self):
        for _ in range(round(CONTROL_DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def encoders(self):

        return np.array([self.data.sensor('left_encoder').data[0],
                         self.data.sensor('right_encoder').data[0]])

    def scan(self, noise=True):
        origin = self.data.site_xpos[self.laser_id].copy()
        rotation = self.data.site_xmat[self.laser_id].reshape(3, 3)
        ranges = np.full(360, np.inf)
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        hit = np.array([-1], dtype=np.int32)
        for i, angle in enumerate(self.angles):
            direction = rotation @ np.array([np.cos(angle), np.sin(angle), 0.0])
            distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                     groups, True, self.base_id, hit)
            if 0.12 <= distance <= 3.5:
                if not noise or self.rng.random() > 0.01:
                    if noise:
                        distance += self.rng.normal(0, 0.01)
                    if 0.12 <= distance <= 3.5:
                        ranges[i] = distance
        return ranges

    def camera(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera='front_camera')
        return self.renderer.render().copy()

    def truth_for_evaluation(self):
        position = self.data.xpos[self.base_id]
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        return np.array([position[0], position[1],
                         np.arctan2(rotation[1, 0], rotation[0, 0])])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def odometry_step(xh, previous, current):
    dl, dr = WHEEL_RADIUS * (current - previous)
    distance = (dl + dr) / 2
    turn = (dr - dl) / WHEEL_SEPARATION
    th = xh[2] + turn / 2
    xh = xh + np.array([distance * np.cos(th), distance * np.sin(th), turn])
    xh[2] = (xh[2] + np.pi) % (2 * np.pi) - np.pi
    return xh
