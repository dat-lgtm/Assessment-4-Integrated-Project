import cv2
import mujoco
import numpy as np


class CompatibleViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.title = 'MuJoCo Track A - 3D view'
        self.renderer = mujoco.Renderer(model, height=720, width=960)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.reset_camera()
        self.previous = None
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 960, 720)
        cv2.setMouseCallback(self.title, self.mouse)

    def reset_camera(self):
        self.cam.lookat[:] = [0, 0, 0]
        self.cam.distance = 10
        self.cam.azimuth = 130
        self.cam.elevation = -55

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff
            if delta >= 32768:
                delta -= 65536
            self.cam.distance = np.clip(self.cam.distance * (0.9 if delta > 0 else 1.1), 0.3, 30)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_camera()
        elif event == cv2.EVENT_MOUSEMOVE and self.previous is not None:
            dx, dy = x - self.previous[0], y - self.previous[1]
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.cam.azimuth -= 0.3 * dx
                self.cam.elevation = np.clip(self.cam.elevation - 0.3 * dy, -89, -5)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                angle = np.deg2rad(self.cam.azimuth)
                right = np.array([-np.sin(angle), np.cos(angle), 0.0])
                forward = np.array([np.cos(angle), np.sin(angle), 0.0])
                self.cam.lookat[:] += self.cam.distance * 0.002 * (-dx * right + dy * forward)
        self.previous = (x, y)

    def is_running(self):
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1

    def sync(self):
        self.renderer.update_scene(self.data, camera=self.cam)
        frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, 'Left drag: orbit | Right drag: pan | Wheel: zoom | Double click: reset',
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.title, frame)

    def close(self):
        self.renderer.close()
