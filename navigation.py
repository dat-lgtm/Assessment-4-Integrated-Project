
import cv2
import heapq
import numpy as np
from robot_model import wrap, LASER_OFFSET

STAGING = np.array([1.15, 2.])
DOCK_GOAL = np.array([1.52, 2., 0.])

MAP_BOXES = [(-.5, .4, .35, .35), (.9, -.4, .25, .5),
             (0., 0., .20, .20), (2., 2., .10, .32),
             (-.3, -1.45, .25, .20), (1.4, .8, .22, .25),
             (-.5, 1.575, .18, .925)]
MAP_PILLAR = (-1., 1.5, .25)


class Navigator:
    def __init__(self):
        self.resolution = .1
        self.coordinates = np.linspace(-2.5, 2.5, 51)
        self.margin = .24

    def cell(self, p):
        return tuple(np.clip(np.round((p+2.5)/self.resolution).astype(int), 0, 50))

    def world(self, c):
        return np.array(c)*self.resolution-2.5

    def occupancy(self, xh, angles, ranges):
        x, y = np.meshgrid(self.coordinates, self.coordinates, indexing='ij')
        occupied = (np.abs(x) > 2.5-self.margin) | (np.abs(y) > 2.5-self.margin)
        for cx, cy, sx, sy in MAP_BOXES:
            occupied |= (np.abs(x-cx) <= sx+self.margin) & (np.abs(y-cy) <= sy+self.margin)
        cx, cy, radius = MAP_PILLAR
        occupied |= np.hypot(x-cx, y-cy) < radius+self.margin


        valid = np.isfinite(ranges) & (ranges <= 1.0)
        origin = xh[:2]+LASER_OFFSET*np.array([np.cos(xh[2]), np.sin(xh[2])])
        a = angles[valid]+xh[2]
        hits = origin+np.column_stack([ranges[valid]*np.cos(a), ranges[valid]*np.sin(a)])
        for point in hits[::3]:
            occupied |= np.hypot(x-point[0], y-point[1]) < self.margin
        return occupied

    def plan(self, xh, angles, ranges):
        occupied = self.occupancy(xh, angles, ranges)
        start, goal = self.cell(xh[:2]), self.cell(STAGING)
        occupied[start] = False
        if occupied[goal]:
            return []
        queue, cost, parent = [(0., start)], {start: 0.}, {}
        moves = [(a, b) for a in [-1, 0, 1] for b in [-1, 0, 1] if a or b]
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal:
                cells = [current]
                while current != start:
                    current = parent[current]
                    cells.append(current)
                return [self.world(c) for c in cells[::-1]]
            for dx, dy in moves:
                nx, ny = current[0]+dx, current[1]+dy
                if not (0 <= nx < 51 and 0 <= ny < 51) or occupied[nx, ny]:
                    continue
                if dx and dy and (occupied[current[0]+dx, current[1]] or
                                  occupied[current[0], current[1]+dy]):
                    continue
                next_cell = (nx, ny)
                new_cost = cost[current]+np.hypot(dx, dy)
                if new_cost < cost.get(next_cell, np.inf):
                    cost[next_cell], parent[next_cell] = new_cost, current
                    priority = new_cost+np.hypot(nx-goal[0], ny-goal[1])
                    heapq.heappush(queue, (priority, next_cell))
        return []


def display_map(xh, path, state):
    image = np.full((520, 520, 3), 235, np.uint8)
    def pixel(p):
        return int((p[0]+2.6)*100), int((2.6-p[1])*100)
    for cx, cy, sx, sy in MAP_BOXES:
        cv2.rectangle(image, pixel((cx-sx, cy+sy)), pixel((cx+sx, cy-sy)), (90, 90, 90), -1)
    cv2.circle(image, pixel(MAP_PILLAR[:2]), 25, (90, 90, 90), -1)
    if len(path) > 1:
        cv2.polylines(image, [np.array([pixel(p) for p in path])], False, (0, 170, 0), 2)
    cv2.circle(image, pixel(xh[:2]), 8, (255, 80, 0), -1)
    tip = xh[:2]+.25*np.array([np.cos(xh[2]), np.sin(xh[2])])
    cv2.arrowedLine(image, pixel(xh[:2]), pixel(tip), (255, 80, 0), 2)
    cv2.putText(image, 'EKF + path | '+state, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    return image
