from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as D
import car_centroid as cc


class CarsTracker:
    def __init__(self, fps, counter_coord=None, counter_is_vertical=False, max_frames_disappear=50):
        # self.next_id = 0
        self.passed_cars = set()
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.fps = fps

        self.counter_coord = counter_coord
        self.counter_is_vertical = counter_is_vertical
        self.max_frames_disappear = max_frames_disappear

    def set_border(self, counter_coord):
        self.counter_coord = counter_coord

    def register_obj(self, centroid):
        x, y = centroid
        car = cc.CarCentroid(x, y, self.fps)
        self.objects[car.id] = car
        self.disappeared[car.id] = 0

    def deregister_obj(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, bounding_boxes):
        if len(bounding_boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self._set_as_disappeared(obj_id)

            return self.objects

        input_centroids = np.zeros((len(bounding_boxes), 2), dtype=int)
        for i, box in enumerate(bounding_boxes):
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            mid_x = int((x1 + x2) / 2.0)
            mid_y = int((y1 + y2) / 2.0)

            input_centroids[i] = mid_x, mid_y

        if len(self.objects) == 0:
            for i in range(input_centroids.shape[0]):
                self.register_obj(input_centroids[i])
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = []

            for car in self.objects.values():
                obj_centroids.append((car.x, car.y))

            obj_centroids = np.array(obj_centroids)
            dist = D.cdist(obj_centroids, input_centroids)

            rows = dist.min(axis=1).argsort()
            cols = dist.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                obj_id = obj_ids[row]
                x, y = input_centroids[col]
                car = self.objects[obj_id]
                car.update_position((x, y))

                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(dist.shape[0])).difference(used_rows)
            unused_cols = set(range(dist.shape[1])).difference(used_cols)

            if dist.shape[0] >= dist.shape[1]:
                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self._set_as_disappeared(obj_id)
            else:
                for col in unused_cols:
                    self.register_obj(input_centroids[col])

        self._count_passed()

        return self.objects

    def count_passed_cars(self):
        return len(self.passed_cars)

    def _count_passed(self):
        if self.counter_coord is None:
            print("Cannot count passed cars: Border wasn't set")
            return 0

        for obj_id, centroid in self.objects.items():
            if obj_id in self.passed_cars:
                continue

            car_x, car_y = centroid.x, centroid.y

            car_pos = car_y
            if self.counter_is_vertical:
                car_pos = car_x

            if car_pos > self.counter_coord:
                self.passed_cars.add(obj_id)

    def _set_as_disappeared(self, obj_id):
        self.disappeared[obj_id] += 1
        if self.disappeared[obj_id] > self.max_frames_disappear:
            self.deregister_obj(obj_id)
