import numpy as np
import cv2


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = np.float32(x1)
        self.x2 = np.float32(x2)
        self.y1 = np.float32(y1)
        self.y2 = np.float32(y2)

        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        # if self.x1 == self.x2:
        #     print([self.x1, self.x2])
        # else:
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def draw(self, img, color=[255, 0, 0], thickness=10):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
