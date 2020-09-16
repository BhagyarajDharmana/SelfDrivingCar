import collections
import numpy as np
from SelfDrivingCar.Advanced_LaneDetection.pipeline import process_img
from SelfDrivingCar.Advanced_LaneDetection import utils


class MyVideoProcessor(object):

    def __init__(self, o_points, i_points, k_size):
        self.o_points = o_points
        self.i_points = i_points
        self.k_size = k_size
        # frame count
        self.count = 0
        # values of the last 10 fits of the line
        self.past_frames_left = collections.deque(maxlen=10)
        self.past_frames_right = collections.deque(maxlen=10)

        # values of fits of the line for previous frame
        self.last_fit_left = []
        self.last_fit_right = []

        # curvature for previous frame
        self.left_curve_rad = []
        self.right_curve_rad = []
        self.curvature = []

        # offset for previous frame
        self.offset = []

        # polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None

    def pipeline_function(self, frame):
        # your lane detection pipeline
        if self.count < 10:
            pipeline = process_img(frame, self.o_points, self.i_points, self.k_size)
            self.past_frames_left.append(pipeline['left_fit'])
            self.past_frames_right.append(pipeline['right_fit'])
            if self.count == 9:
                self.last_fit_left = pipeline['left_fit']
                self.last_fit_right = pipeline['right_fit']
                self.left_curve_rad = pipeline['left_curve_rad']
                self.right_curve_rad = pipeline['right_curve_rad']
                self.curvature = pipeline['curvature']
                self.offset = pipeline['offset']
                self.best_fit_left = np.mean(self.past_frames_left, axis=0)
                self.best_fit_right = np.mean(self.past_frames_right, axis=0)
            self.count += 1
            return pipeline['lane_img']
        else:
            previous_left_fit = self.last_fit_left
            previous_right_fit = self.last_fit_right
            previous_left_curv = self.left_curve_rad
            previous_right_curv = self.right_curve_rad
            previous_curvature = self.curvature
            previous_offset = self.offset
            avg_left_fit = self.best_fit_left
            avg_right_fit = self.best_fit_right
            y_delx = frame.shape[0] - 1
            prev_delx = abs(avg_left_fit[0] * y_delx ** 2 + avg_left_fit[1] * y_delx + avg_left_fit[2] \
                            - avg_right_fit[0] * y_delx ** 2 + avg_right_fit[1] * y_delx + avg_right_fit[2])

            # process and evaluate vals for current frame
            pipeline = process_img(frame, self.o_points, self.i_points, self.k_size)
            und_img = pipeline['und_img']
            current_left_fit = pipeline['left_fit']
            current_right_fit = pipeline['right_fit']
            current_left_curv = pipeline['left_curve_rad']
            current_right_curv = pipeline['right_curve_rad']
            current_curvature = pipeline['curvature']
            current_offset = pipeline['offset']
            current_delx = abs(current_left_fit[0] * y_delx ** 2 + current_left_fit[1] * y_delx + current_left_fit[2] \
                               - current_right_fit[0] * y_delx ** 2 + current_right_fit[1] * y_delx + current_right_fit[
                                   2])

            # perform sanity checks
            if (current_left_fit.shape[0] == 0) | (current_right_fit.shape[0] == 0) | \
                    (abs(current_curvature) > 10000) | (abs(current_curvature) < 800) | (current_delx > 1550) | (
                    current_delx < 1100):
                current_left_fit = avg_left_fit
                current_right_fit = avg_right_fit
                current_curvature = previous_curvature
                current_offset = previous_offset

            self.past_frames_left.append(current_left_fit)
            self.past_frames_right.append(current_right_fit)
            current_avg_left_fit = np.mean(self.past_frames_left, axis=0)
            current_avg_right_fit = np.mean(self.past_frames_right, axis=0)
            self.best_fit_left = current_avg_left_fit
            self.best_fit_right = current_avg_right_fit
            self.curvature = current_curvature
            self.offset = current_offset

            m_inv = pipeline['m_inv']
            warped = pipeline['warped']
            plot_y = pipeline['plot_y']
            left_fitx = current_avg_left_fit[0] * plot_y ** 2 + current_avg_left_fit[1] * plot_y + current_avg_left_fit[2]
            right_fitx = current_avg_right_fit[0] * plot_y ** 2 + current_avg_right_fit[1] * plot_y + current_avg_right_fit[2]
            lane_img = utils.map_color(m_inv, warped, und_img, left_fitx, right_fitx, plot_y)
            lane_img = utils.map_curv(lane_img, current_curvature, current_offset)
            self.count += 1
            return lane_img
