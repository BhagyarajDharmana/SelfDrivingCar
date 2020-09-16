import numpy as np
from SelfDrivingCar.Advanced_LaneDetection.calibration_utils import calibrate_camera, un_distort
from SelfDrivingCar.Advanced_LaneDetection import utils


def process_img(img, obj_points, img_points, k_size, n_windows=40, margin=40, min_pix=40):
    """
    Pipeline function to detect lanes and transform it back on the road
    :param img: Input image
    :param obj_points: object points in §D space
    :param img_points: "D points in real space
    :param k_size: kernal size
    :param n_windows: no.of windows for sliding window algorithm
    :param margin : width of windows +/- margin
    :param min_pix : minimum no.of pixels found to recenter the window
    """
    # Correct the distortion
    und_img = un_distort(img, obj_points, img_points)

    # Apply each of the threshold functions
    grad_x = utils.sobel_thresh(und_img, k_size, orient='x', threshold=(30, 100))
    s_binary = utils.s_select(und_img, threshold=(150, 255))
    mag_binary = utils.mag_thresh(und_img, k_size, threshold=(50, 100))
    combined = np.zeros_like(grad_x)
    combined[(((grad_x == 1) & (mag_binary == 1)) | (s_binary == 1))] = 1

    # Apply Perspective Transform
    persp_obj = utils.perspective_transform(combined)
    warped = persp_obj['warped']
    m_inv = persp_obj['m_inv']

    # Find lane line pixels in a binary warped image and fit 2nd order polynomial
    # Find radius of curvature  and offset of vehicle from lane centre line

    curv_obj = utils.curvature_eval(warped, n_windows, margin, min_pix)

    left_fit = curv_obj['left_fit']
    right_fit = curv_obj['right_fit']
    left_lane_idxs = curv_obj['left_lane_idxs']
    right_lane_idxs = curv_obj['right_lane_idxs']
    left_curve_rad = curv_obj['left_curve_rad']
    right_curve_rad = curv_obj['right_curve_rad']
    offset = curv_obj['offset']
    curvature = 0.5 * (left_curve_rad + right_curve_rad)

    # overlay identified lane on original image
    plot_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    lane_img = utils.map_color(m_inv, warped, und_img, left_fit_x, right_fit_x, plot_y)
    lane_img = utils.map_curv(lane_img, curvature, offset)

    return {'left_fit': left_fit, 'right_fit': right_fit, 'plot_y':plot_y,
            'left_lane_idxs': left_lane_idxs, 'right_lane_idxs': right_lane_idxs,
            'left_curve_rad': left_curve_rad, 'right_curve_rad': right_curve_rad,
            'offset': offset, 'lane_img': lane_img, 'curvature': curvature,
            'warped': warped, 'm_inv': m_inv, 'und_img': und_img}
