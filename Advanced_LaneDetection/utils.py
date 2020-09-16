import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping


def sobel_thresh(img, k_size=3, orient='x', threshold=(0, 255)):
    """
    Function that applies Sobel X or Y
    :param img: Input image where sobel filter should be applied
    :param k_size: Kernel size to smothen the gradient
    :param orient: Direction odf orientation  'X' or 'Y'
    :param threshold: Limits of the pixel values
    :Returns Sx_binary: Mask as binary output image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return sx_binary


def mag_thresh(img, k_size=3, threshold=(0, 255)):
    """
        Function that applies Sobel X and Y and computes magnitude
        :param img: Input image where sobel filter should be applied
        :param k_size: Kernel size to smothen the gradient
        :param threshold: Limits of the pixel values
        Returns Sx_binary: Mask as binary output image
        """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return sx_binary


def dir_thresh(img, k_size=3, threshold=(0, np.pi / 2)):
    """
            Function that applies Sobel X and Y and computes direction magnitude
            :param img: Input image where sobel filter should be applied
            :param k_size: Kernel size to smothen the gradient
            :param threshold: Limits of the direction values
            Returns Sx_binary: Mask as binary output image
            """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)
    abs_graddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(abs_graddir)
    binary_output[(abs_graddir >= threshold[0]) & (abs_graddir <= threshold[1])] = 1

    return binary_output


def s_select(img, threshold=(0, 255)):
    """
    Function that thresholds the the s_chanel of HLS image
    :param img: Input image
    :param threshold: Limits of the pixel threshold values
    :Returns binary Image of threshold result
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > threshold[0]) & (s_channel <= threshold[1])] = 1
    return binary_output


def h_select(img, threshold=(0, 255)):
    """
    Function that thresholds the the h_chanel of HLS image
    :param img: Input image
    :param threshold: Limits of the pixel threshold values
    :Returns binary Image of threshold result
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    binary_output = np.zeros_like(h_channel)
    binary_output[(h_channel > threshold[0]) & (h_channel <= threshold[1])] = 1
    return binary_output


def perspective_transform(img):
    """
    Function for converting to birds eye view
    :param img : Input image which needs to be transformed
    :Returns birds eye view image
    """
    size_x = img.shape[0]
    size_y = img.shape[1]
    src = np.float32([[588, 470], [245, 719], [1142, 719], [734, 470]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, (size_y, size_x), flags=cv2.INTER_LINEAR)

    return {'warped': warped, 'dst': dst, 'src': src, 'm_inv': m_inv}


def curvature_eval(binary_img, n_windows=30, margin=50, min_pix=50):
    """
    Function evaluates curvature and offset for given frame using sliding window method
    :param binary_img: Input warped binary image after perspective transform
    :param n_windows : no.of sliding windows
    :param margin : width of windows +/- margin
    :param min_pix : minimum no.of pixels found to recenter the window
    : Return: Curvature
    """
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    left_base_x = np.argmax(histogram[:midpoint])
    right_base_x = np.argmax(histogram[midpoint:]) + midpoint

    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
    window_height = np.int(binary_img.shape[0] // n_windows)

    non_zero = binary_img.nonzero()
    non_zero_x = np.array(non_zero[1])
    non_zero_y = np.array(non_zero[0])

    left_x_current = left_base_x
    right_x_current = right_base_x

    left_lane_idxs = []
    right_lane_idxs = []

    for window in range(n_windows):
        window_y_low = binary_img.shape[0] - ((window + 1) * window_height)
        window_y_high = binary_img.shape[0] - (window * window_height)
        window_x_left_low = left_x_current - margin
        window_x_left_high = left_x_current + margin
        window_x_right_low = right_x_current - margin
        window_x_right_high = right_x_current + margin

        cv2.rectangle(out_img, (window_x_left_low, window_y_low),
                      (window_x_left_high, window_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (window_x_right_low, window_y_low),
                      (window_x_right_high, window_y_high), (0, 255, 0), 2)

        good_left_idx = ((non_zero_y >= window_y_low) & (non_zero_y < window_y_high) & (
                non_zero_x >= window_x_left_low) & (non_zero_x < window_x_left_high)).nonzero()[0]
        good_right_idx = ((non_zero_y >= window_y_low) & (non_zero_y < window_y_high) & (
                non_zero_x >= window_x_right_low) & (non_zero_x < window_x_right_high)).nonzero()[0]

        left_lane_idxs.append(good_left_idx)
        right_lane_idxs.append(good_right_idx)

        # Recent window on next window on mean pixel position
        if len(good_left_idx) > min_pix:
            left_x_current = np.int(np.mean(non_zero_x[good_left_idx]))

        if len(good_right_idx) > min_pix:
            right_x_current = np.int(np.mean(non_zero_x[good_right_idx]))

    # Concatenate array of indices
    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    # Extract Left and right pixel positions

    left_x = non_zero_x[left_lane_idxs]
    right_x = non_zero_x[right_lane_idxs]
    left_y = non_zero_y[left_lane_idxs]
    right_y = non_zero_y[right_lane_idxs]

    # Fit Second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Conversions from pixels to meters
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    # Fit new polynomials to x and y in real world space
    left_fit_cr = np.polyfit((left_y * ym_per_pix), (left_x * xm_per_pix), 2)
    right_fit_cr = np.polyfit((right_y * ym_per_pix), (right_x * xm_per_pix), 2)

    # calculate radius of curvature
    y_eval = binary_img.shape[0] - 1
    left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    offset_val = xm_per_pix * 0.5 * (binary_img.shape[1] - (left_base_x + right_base_x))

    if offset_val < 0:
        offset_dir = 'left'
    else:
        offset_dir = 'right'
    offset = {'offset_val': offset_val, 'offset_dir': offset_dir}

    return {'left_fit': left_fit, 'right_fit': right_fit,
            'non_zero_x': non_zero_x, 'non_zero_y': non_zero_y,
            'left_lane_idxs': left_lane_idxs, 'right_lane_idxs': right_lane_idxs,
            'left_curve_rad': left_curve_rad, 'right_curve_rad': right_curve_rad,
            'left_fit_cr': left_fit_cr, 'right_fit_cr': right_fit_cr,
            'offset': offset, 'out_img': out_img}


def map_color(m_inv, warped, undist, left_fit_x, right_fit_x, plot_y):
    """
    Function that projects identified lanes back on to the road
    """
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_wrap = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 0))

    new_warp = cv2.warpPerspective(color_wrap, m_inv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)

    return result


def map_curv(img, curvature, offset):
    """
    Function to put text on the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    offset_val = offset['offset_val']
    offset_dir = offset['offset_dir']
    curv_text = 'Radius of curvature is: ' + str(curvature) + ' m'
    offset_text = 'Car is offset: ' + str(abs(offset_val)) + ' m towards ' + offset_dir
    cv2.putText(img, curv_text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, offset_text, (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img


