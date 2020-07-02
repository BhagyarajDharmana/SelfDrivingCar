import numpy as np
import cv2
from SelfDrivingCar.LaneDetection.Line import Line
import matplotlib.pyplot as plt


def region_of_interest(image, vertices):
    """
    Only keeps the region of the image defined by the polygon
    formed from vertices. The rest of the image is set to black
    """
    mask_image = np.zeros_like(image)
    ignore_mask_color = 255
    cv2.fillPoly(mask_image, [vertices], ignore_mask_color)
    masked_edges = cv2.bitwise_and(image, mask_image)
    return masked_edges, mask_image


def hough_lines_detection(canny_image, rho, theta, threshold, max_line_len, max_line_gap):
    """
    Perform hough transform on image which is output of canny edge detection
    Returns lines after Hough Transform
    """
    lines = cv2.HoughLinesP(canny_image, rho, theta, threshold, np.array([]), max_line_len, max_line_gap)
    return lines


def weighted_image(masked_image, img_color, alpha=0.8, beta=1., lamda=0.):
    """

    """
    return cv2.addWeighted(img_color, alpha, masked_image, beta, lamda)


def mean(line_list):
    """
    calculate mean of list
    """
    return float(sum(line_list)) / max(len(line_list), 1)


def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # initialize lists to hold line formula values
    b_left_values = []  # b of left lines
    b_right_values = []  # b of Right lines
    m_positive_values = []  # m of Left lines
    m_negative_values = []  # m of Right lines

    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # calculate slope and intercept
            m = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * m

            # threshold to check for outliers
            if m >= 0 and (m < 0.2 or m > 0.8):
                continue
            elif m < 0 and (m < -0.8 or m > -0.2):
                continue

            # seperate positive line and negative line slopes
            if m > 0:
                m_positive_values.append(m)
                b_left_values.append(b)
            else:
                m_negative_values.append(m)
                b_right_values.append(b)

    # Get image shape and define y region of interest value
    im_shape = img.shape
    y_max = im_shape[0]  # lines initial point at bottom of image
    y_min = 330  # lines end point at top of ROI

    # Get the mean of all the lines values
    avg_positive_m = mean(m_positive_values)
    avg_negative_m = mean(m_negative_values)
    avg_left_b = mean(b_left_values)
    avg_right_b = mean(b_right_values)

    # use average slopes to generate line using ROI endpoints
    if avg_positive_m != 0:
        x1_left = (y_max - avg_left_b) / avg_positive_m
        y1_left = y_max
        x2_left = (y_min - avg_left_b) / avg_positive_m
        y2_left = y_min

    if avg_negative_m != 0:
        x1_right = (y_max - avg_right_b) / avg_negative_m
        y1_right = y_max
        x2_right = (y_min - avg_right_b) / avg_negative_m
        y2_right = y_min

        # define average left and right lines
        cv2.line(img, (int(x1_left), int(y1_left)), (int(x2_left), int(y2_left)), color, thickness)  # avg Left Line
        cv2.line(img, (int(x1_right), int(y1_right)), (int(x2_right), int(y2_right)), color,
                 thickness)  # avg Right Line

        return img


"""
def compute_lane_from_candidates(candidates, img_shape):
    
    Computes solid  lines that approximates both road lanes
    :param candidates: selected Lines from hough Transform
    :param img_shape: Shape of the image after edge detection
    :return  Approximated road lane with solid lines
    
    b_left_values = []  # b of left lines
    b_right_values = []  # b of Right lines
    m_positive_values = []  # m of Left lines
    m_negative_values = []  # m of Right lines

    for line in candidates:
        for x1, y1, x2, y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # calculate slope and intercept
            m = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * m

            # threshold to check for outliers
            if m >= 0 and (m < 0.2 or m > 0.8):
                continue
            elif m < 0 and (m < -0.8 or m > -0.2):
                continue

            if m > 0:
                m_positive_values.append(m)
                b_left_values.append(b)
            else:
                m_negative_values.append(m)
                b_right_values.append(b)

    # Get the mean of all the lines values
    avg_positive_m = mean(m_positive_values)
    avg_negative_m = mean(m_negative_values)
    avg_left_b = mean(b_left_values)
    avg_right_b = mean(b_right_values)

    y_max = img_shape[0]
    y_min = 310

    if avg_positive_m != 0:
        x1_left = (y_max - avg_left_b) / avg_positive_m
        y1_left = y_max
        x2_left = (y_min - avg_left_b) / avg_positive_m
        y2_left = y_min

    if avg_negative_m != 0:
        x1_right = (y_max - avg_right_b) / avg_negative_m
        y1_right = y_max
        x2_right = (y_min - avg_right_b) / avg_negative_m
        y2_right = y_min

    left_lane = Line(x1_left, y1_left, x2_left, y2_left)
    right_lane = Line(x1_right, y1_right, x2_right, y2_right)

    return left_lane, right_lane
"""


def get_lane_lines(color_image, solid_lines=False):
    """
    This Function takes as an input color image and returns another color image as output.
    :param color_image: input image
    :param solid_lines: True only selected lines are returned. If False all detected lines are returned.
    :return: list of lines
    """
    # resize to 960,540
    color_image = cv2.resize(color_image, (960, 540))
    # Convert to gray scale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # Perform gaussian blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform Canny Edge Detection
    edge_image = cv2.Canny(blur_image, threshold1=100, threshold2=150)
    # Keep only region of interest
    vertices = np.array([[0, gray_image.shape[0]],
                         [410, 315],
                         [510, 310],
                         [gray_image.shape[1], gray_image.shape[0]]],
                        dtype=np.int32)

    masked_image, _ = region_of_interest(edge_image, vertices)
    # Perform hough Transform
    detected_lines = hough_lines_detection(masked_image,
                                           rho=1,
                                           theta=np.pi / 180,
                                           threshold=40,
                                           max_line_len=10,
                                           max_line_gap=2
                                           )
    line_img = np.copy(color_image) * 0
    # detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]
    if solid_lines:
        #  lane_lines = compute_lane_from_candidates(detected_lines, gray_image.shape)
        line_img = draw_lines(line_img, detected_lines, color=[255, 0, 0], thickness=12)
    else:
        for line in detected_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=12)

    return line_img


def color_frame_pipeline(frames, solid_lines=False):
    """
    Entry point for Lane detection oioeline. Takes as input a list of frames(RGB) and returns an image(RGB)
    with overide of infrared road lanes. len(frames) == 1 in case of a single image

    """
    is_video_clip = False
    if len(frames) > 0:
        is_video_clip = True
    for t in range(0, len(frames)):
        line_image = get_lane_lines(frames[t], solid_lines=solid_lines)

    img_color = frames[-1] if is_video_clip else frames[0]
    img_blend = weighted_image(line_image, img_color, alpha=0.8, beta=1, lamda=0)

    return img_blend
