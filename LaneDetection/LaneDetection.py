import numpy as np
import cv2


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
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    imshape = img.shape
    ysize = imshape[0]
    ytop = int(0.6 * ysize)  # need y coordinates of the top and bottom of left and right lane
    ybtm = int(ysize)  # to calculate x values once a line is found

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = float(((y2 - y1) / (x2 - x1)))
            if (slope > 0.5):  # if the line slope is greater than tan(26.52 deg), it is the left line
                x_left.append(x1)
                x_left.append(x2)
                y_left.append(y1)
                y_left.append(y2)
            if (slope < -0.5):  # if the line slope is less than tan(153.48 deg), it is the right line
                x_right.append(x1)
                x_right.append(x2)
                y_right.append(y1)
                y_right.append(y2)
    # only execute if there are points found that meet criteria, this eliminates borderline cases i.e. rogue frames
    if (x_left != []) & (x_right != []) & (y_left != []) & (y_right != []):
        left_line_coeffs = np.polyfit(x_left, y_left, 1)
        left_xtop = int((ytop - left_line_coeffs[1]) / left_line_coeffs[0])
        left_xbtm = int((ybtm - left_line_coeffs[1]) / left_line_coeffs[0])
        right_line_coeffs = np.polyfit(x_right, y_right, 1)
        right_xtop = int((ytop - right_line_coeffs[1]) / right_line_coeffs[0])
        right_xbtm = int((ybtm - right_line_coeffs[1]) / right_line_coeffs[0])
        cv2.line(img, (left_xtop, ytop), (left_xbtm, ybtm), color, thickness)
        cv2.line(img, (right_xtop, ytop), (right_xbtm, ybtm), color, thickness)

        return img


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
