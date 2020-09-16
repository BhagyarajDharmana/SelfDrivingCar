import random
import cv2
import os
import numpy as np
import matplotlib.image as mping
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from SelfDrivingCar.Advanced_LaneDetection.calibration_utils import calibrate_camera, un_distort
from SelfDrivingCar.Advanced_LaneDetection import utils
from SelfDrivingCar.Advanced_LaneDetection.pipeline import process_img
from SelfDrivingCar.Advanced_LaneDetection.MyVideoProcessor import MyVideoProcessor

if __name__ == '__main__':
    images_dir = '/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/camera_cal'
    out_path = '/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/camera_cal/'
    obj_points, img_points = calibrate_camera(images_dir, out_path)
    load_path = '/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/test_images/test' + str(
        random.randint(1, 6)) + '.jpg'
    img = mping.imread(load_path)

    # Uncomment this to see the out directly without visualising the steps
    # process_img_obj = process_img(img, obj_points, img_points, 5)
    # plt.imshow(process_img_obj['lane_img'])
    # plt.show()

    my_video_processor_object = MyVideoProcessor(obj_points, img_points, 5)
    output = 'project_video_output_v2_1.mp4'
    clip = VideoFileClip("/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/project_video.mp4")

    white_clip = clip.fl_image(my_video_processor_object.pipeline_function)

    white_clip.write_videofile(output, audio=False)
    HTML("""
    <video width="960" height="540" controls>
      <source src="{0}">
    </video>
    """.format(output))

    # Go through the entire process step by step
    # Step_1  Un_distort the  test image
    und_img = un_distort(img, obj_points, img_points)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(und_img)
    ax2.set_title('Undistorted Image', fontsize=30)
    f.show()
    f.savefig('/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/writeup_images'
              '/Dist_and_undist')

    # Step_2 Create a threshold binary image
    kernel_size = 5
    grad_x = utils.sobel_thresh(und_img, k_size=kernel_size, orient='x', threshold=(30, 100))
    s_binary = utils.s_select(und_img, threshold=(150, 255))
    mag_binary = utils.mag_thresh(und_img, k_size=kernel_size, threshold=(50, 100))
    combined = np.zeros_like(grad_x)
    combined[(((grad_x == 1) & (mag_binary == 1)) | (s_binary == 1))] = 1

    plt.imshow(combined, cmap='gray')
    plt.show()
    plt.savefig('/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/writeup_images'
                '/threshold')

    # Step_3 Apply perspective transform
    persp_obj = utils.perspective_transform(und_img)
    warped = persp_obj['warped']
    dst = persp_obj['dst']
    src = persp_obj['src']
    m_inv = persp_obj['m_inv']

    pts_1 = np.array(src, np.int32)
    pts_1 = pts_1.reshape((-1, 1, 2))
    cv2.polylines(img, [pts_1], True, (255, 0, 0), 3)

    pts_2 = np.array(dst, np.int32)
    pts_2 = pts_2.reshape((-1, 1, 2))
    cv2.polylines(warped, [pts_2], True, (255, 0, 0), 3)

    f1, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 10))
    ax3.imshow(img)
    ax3.set_title('Original Image', fontsize=30)
    ax4.imshow(warped)
    ax4.set_title('After Perspective Transform', fontsize=30)
    f1.savefig('/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/writeup_images'
               '/perspective')
    f1.show()

    # Step_4 Find lane line pixels in a binary warped image and fit 2nd order polynomial
    persp_obj_2 = utils.perspective_transform(combined)
    binary_warped = persp_obj_2['warped']

    curv_obj = utils.curvature_eval(binary_warped, n_windows=30, margin=40, min_pix=40)

    left_fit = curv_obj['left_fit']
    right_fit = curv_obj['right_fit']
    non_zero_x = curv_obj['non_zero_x']
    non_zero_y = curv_obj['non_zero_y']
    left_lane_idxs = curv_obj['left_lane_idxs']
    right_lane_idxs = curv_obj['right_lane_idxs']
    offset = curv_obj['offset']
    out_img = curv_obj['out_img']

    curvature = 0.5 * (curv_obj['left_curve_rad'] + curv_obj['right_curve_rad'])
    print('The radius of curvature is: ' + str(curvature))
    print('The car is offset ' + str(offset['offset_val']) + ' towards ' + offset['offset_dir'] + '.')

    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    out_img[non_zero_y[left_lane_idxs], non_zero_x[left_lane_idxs]] = [255, 0, 0]
    out_img[non_zero_y[right_lane_idxs], non_zero_x[right_lane_idxs]] = [0, 0, 255]
    f2, (ax4, ax5) = plt.subplots(1, 2, figsize=(18, 9))
    ax4.imshow(binary_warped, cmap='gray')
    ax4.set_title('Binary warped bird\'s eye view', fontsize=20)
    ax5.imshow(out_img)
    ax5.plot(left_fit_x, plot_y, color='yellow')
    ax5.plot(right_fit_x, plot_y, color='yellow')
    ax5.set_title('Lane lines identified', fontsize=20)
    f2.savefig('/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/writeup_images'
               '/sliding_window')
    f2.show()

    final_out = utils.map_color(m_inv, binary_warped, und_img, left_fit_x, right_fit_x, plot_y)
    final_out = utils.map_curv(final_out, curvature, offset)
    plt.imshow(final_out)
    plt.show()
