import glob
import os.path as path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera(calib_images_dir, save_path, nx=9, ny=6):
    """
    Calibrate the camera given a directory containing calibration chessboards.
    :param calib_images_dir: directory containing chessboard frames
    :param save_path : Directory to save after drawing corners
    :param nx: no of corners in x direction
    :param ny: no of corners in y direction
    :return: calibration parameters
    """
    # Prepare object points like (0,0,0), (0,0,1), (2,0,0)........

    obj_p = np.zeros((nx * ny, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points for all images
    object_points = []  # 3D points in real space
    image_points = []  # 2D points in image plane

    # Make a list of calibration images
    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))
    # Step through the list and search for chess board images
    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pattern_found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if pattern_found:
            image_points.append(corners)
            object_points.append(obj_p)
            """
            img_out = cv2.drawChessboardCorners(img, (nx, ny), corners, pattern_found)
            write_name = save_path + filename.split('.')[0] + '_corners_found' + '.jpg'
            cv2.imwrite(write_name, img_out)
            plt.imshow(img_out)
            plt.show()
            
            """

    return object_points, image_points


def un_distort(img, object_points, image_points):
    """
         Un_distort a frame given camera matrix and distortion coefficients.
        :param img: input image
        :param object_points: camera matrix
        :param image_points: distortion coefficients
        :return: undistorted frame
        """
    h, w = img.shape[:2]
    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (w, h), None, None)
    frame_undistorted = cv2.undistort(img, mtx, dst, None, mtx)
    return frame_undistorted


if __name__ == '__main__':
    images_dir = '/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/camera_cal'
    out_path = '/home/intence/SelfDrivingCar/SelfDrivingCar/Advanced_LaneDetection/DataLib/camera_cal/'
    obj_points, img_points = calibrate_camera(images_dir, out_path)
    images = glob.glob(path.join(images_dir, 'calibration*.jpg'))
    for file in images:
        img = cv2.imread(file)
        u_img = un_distort(img, obj_points, img_points)
        plt.imshow(u_img)
        plt.show()

