import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
from SelfDrivingCar.LaneDetection.LaneDetection import color_frame_pipeline
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    result = color_frame_pipeline([image], solid_lines=True)
    return result


if __name__ == "__main__":
    test_images_dir = "LaneDetection/DataLib/test_images/"
    test_images = os.listdir(test_images_dir)
    test_video_dir = "LaneDetection/DataLib/test_videos/"
    test_videos = os.listdir(test_video_dir)

    for test_image in test_images:
        print("Processing {}".format(test_image))
        in_image = cv2.cvtColor(cv2.imread((test_images_dir + test_image), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
        out_path = "LaneDetection/DataLib/output_images/" + test_image
        out_image = color_frame_pipeline([in_image], solid_lines=True)
        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
        plt.imshow(out_image)
        plt.show()

    for test_video in test_videos:
        print("Processing {}".format(test_video))
        out_path = "LaneDetection/DataLib/output_videos/" + test_video

        clip1 = VideoFileClip(test_video_dir + test_video)
        out_clip = clip1.fl_image(process_image)
        out_clip.write_videofile(out_path, audio=False)
