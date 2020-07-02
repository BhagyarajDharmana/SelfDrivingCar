import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
from SelfDrivingCar.LaneDetection.LaneDetection import color_frame_pipeline
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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


        cap = cv2.VideoCapture(test_video)
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("drop.avi", four_cc, 20, (960, 540))
        # out = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc(*'XVID'),
        # fps=20.0, framesize=(960, 540))
        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, (960, 540))
                frame_buffer.append(color_frame)
                blend_frame = color_frame_pipeline(frame_buffer, solid_lines=True)
                out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)

            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
