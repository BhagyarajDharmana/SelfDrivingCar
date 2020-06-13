import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2

img = mping.imread("LaneDetection/DataLib/exit_ramp.jpeg")
plt.imshow(img)
plt.show()

grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(grey, cmap='gray')
plt.show()

grey_blur = cv2.GaussianBlur(grey, (3, 3), 1)
plt.imshow(grey_blur, cmap='gray')
plt.show()

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(grey_blur, low_threshold, high_threshold)
plt.imshow(edges, cmap='Greys_r')
plt.show()

# 'Define a four sided polygon'
mask = np.zeros_like(edges)
ignore_mask_color = 255

im_shape = img.shape
vertices = np.array([[0, im_shape[0]], [450, 290], [500, 290], [im_shape[1], im_shape[0]]])
cv2.fillPoly(mask, [vertices], ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
'''
Define the Hough Transform parameters
Implementing Hough transform on Canny edges
'''
rho = 1
theta = np.pi / 180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(img) * 0
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
color_edges = np.dstack((edges, edges, edges))
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(combo)
plt.show()
