import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

img = mping.imread("LaneDetection/DataLib/test.jpg")
plt.imshow(img)
plt.show()
print("Image loaded of type", type(img), "and dimensions", img.shape)
X_size = img.shape[1]
Y_size = img.shape[0]
color_select = np.copy(img)
line_img = np.copy(img)

'''
 Defining Color criteria
 Setting thresholds for Color Selection 
 Mask pixels below thresholds
'''
r_threshold = 200
g_threshold = 200
b_threshold = 200
rgb_threshold = [r_threshold, g_threshold, b_threshold]
color_threshold = (img[:, :, 0] < rgb_threshold[0]) | (img[:, :, 1] < rgb_threshold[1]) | \
                  (img[:, :, 2] < rgb_threshold[2])


'''
Region of Interest
Defining Boundaries
Find Region inside the lines
'''
bottom_left = [0, 539]
bottom_right = [900, 300]
apex_centre = [475, 320]

fit_left = np.polyfit((bottom_left[0], apex_centre[0]), (bottom_left[1], apex_centre[1]), 1)
fit_right = np.polyfit((bottom_right[0], apex_centre[0]), (bottom_right[1], apex_centre[1]), 1)
bottom = np.polyfit((bottom_left[0], bottom_right[0]), (bottom_left[1], bottom_right[1]), 1)

XX, YY = np.meshgrid(np.arange(0, X_size), np.arange(0, Y_size))

region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * bottom[0] + bottom[1]))
color_select[color_threshold] = [0, 0, 0]
plt.imshow(color_select)
plt.show()
line_img[~color_threshold & region_thresholds] = [255, 0, 0]

plt.imshow(line_img)
plt.show()

