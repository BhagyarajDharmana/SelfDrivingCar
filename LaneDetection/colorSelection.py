import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

img = mping.imread("LaneDetection/DataLib/test.jpg")
plt.imshow(img)
plt.show()
print("Image loaded of type", type(img), "and dimensions", img.shape)

y_size = img.shape[0]
x_size = img.shape[1]

color_select = np.copy(img)
b = 200
r = 200
g = 200
rgb = [r, g, b]
threshold = (img[:, :, 0] < rgb[0]) | (img[:, :, 1] < rgb[1]) | (img[:, :, 2] < rgb[2])
color_select[threshold] = [0, 0, 0]
plt.imshow(color_select)
plt.show()
