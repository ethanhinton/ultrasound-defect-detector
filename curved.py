import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
from skimage import measure

file = os.path.join(Path.cwd(), '99A15X6063 RK99A1742034')

image = pydicom.read_file(file)
pixels = image.pixel_array
pixels = pixels[40:,:]
bw = (pixels > 0)
np.set_printoptions(threshold=sys.maxsize)

labels = measure.label(bw, connectivity=1)
properties = measure.regionprops(labels)

# empty area list to add to and then find the biggest area

maxArea = 0
maxIndex = 0

for prop in properties:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
    if prop.area > maxArea:
        maxArea = prop.area
        maxIndex = prop.label

bboxCoord = properties[maxIndex - 1].bbox
minx = bboxCoord[1]
miny = bboxCoord[0]
maxx = bboxCoord[3]
maxy = bboxCoord[2]
print(minx)
print(miny)
print(maxx)
print(maxy)


pixels = pixels[miny:maxy, minx:maxx]
print(pixels.shape)
plt.imshow(pixels)
plt.show()
