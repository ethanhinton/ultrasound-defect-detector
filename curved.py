import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
from skimage import measure

class DICOMimage:

    def __init__(self, path):
        self.path = path
        self.image = pydicom.read_file(path)
        self.pixels = self.image.pixel_array

    def crop(self):
        pixels = self.pixels
        bw = (pixels > 0)
        labels = measure.label(bw, connectivity=1)
        properties = measure.regionprops(labels)
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


        self.pixels = pixels[miny:maxy, minx:maxx]
        self.image.PixelData = self.pixels.tobytes()

    def showimage(self):
        plt.imshow(self.pixels)
        plt.show()

    def pixelarray(self):
        np.set_printoptions(threshold=sys.maxsize)
        return self.pixels

def main():
    path = os.path.join(Path.cwd(), 'I2LBHP18')
    im1 = DICOMimage(path)
    print(im1.image[0x18,0x6011][0][0x18,0x6020].value)
    # im1.crop()
    # im1.showimage()

if __name__ == '__main__':
    main()

