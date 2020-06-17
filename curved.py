import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
from math import sqrt

class DICOMimage:


    def __init__(self, path):
        self.path = path
        self.data = pydicom.read_file(path)
        self.pixels = self.data.pixel_array

        # Extracts image location data from DICOM header in form (x1, y1, x2, y2, x(centre of image))
        self.region = (self.data[0x18, 0x6011][0][0x18, 0x6018].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601A].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601C].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601E].value,
                       self.data[0x18, 0x6011][0][0x18, 0x6020].value)


class curvedDICOMimage(DICOMimage):

    def __init__(self, path):
        super().__init__(self, path)
        # Finds the coordinates of the top two points of the curved image and the coordinates of the middle of the sector
        self.sectorcoords = self.find_top_values(), self.find_middle_value()


    def crop(self):
        pixels = self.pixels
        region = self.region
        self.pixels = pixels[region[1]:region[3], region[0]:region[2]]
        self.data.PixelData = self.pixels.tobytes()

    def find_top_values(self):
        xmiddle = self.region[4]
        height = self.region[1] + 1
        for index, pixel in enumerate(self.pixels[height, xmiddle:]):
            if pixel.all() != 0:
                return [height,xmiddle - index], [height, xmiddle + index]

    def find_middle_value(self):
        xmiddle = self.region[4]
        s_height = self.region[1]
        for index, pixel in enumerate(self.pixels[s_height:, xmiddle]):
            if pixel.all() != 0:
                return [s_height + index, xmiddle]


    def circle_centre(self):
        x1, x2 = self.sectorcoords[0]
        middle = self.sectorcoords[1]
        m = middle[0] - x1[0]
        l = middle[1] - x1[1]
        r1 = (l**2 + m**2) / (2*m)
        h1 = int(sqrt(r1**2 - l**2))
        return [middle[0] - m - h1 , middle[1]]


    def showimage(self):
        plt.imshow(self.pixels)
        plt.show()


    def pixelarray(self):
        np.set_printoptions(threshold=sys.maxsize)
        return self.pixels

    def refactor(self):
        pass

def main():
    path = os.path.join(Path.cwd(), 'I2LBHP18')
    im1 = curvedDICOMimage(path)
    print(im1.circle_centre())
    im1.crop()
    im1.showimage()
    print(im1.sectorcoords)

if __name__ == '__main__':
    main()

