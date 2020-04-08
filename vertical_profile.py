import pydicom
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import os
from pathlib import Path
import sys
import statistics as st
import scipy.stats as stats


def cropImages(imageDir):
    # read in the dicom file
    image = pydicom.read_file(imageDir)

    # get the pixel information
    pixels1 = image.pixel_array
    cutoff = int(0.08 * pixels1.shape[0])
    pixels = pixels1[cutoff:, :]

    # if rgb then header tage samples per pixels will be three
    if image[0x28, 0x2].value == 3:
        pixels = pixels[:, :, 2]

    # convert to a black and white image
    bw = (pixels > 0)

    # find connected white regions
    labels = measure.label(bw, connectivity=1)
    properties = measure.regionprops(labels)

    # empty area list to add to and then find the biggest area

    maxArea = 0
    maxIndex = 0

    for prop in properties:
        # print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
        if prop.area > maxArea:
            maxArea = prop.area
            maxIndex = prop.label

    bboxCoord = properties[maxIndex - 1].bbox
    minx = bboxCoord[1]
    miny = bboxCoord[0]
    maxx = bboxCoord[3]
    maxy = bboxCoord[2]

    if miny > int(bw.shape[0] / 6):
        bw = bw[:miny, :]
        labels = measure.label(bw, connectivity=1)
        properties = measure.regionprops(labels)
        maxArea = 0
        maxIndex = 0

        # loop over the connected white regions and select the largest region size
        for prop in properties:
            if prop.area > maxArea:
                maxArea = prop.area
                maxIndex = prop.label

        # crop the original image to the bounding box of the maximum white region

        bboxCoord = properties[maxIndex - 1].bbox

        minx_new = bboxCoord[1]
        miny_new = bboxCoord[0]
        maxx_new = bboxCoord[3]
        maxy_new = bboxCoord[2]

        if maxy_new - miny_new > 0.05 * pixels.shape[0]:
            croppedImage = pixels[miny_new:maxy_new, minx_new:maxx_new]
        else:
            croppedImage = pixels[miny:maxy, minx:maxx]
    else:
        croppedImage = pixels[miny:maxy, minx:maxx]

    # crop again using the lineplot function to neaten up edges of image (remove mainly black space)
    x2 = crop_right_to_left(croppedImage)
    y1 = crop_up_to_down(croppedImage)

    croppedImage2 = croppedImage[y1:, :x2]

    # y2 = crop_down_to_up(croppedImage2)

    # croppedImage2 = croppedImage2[:y2, :]

    width = int(croppedImage2.shape[1] / 20)
    mean_values = middle_values(croppedImage2, width)
    factor_constant = 15
    factor = int(len(mean_values) / factor_constant)
    condensed_list, original_list = condense(mean_values, factor)
    index_cut = cutoff_index(condensed_list, original_list, factor)

    croppedImage2 = croppedImage2[:index_cut, :]
    x1 = crop_left_to_right(croppedImage2)
    croppedImage2 = croppedImage2[:, x1:]

    # save cropped images
    image.PixelData = croppedImage2.tobytes()
    # save new height and width in the header
    image.Columns = croppedImage2.shape[1]
    image.Rows = croppedImage2.shape[0]
    # save header as one channel
    image[0x28, 0x2].value = 1
    return image

def discard(values, percentage):
    # Returns True if the % of numbers in a list (values) that are 0 is greater than the percentage set by the user
    if values.count(0) / len(values) >= percentage / 100:
        return True
    else:
        return False



def crop_left_to_right(image):
    h = 0

    while h <= image.shape[1] - 1:
        values = []
        n = 0
        while n <= image.shape[0] - 1:
            values.append(image[n,h])
            n += 1
        if discard(values, 50) is True:
            h += 1
            continue
        else:
            return h


def crop_right_to_left(image):
    h = -1

    while abs(h) <= image.shape[1]:
        values = []
        n = 0
        while n <= image.shape[0] - 1:
            values.append(image[n,h])
            n += 1
        if discard(values, 50) is True:
            h -= 1
            continue
        else:
            return image.shape[1] + h


def crop_up_to_down(image):
    h = 0

    while h <= image.shape[0] - 1:
        values = []
        n = 0
        while n <= image.shape[1] - 1:
            values.append(image[h,n])
            n += 1
        if discard(values, 50) is True:
            h += 1
            continue
        else:
            return h


def crop_down_to_up(image):
    h = 0
    values = []
    while abs(h) <= image.shape[0] - 1:
        n = 0
        while n <= image.shape[1] - 1:
            values.append(int(image[h,n]))
            n += 1
        if h % 4 == 0:
            if st.mean(values) >= 10:
                h += 1
                values = []
                continue
            else:
                if h < 5:
                    return image.shape[0] - 1
                else:
                    return h
        h += 1
    return image.shape[0] - 1

def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)
    return max_val - min_val


def condense(List, factor):
    new_length = int(len(List) / factor)
    new_list = []
    old_list = List[:]
    for i in range(new_length):
        mean = 0
        for element in range(factor):
            mean += List.pop(0) / factor
        new_list.append(mean)
    return new_list, old_list

def middle_values(image_pixels, width):
    middle = int(image_pixels.shape[1] / 2)
    mean_values = []
    for i in range(-(int(width / 2)),(int(width / 2))):
        values = []
        for pixel in range(image_pixels.shape[0]):
            values.append(image_pixels[pixel, middle + i] / width)
        if mean_values == []:
            mean_values = values
        else:
            for pixel in range(len(mean_values)):
                mean_values[pixel] += values[pixel]
    return mean_values


def cutoff_index(condensed_list, original_list, factor):
    condensed_list.reverse()
    for index, value in enumerate(condensed_list):
        try:
            if condensed_list[index + 1] - value > 0.1 * minmax(original_list):
                new_index = len(original_list) - (index * factor) - 1
                break
        except IndexError:
            print('no cut-off point found')
            return None
    return new_index
path = Path.cwd() / 'Linear All'
string = str(path)
print(string)
#dir = os.path.join(path, file)

for file in os.listdir(string):
    print(string + '\\' + file)
    croppedimage = cropImages(os.path.join(path, file))
    croppedimage_pixels = croppedimage.pixel_array
    #save_image(croppedimage, file, savedirect)

    plt.gray()
    plt.imshow(croppedimage_pixels)
    plt.show()