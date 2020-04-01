import pydicom
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import os
from pathlib import Path
import sys
import statistics as st
import xlsxwriter as xl
import scipy.stats as stats

def cropImages(imageDir):
    #read in the dicom file
    image = pydicom.read_file(imageDir)

    #get the pixel information
    pixels1 = image.pixel_array
    pixels = pixels1[40:,:]

    #if rgb then header tage samples per pixels will be three
    if image[0x28,0x2].value==3:
        pixels = pixels[:,:,2]


    #convert to a black and white image
    bw = (pixels>0)

    #find connected white regions
    labels = measure.label(bw,connectivity=1)
    properties = measure.regionprops(labels)

    #empty area list to add to and then find the biggest area

    maxArea = 0
    maxIndex = 0

    for prop in properties:
        #print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
        if prop.area>maxArea:
            maxArea = prop.area
            maxIndex = prop.label

    bboxCoord = properties[maxIndex - 1].bbox
    miny = bboxCoord[0]

    if miny > int(bw.shape[0] / 5):
        bw = bw[:miny,:]
        labels = measure.label(bw, connectivity=1)
        properties = measure.regionprops(labels)
        maxArea = 0
        maxIndex = 0

        #loop over the connected white regions and select the largest region size
        for prop in properties:
            if prop.area>maxArea:
                maxArea = prop.area
                maxIndex = prop.label
        
    #crop the original image to the bounding box of the maximum white region

        bboxCoord = properties[maxIndex-1].bbox

    minx = bboxCoord[1]
    miny = bboxCoord[0]
    maxx = bboxCoord[3]
    maxy = bboxCoord[2]
    # h = maxy - miny
    # w = maxx - minx

    croppedImage = pixels[miny:maxy,minx:maxx]

    middle = []

    # Looks from the bottom of the image to the top to find noise below the reverb image and crop it out
    for n in range(croppedImage.shape[0]):
        # Creates a list with the pixel values from the middle column of the image
        middle.append(croppedImage[n, int((croppedImage.shape[1] - 1) / 2)])

    # Reverses the list so the pixel values are listed from the bottom of the image upwards
    middle.reverse()

    # Finds the first non-zero pixel value (noise)
    index = next((i for i, x in enumerate(middle) if x != 0), None)
    middle_cut = middle[index:]

    # Finds the next value that is less than pixel value 3 (hopefully the end of noise)
    index_final = next((i for i, x in enumerate(middle_cut) if x < 3), None)

    # If there is no next value that is less than 3, index_final = None and this next step will cause a TypeError, if this happens there is no noise so pixel_cut_point is set to 0
    try:
        pixel_cut_point = croppedImage.shape[0] - (index + index_final + 2)
    except TypeError:
        pixel_cut_point = 0

    # If the image is being cropped in the first 1/3rd of the image then it is probably cropping the reverb and not the noise so the initial image is assumed to have no noise
    # Otherwise, the noise is cropped off by the code below
    if pixel_cut_point > (1 / 3) * len(middle):
        croppedImage = croppedImage[:pixel_cut_point, :]
        h = pixel_cut_point
    else:
        croppedImage = croppedImage

    #crop again using the lineplot function to neaten up edges of image (remove mainly black space)
    x1 = crop_left_to_right(croppedImage)
    x2 = crop_right_to_left(croppedImage)
    y1 = crop_up_to_down(croppedImage)

    croppedImage2 = croppedImage[y1:,x1:x2]

    y2 = crop_down_to_up(croppedImage2)

    croppedImage2 = croppedImage2[:y2,:]

    #save cropped images
    image.PixelData = croppedImage2.tobytes()
    #save new height and width in the header
    image.Columns = x2 - x1
    image.Rows = y2 - y1
    #save header as one channel
    image[0x28,0x2].value=1
    return image
    

def line_plot(image, column):
    
    #image = pydicom.read_file(os.path.join(imageDir,file))
    np.set_printoptions(threshold=sys.maxsize)
    pixels = image.pixel_array
    plt.plot(pixels[:,column])
    plt.show()


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
                #print(len(values))
                values = []
                continue
            else:
                if h < 5:
                    return image.shape[0] - 1
                else:
                    return h
        h += 1
    return image.shape[0] - 1

def save_image(image, file, saveDirectory):
    #create output folder if it does not exist
    Path(saveDirectory).mkdir(parents=True, exist_ok=True)
    #save image to output folder
    image.save_as(os.path.join(saveDirectory,file))


def column_totals(image, row_start = 0, row_end = None):
    List = []
    #Sums the pixel values from the top half of each column and adds them to a list
    for n in range(image.shape[1]):
        List.append(sum(image[row_start:row_end, n]))
    return List

def cov(List):
    mean = st.mean(List)
    std_dev = np.std(List)
    return 100 * (std_dev / mean)

def defect_check(List, threshold):
    median = st.median(List)
    banding = []
    large_defect = []
    bad_columns = []
    g = 0
    for column in List:
        #Ignores the two columns at the left and right of the image as these have a lower pixel value and mess up the results
        if g == 0 or g + 1 == len(List) or g == 1 or g + 2 == len(List):
            g += 1
            continue
        #Adds column number to a list of defective columns if it's value is significantly below median column pixel value
        elif column < (median - (threshold * median)):
            bad_columns.append(g)
        #Prints statements if the previous column was defective and this column is not defective (means end of defect in image)
        elif len(bad_columns) != 0:
            #Notes defect as large element dropout if size of defect is above 5%
            if len(bad_columns) > 0.05 * len(List):
                print(f'There is a-- large -element dropout defect between columns {bad_columns[0]} and {bad_columns[-1]}')
                large_defect.append([bad_columns[0], bad_columns[-1]])
            #Otherwise just a normal banding defect
            else:
                print(f'There is a banding defect between columns {bad_columns[0]} and {bad_columns[-1]}')
                banding.append([bad_columns[0], bad_columns[-1]])
            #Resets list of defective columns so size of next defect can be found
            bad_columns = []
        g += 1


    return large_defect, banding


#Returns a list as a string
def list_as_string(list):
    string = ''
    for i in range(len(list)):
        if i + 1 == len(list):
            string += f'{list[i]}'
        else:
            string += f'{list[i]}, '
    return string

#Calculates the ideal with of a column so cells are wide enough to fit all of the text in
def table_cell_widths(data, headers):
    max_widths = []
    for column in range(len(headers)):
        max_width = len(headers[column]["header"])
        for row in data:
            if isinstance(row[column], str) == True:
                if len(row[column]) > max_width:
                    max_width = len(row[column])
            else:
                if len(str(row[column])) > max_width:
                    max_width = len(row[column])
        max_widths.append(max_width)
    return max_widths





#------------------------------------------------PROGRAM-------------------------------------------------------------

#Directory the code looks in to find image: Change this to the folder that the images are in
direct = Path.cwd() / 'Faulty'

#Used to loop and to print the loctaion of the image later on
string = str(direct)

#Filename of image: Use if testing a single image
#file = 'I2LBHP2A'

#Directory where new image is saved
savedirect = Path.cwd() / 'Output'
if not os.path.exists(savedirect):
    os.mkdir(savedirect)

#Executed code : Use this if you are testing on a single image, make sure you have the image name set as the 'file' variable above
# croppedimage = cropImages(os.path.join(direct, file))
# save_image(croppedimage, file, savedirect)
# print(banding_check(croppedimage))
# print(pen_depth_check(croppedimage))

#Loops over images and checks for banding and penetration depth issues, outputs results of tests to an excel spreadsheet
i = 0
n = 0
data = []

for file in os.listdir(string):
    n += 1
    print(string + '\\' + file)
    croppedimage = cropImages(os.path.join(direct, file))
    croppedimage_pixels = croppedimage.pixel_array
    save_image(croppedimage, file, savedirect)
    fifth = int(croppedimage_pixels.shape[0] / 5)
    half = int(croppedimage_pixels.shape[0] / 2)

    MAD = stats.median_absolute_deviation(column_totals(croppedimage_pixels))
    COV = cov(column_totals(croppedimage_pixels))
    x, y = defect_check(column_totals(croppedimage_pixels, fifth, half), 0.1)
    a = defect_check(column_totals(croppedimage_pixels, half), 0.3)[0]

    plt.gray()
    plt.imshow(croppedimage_pixels)
    plt.show()
    #sets up data from current file for excel output
    tempdata = [file,
                'Yes' if len(x) != 0 else 'No',
                'N/A' if len(x) == 0 else f'Between columns {list_as_string(x).replace("[","(").replace("]",")")}',
                'Yes' if len(y) != 0 else 'No',
                'N/A' if len(y) == 0 else f'Between columns {list_as_string(y).replace("[","(").replace("]",")")}',
                'Yes' if len(a) != 0 else 'No',
                'N/A' if len(a) == 0 else f'Between columns {list_as_string(a).replace("[","(").replace("]",")")}',
                MAD,
                COV]

    data.append(tempdata)
    i += 1
print(n, ' images were procesed')
#column headers for Excel
columns = [{'header': 'Image Name'},
           {'header': 'Element Dropout?'},
           {'header': 'Dropout columns'},
           {'header': 'Banding?  '},
           {'header': 'Banding columns'},
           {'header': 'Penetration Depth Defect?'},
           {'header': 'Pen Depth columns'},
           {'header': 'Median Absolute Deviation'},
           {'header': 'Coefficiant of Variation'}]


wb = xl.Workbook('Probe Defects.xlsx')
sheet1 = wb.add_worksheet()
sheet1.write('A1', 'This table summarises the defects found on probes analysed by lineplot.py')

sheet1.add_table('B3:J' + str(i + 3), {'data': data,'columns': columns})

for cell, width in enumerate(table_cell_widths(data, columns),1):
    sheet1.set_column(cell, cell, width)

wb.close()

os.system('start "excel.exe" "Probe Defects.xlsx"')



