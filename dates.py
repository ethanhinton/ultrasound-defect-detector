import pydicom
from pathlib import Path
import os
import datetime

def convert_datetime(datestring):
    year = int(datestring[:4])
    if datestring[4] == 0:
        month = int(datestring[5])
    else:
        month = int(datestring[4:6])
    if datestring[6] == 0:
        day = int(datestring[7])
    else:
        day = int(datestring[6:8])
    return datetime.date(year,month,day)


def within_date_check(start,end,date):
    if start <= date <= end:
        return True
    else:
        return False

#user enters date boundaries in the form YYYYMMDD
#This code goes before the loop so you don't have to enter the date boundaries for every image
startdate = input("Start Date: ")
enddate = input("End Date: ")
converted_startdate = convert_datetime(startdate)
converted_enddate = convert_datetime(enddate)


# EXAMPLE CODE
#You will probably already have this in your code before your loop
path = Path.cwd() / 'Serious Defect'
pathstring = str(path)

for file in os.listdir(pathstring):
    #Try/Except statement needed so code isn't broken by non dicom files e.g. PNG images
    try:
        image = pydicom.read_file(os.path.join(path, file))
        converted_studydate = convert_datetime(image.StudyDate)

        if within_date_check(converted_startdate, converted_enddate, converted_studydate) == True:
            print("Between Dates!")
            #Put rest of code for analysing image here
        else:
            print("Not Between Dates!")
    except:
        print('File must be a DICOM image')




