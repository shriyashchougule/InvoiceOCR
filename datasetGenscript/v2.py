import numpy as np
import cv2
import string
from collections import namedtuple
from random import randrange
import random
import os
# Image Gen
#img = np.zeros((900,1500,3), np.uint8)


""" The Plan ...
while n = 2000 #for 20K images
for read images 1 to 10
	add_table_headings
	add_table_entries
	add_Nature Colonm
	add_dimentions
	add_otherDetails
	add_surrounding text
	generate_sample_croping and_adding noise
	save sample
"""

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def read_templates(folderPath):
    images = []
    for filename in os.listdir( folderPath ):
        img = cv2.imread( os.path.join( folderPath, filename),0)
        if img is not None:
            images.append(img)
    return images

def preProcess(img):
    img = 255 -img
    img = img*0.8
    img = img.astype(int)
    img = 255 - img
    img = cv2.blur(img,(2,2))
    
    return img



def add_surrounding_text(img):
    rowRange = (725,770)
    colRange = (90, 200)
    strechRange = (620,1100)
    lineRange = (1,4)
    wordRange = (5, 20)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange)

    # handling section    
    rowRange = (760,800)
    colRange = (700, 1000)
    strechRange = (1200,1250)
    img =   add_handling_information(img, rowRange, colRange, strechRange, wordRange=(1,3))
    
    # Informative section    
    rowRange = (665,670)
    colRange = (655, 800, 100)
    strechRange = (1200,1250)
    lineRange   = (3,5)
    wordRange=(10,30)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.4)
    
    # Destination Airport section    
    rowRange = (670,700)
    colRange = (80, 110)
    strechRange = (300,301)
    lineRange   = (1,2)
    wordRange=(2,3)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.6)
    
    # Source Airport section    
    rowRange = (565,615)
    colRange = (80, 120)
    strechRange = (300,301)
    lineRange   = (1,2)
    wordRange=(2,3)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.6)

    # Source Airport section    
    rowRange = (675,715)
    colRange = (290, 330)
    strechRange = (450,451)
    lineRange   = (1,2)
    wordRange=(2,3)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.7)
    
    # Source Airport section    
    rowRange = (675,715)
    colRange = (460, 500)
    strechRange = (650,651)
    lineRange   = (1,2)
    wordRange=(2,3)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.7)
    
    # Source Airport section    
    rowRange = (625,635)
    colRange = (80, 120)
    strechRange = (1200,1201)
    lineRange   = (2,3)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=5)
    
    # Source Airport section    
    rowRange = (1230,1240)
    colRange = (80, 100,10)
    strechRange = (600,1200, 50)
    lineRange   = (2,5)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=4)

    # Source Airport section    
    rowRange = (1290,1310)
    colRange = (70, 110, 10)
    strechRange = (600,1200,100)
    lineRange   = (2,3)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=4)
    
    rowRange = (1230,1310)
    colRange = (560, 700, 30)
    strechRange = (1250,1251)
    lineRange   = (2,3)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=10)
       
    return img
    
def add_handling_information(img, rowRange, colRange, strechRange, lineRange=(1,2), wordRange=(1,2), fontScale = None, spacing =1):
    cv2fontList = [0, 2, 3, 4, 6, 7] #Ref https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
    font                   = random.choice(cv2fontList)
    
    if fontScale is None: 
        fontScale              = random.uniform(0.5, 0.75)
    fontColor              = 0
    lineType               = 2
    
    startRow = randrange(*rowRange)
    startCol = randrange(*colRange)
    textMaxStrech = randrange(*strechRange)
    lines = randrange(*lineRange)

    discription = ''

    maxStrechOfTextBox = 0
    baseline = 0
    
    for l in range(1,lines+1):
        lineNotReady = True
        while lineNotReady:
            words = randrange(*wordRange)
            #print("words: ", words)
            line = ''
            for w in range(1, words+1):
                letters = randrange(1, 12)
                #print(w, letters)
                myword = "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k = letters))
                line += myword
                line += " "*spacing
                #print("line :", line)    
            (label_width, label_height), baseline = cv2.getTextSize(line, font, fontScale*0.98, lineType)
            print(label_width)
            if (label_width+startCol) > textMaxStrech:
                continue
            else:
                maxStrechOfTextBox = max(maxStrechOfTextBox, label_width)
                lineNotReady = False
                discription += line
                discription += "\n"

    y = 0
    for i, line in enumerate(discription.split('\n')):
        (label_width, label_height), baseline = cv2.getTextSize(line, font, fontScale*0.98, lineType)
        y = startRow + i* (baseline+label_height)
        cv2.putText(img, line, (startCol, y), font, fontScale*0.98, fontColor, lineType)

    return img

def add_table_headings(img):
    pieces  = ["No of\npieces \n RCP", 84, 832]
    GW      = [" Gross \nWeight", 175, 838]
    Kglb    = ["Kg\nlb", 255, 832]
    rateC   = ["Rate Class", 310, 832]
    commodity = ["Commodity\nItem no", 310, 854]
    CW      = ["Chargeable\n   Weight   ", 436, 840]
    rateCrg = ["Rate     \n      Charge", 558, 835]
    total   = ["Total", 737, 840]
    nature  = ["Nature and Quantity of Goods\n (Incl. Dimensions or Volume)", 930, 840]

    cv2fontList = [0, 2, 3] #Ref https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
    font                   = random.choice(cv2fontList) 
    fontScale              = random.uniform(0.5, 0.75)
    fontColor              = 0
    lineType               = 2
    
    itemsToBePrinted = [pieces, GW, Kglb, rateC, commodity, CW, rateCrg, total, nature]
    print("font: ", font)
    for item in itemsToBePrinted:
        for i, line in enumerate(item[0].split('\n')):
            (label_width, label_height), baseline = cv2.getTextSize(line, font, 0.46, lineType)
            y = item[2] + i* (baseline+label_height)
            cv2.putText(img, line, (item[1], y), font, 0.46, fontColor, lineType)
    return img

templateFolderPath = "/home/catz/Downloads/templates for Datageneration"

templateList = read_templates(templateFolderPath)
n = 9
while n > 0:
    for tl in templateList:
        tl = ResizeWithAspectRatio(tl, width=1280)
        tl = add_table_headings(tl)
        #tl = preProcess(tl)
        tl = add_surrounding_text(tl)
        tl = add_table_entries(tl)
        tlcrop = tl[int(tl.shape[0]*0.38):int(tl.shape[0]*0.72),:]
        cv2.imshow("image", tlcrop)

        cv2.imshow("image2", tl)
        cv2.waitKey(3)



