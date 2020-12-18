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

TableLable = 50
NatureLable = 150
DimentionLable = 150
otherLable = 100

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

def preProcess(img, gamma):

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    img = cv2.LUT(img, lookUpTable)
    
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
    rowRange = (1240,1250)
    colRange = (80, 100,10)
    strechRange = (600,1200, 50)
    lineRange   = (2,5)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=4)

    # Source Airport section    
    rowRange = (1310,1330)
    colRange = (70, 110, 10)
    strechRange = (600,1200,100)
    lineRange   = (2,3)
    wordRange=(3,50)
    img =   add_handling_information(img, rowRange, colRange, strechRange, lineRange, wordRange, fontScale = 0.42, spacing=4)
    
    rowRange = (1240,1320)
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
    fontColor              = random.uniform(2, 50)
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
            #print(label_width)
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

def add_text_with_mask(img, mask, label, rowRange, colRange, strechRange, lineRange=(1,2), wordRange=(1,2),  fontScale = None, spacing =1):
    cv2fontList = [0, 2, 3, 4, 6, 7] #Ref https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
    font                   = random.choice(cv2fontList)
    
    if fontScale is None: 
        fontScale              = random.uniform(0.5, 0.75)
    fontColor              =  random.uniform(2, 50)
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
            #print(label_width)
            if (label_width+startCol) > textMaxStrech:
                continue
            else:
                maxStrechOfTextBox = max(maxStrechOfTextBox, label_width)
                lineNotReady = False
                discription += line
                if l < lines:
                    discription += "\n"


    y = 0
    for i, line in enumerate(discription.split('\n')):
        if line != '\n':
            (label_width, label_height), baseline = cv2.getTextSize(line, font, fontScale*0.98, lineType)
            y = startRow + i* (baseline+label_height)
            cv2.putText(img, line, (startCol, y), font, fontScale*0.98, fontColor, lineType)

            topLeftCorner      = (startCol-5,  y - (baseline+ label_height + 5))
            bottomRightCorner  = (startCol + label_width + 5,   y)
            cv2.rectangle(mask, topLeftCorner, bottomRightCorner, label, -1)

    endOfBox = y
    return (img, mask, endOfBox)


    
def add_table_entries(img):
    KgOptions = ["K", "k", "lb", ""]
    RCOptions = ["Q", "q", "Q", "", "Q", "q", "Q", ""]
    CCOptions = ["", "", "M", "", "", "N", "", "", "K", "", "", ""]

    ratcCOptions = ["AS AGREED", "pay", "pay", "pay"]

    pieces = randrange(1,140)
    GW     = randrange(1,1000)
    Kg     = random.choice(KgOptions)
    RC     = random.choice(RCOptions)
    CC     = random.choice(CCOptions)
    CW     = randrange(1,240)

    rateC  = "AS AGREED"
    if random.choice(ratcCOptions) == "pay":
	    rateC  = randrange(1,140)

    Total = "AS AGREED"
    if rateC != "AS AGREED":
	    Total = rateC * CW

    Nature = "one\ntwo\nthree"


    #font                   = cv2.FONT_HERSHEY_SIMPLEX
    cv2fontList = [0, 2, 3, 4] #Ref https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html

    font                   = random.choice(cv2fontList) 
    fontScale              = random.uniform(0.5, 0.75)
    fontColor              = random.uniform(10, 50)
    lineType               = 2


    rowStart = randrange(913, 1001)

    piecesStart = randrange (78, 130)
    GWStart = randrange(170,215)
    KgStart = randrange(262,272)
    RCStart = randrange(287,297)
    CCStart = randrange(330, 381)
    CWStart = randrange(445,490)

    rateCStart = 570
    if rateC != "AS AGREED":
	    rateCStart = randrange(560,630)

    TotalStart = 720
    if rateC != "AS AGREED":
	    TotalStart = randrange(720,800)

    # No.of peices
    cv2.putText(img, str(pieces), (piecesStart, rowStart), font, fontScale, fontColor, lineType)
    # gross weight
    cv2.putText(img, str(GW), (GWStart, rowStart), font, fontScale, fontColor, lineType)
    # kg
    cv2.putText(img, Kg, (KgStart, rowStart), font, fontScale, fontColor, lineType)
    # RC
    cv2.putText(img, RC, (RCStart, rowStart), font, fontScale, fontColor, lineType)
    # CC
    cv2.putText(img, CC, (CCStart, rowStart), font, fontScale, fontColor, lineType)
    # CW
    cv2.putText(img, str(CW), (CWStart, rowStart), font, fontScale, fontColor, lineType)
    # rateC
    cv2.putText(img, str(rateC), (rateCStart, rowStart), font, fontScale, fontColor, lineType) 
    # Total
    cv2.putText(img, str(Total), (TotalStart, rowStart), font, fontScale, fontColor, lineType)
    
    # Step 2 Create Segmentation Mask for text
    segMask = np.zeros((img.shape), np.uint8)

    # Strech of the last word i.e value in the Total column 
    (label_width, label_height), baseline = cv2.getTextSize(str(Total), font, fontScale, lineType)

    topLeftCorner      = (piecesStart-5,  rowStart - (baseline+ label_height + 5))
    bottomRightCorner  = (TotalStart + label_width + 5,  rowStart + 5)

    cv2.rectangle(segMask, topLeftCorner, bottomRightCorner, TableLable, -1)

    tableBoxEndRow = rowStart + 40

    on = [1]
    off = [0]
    rowRange    = (900, 990)
    colRange    = (850, 910)
    strechRange = (1150, 1230)
    lineRange   = (1,6)
    wordRange   = (3, 7)

    img, segMask,eob =add_text_with_mask(img, segMask, NatureLable,rowRange, colRange, strechRange, lineRange, wordRange,  spacing =1)

    rowRange    = (tableBoxEndRow, 1100)
    colRange    = (60, 450)
    strechRange = (550, 750)
    lineRange   = (1,4)
    wordRange   = (3, 20)  
    puttext     = random.choice(on*75 + off*25)
    if puttext:
        img, segMask,e =add_text_with_mask(img, segMask,otherLable, rowRange, colRange, strechRange, lineRange, wordRange,  spacing =1)

    rowRange    = (tableBoxEndRow, 1100)
    colRange    = (200, 710)
    strechRange = (800, 801)
    lineRange   = (1,3)
    wordRange   = (2, 4)
    puttext     = random.choice(on*50 + off*50)
    if puttext:
        img, segMask,e =add_text_with_mask(img, segMask, otherLable, rowRange, colRange, strechRange, lineRange, wordRange, spacing =1)

    rowRange    = (tableBoxEndRow, 1100)
    colRange    = (60, 710)
    strechRange = (800, 801)
    lineRange   = (1,2)
    wordRange   = (2, 4)
    puttext     = random.choice(on*20 + off*80)
    if puttext:   
        img, segMask,e =add_text_with_mask(img, segMask, otherLable, rowRange, colRange, strechRange, lineRange, wordRange, spacing =1)


    # Step 4 Add Dimention Block
    dimentionOptions = ["DIMS IN INCHES:", "DIMS IN CMS:", "DIMENSIONS (CMS):", "DIMENSIONS:", "DIMS(CMS):",
		        "DIMS(INCHES):", "DIMS:", "BOX DIMENTIONS", "DIMENTION OF BOX AS", "Dimenstions of box:",
                            "Dimenstions", "Dims:", "dimentions (CMS)", "dimentions(INCHES)", "dims(CMS)", "dimentions in INC:"]
    mulOptions = ["*", " * ", "x", " x ", "X", " X "]

    boxes = randrange(1,6)
    dimenstionDescription = random.choice(dimentionOptions)
    (label_width, label_height), baseline = cv2.getTextSize(dimenstionDescription, font, fontScale*0.98, lineType)
    dimenstionDescription += "\n"
    mul = random.choice(mulOptions)

    dimentionBoxStartRow = max(tableBoxEndRow, eob+10)
    dimBoxTextStart = randrange(812, 900)

    puttext     = random.choice(on*65 + off*35)
    if puttext:
        dimentionBoxStartRow = eob+10
        dimBoxTextStart = randrange(885, 895)

    puttext     = random.choice(on*70 + off*30)
    if puttext:
        dimentionBoxStartRow += int(np.random.normal(50, 10))
        maxStrechOfTextBox = 0
        for box in range(boxes):
	        boxdims1 = ''+str(randrange(1,140))+mul+str(randrange(1,140))+mul+str(randrange(1,140))
	        dimenstionDescription += boxdims1
	        dimenstionDescription += "\n"
	        (label_width, label_height), baseline = cv2.getTextSize(boxdims1, font, fontScale*0.9, lineType)
	        maxStrechOfTextBox = max(maxStrechOfTextBox,label_width)

        dimenstionDescription = dimenstionDescription[:-1]
        for i, line in enumerate(dimenstionDescription.split('\n')):
            y = dimentionBoxStartRow + i* (baseline+label_height)
            cv2.putText(img, line, (dimBoxTextStart, y), font, fontScale*0.9, fontColor, lineType)
            (label_width, label_height), baseline = cv2.getTextSize(line, font, fontScale*0.9, lineType)
            topLeftCorner      = (dimBoxTextStart-5,  dimentionBoxStartRow - (baseline+ label_height + 5))
            bottomRightCorner  = (dimBoxTextStart + label_width + 5,   y)
            cv2.rectangle(segMask, topLeftCorner, bottomRightCorner, DimentionLable, -1)
       
    
    return (img, segMask)

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
    fontColor              = random.uniform(1, 30)
    lineType               = 2
    
    itemsToBePrinted = [pieces, GW, Kglb, rateC, commodity, CW, rateCrg, total, nature]
    #print("font: ", font)
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
        
        gamma = max(np.random.normal(0.3, 0.2),0.15)
        tl = preProcess(tl, gamma)
        
        tl = add_surrounding_text(tl)
        tl,mask = add_table_entries(tl)
        
        cropRowPos = int(np.random.normal(int(tl.shape[0]*0.38), 40))
        tlcrop = tl[cropRowPos : cropRowPos+615,:]
        #maskcrop = mask[int(tl.shape[0]*0.38):int(tl.shape[0]*0.72),:]
        maskcrop = mask[cropRowPos : cropRowPos+615,:]

        tx = int(np.random.normal(0, 20))
        ty = 0
        M = np.float32([[1,0, tx],[0,1, ty]])
        rows,cols = tlcrop.shape
        tlcrop = cv2.warpAffine(tlcrop,M,(cols,rows))        
        maskcrop = cv2.warpAffine(maskcrop,M,(cols,rows))

        tx1 = int(np.random.normal(0, 6.5))
        tx2 = int(np.random.normal(0, 2))
        tx3 = int(np.random.normal(0, 2))
        pts1 = np.float32([[50,50],[1200,50],[50,550], [1200,550]])
        pts2 = np.float32([[50+tx1, 50],[1200-tx1,50],[50-tx1,550], [1200+tx1,550]])

        M = cv2.getPerspectiveTransform(pts1,pts2)

        tlcrop = cv2.warpPerspective(tlcrop,M,(cols,rows))
        maskcrop = cv2.warpPerspective(maskcrop,M,(cols,rows))

        rotation = np.random.normal(0, 0.75)
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
        tlcrop = cv2.warpAffine(tlcrop, rotation_matrix, (cols, rows))
        maskcrop = cv2.warpAffine(maskcrop, rotation_matrix, (cols, rows))

        # Generate Gaussian noise
        #noiseVar = max(np.random.normal(0.01, 0.2),0.001)
        gauss = np.random.normal(0, 0.5,tlcrop.size)
        gauss = gauss.reshape(tlcrop.shape[0],tlcrop.shape[1]).astype('uint8')
        # Add the Gaussian noise to the image
        tlcrop = cv2.add(tlcrop,gauss)
        
        gamma = max(np.random.normal(0.8, 0.2),0.6)
        tlcrop = preProcess(tlcrop, gamma)
        

 
        # Generate Gaussian noise
        noiseVar = max(np.random.normal(200, 200), 10)        
        gauss = np.random.normal(0,noiseVar,tlcrop.size)
        gauss = gauss.reshape(tlcrop.shape[0],tlcrop.shape[1]).astype('uint8')
        tlcrop = cv2.bitwise_and(tlcrop,tlcrop,mask = gauss)


        tlcrop = cv2.resize(tlcrop, (512,256), interpolation = cv2.INTER_AREA)
        maskcrop = cv2.resize(maskcrop, (512,256), interpolation = cv2.INTER_AREA)
                
        cv2.imshow("image", tlcrop)
        cv2.imshow("mask", maskcrop)
        #cv2.imshow("image2", tl)
        cv2.waitKey(0)



