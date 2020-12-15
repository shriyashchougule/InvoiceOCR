import numpy as np
import cv2
import string

from random import randrange
import random
# Image Gen
#img = np.zeros((900,1500,3), np.uint8)

# Read template
imgOriginal = cv2.imread("t01.png",0)

def DilateH(img):
	# Taking a matrix of size 5 as the kernel 
	kernel = np.ones((2,1), np.uint8)
	imgP = cv2.dilate(img, kernel, iterations=1)
	return imgP

def DilateV(img):
	# Taking a matrix of size 5 as the kernel 
	kernel = np.ones((1,2), np.uint8)
	imgP = cv2.dilate(img, kernel, iterations=1)
	return imgP

operations = [DilateH, DilateV]

def preprocess(img):
	operation = random.choice(operations)
	return operation(img)

n=9
while n>0:
	img = imgOriginal.copy()
	img = preprocess(img)

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
	cv2fontList = [0, 2, 3, 4, 6, 7] #Ref https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html

	font                   = random.choice(cv2fontList) 
	fontScale              = random.uniform(0.5, 0.75)
	fontColor              = (2)
	lineType               = 2


	rowStart = randrange(152,320)

	piecesStart = randrange (31,68)
	GWStart = randrange(112,157)
	KgStart = randrange(211,221)
	RCStart = randrange(251,260)
	CCStart = randrange(290, 341)
	CWStart = randrange(437,490)

	rateCStart = 570
	if rateC != "AS AGREED":
		rateCStart = randrange(585,650)

	TotalStart = 720
	if rateC != "AS AGREED":
		TotalStart = randrange(740,775)

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
	mask = np.zeros((img.shape), np.uint8)

	# Strech of the last word i.e value in the Total column 
	(label_width, label_height), baseline = cv2.getTextSize(str(Total), font, fontScale, lineType)

	topLeftCorner      = (piecesStart-10,  rowStart - (baseline+ label_height + 5))
	bottomRightCorner  = (TotalStart + label_width + 10,  rowStart + 5)

	cv2.rectangle(mask, topLeftCorner, bottomRightCorner, 60, -1)


	# Step 3 Add text to Nature Column
	
	textStart = int(np.random.normal(930, 10))

	ProductDiscription = ''

	maxStrechOfTextBox = 0
	baseline = 0
	lines = randrange(1,6+1)
	for l in range(1,lines):
		lineNotReady = True
		while lineNotReady:
			words = randrange(2,5)
			line = ''
			for w in range(2, words+1):
				letters = randrange(2, 5)
				line = line.join(random.choices(string.ascii_uppercase + string.digits, k = letters))
				line += " "
		
			(label_width, label_height), baseline = cv2.getTextSize(line, font, fontScale*0.9, lineType)
			if (label_width+textStart) > 1165:
				continue
			else:
				ProductDiscription += line
				ProductDiscription += "\n"
				maxStrechOfTextBox = max(maxStrechOfTextBox, label_width)
				lineNotReady = False

	print(ProductDiscription)
	y = 0
	for i, line in enumerate(ProductDiscription.split('\n')):
		y = rowStart + i* (baseline+label_height)
    	#cv2.putText(img, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
		cv2.putText(img, line, (textStart, y), font, fontScale*0.9, fontColor, lineType)		
	print(maxStrechOfTextBox)
	topLeftCorner      = (textStart-10,  rowStart - (baseline+ label_height + 5))
	bottomRightCorner  = (textStart + maxStrechOfTextBox + 10,   y)

	cv2.rectangle(mask, topLeftCorner, bottomRightCorner, 100, -1)

	cv2.imshow("Display window", img)
	cv2.imshow("Segmentation Mask", mask)
	cv2.waitKey(0)




