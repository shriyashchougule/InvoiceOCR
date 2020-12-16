import numpy as np
import cv2
import string
from collections import namedtuple
from random import randrange
import random
line = ''
#letters = 8
#line = line.join(random.choices(string.ascii_lowercase+string.ascii_uppercase + string.digits, k = letters))

words = randrange(4, 20)
print("words: ", words)

for w in range(1, words+1):
    letters = randrange(1, 12)
    print(w, letters)
    myword = "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k = letters))
    line += myword
    line += " "
print(line)
