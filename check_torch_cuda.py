# import torch
# print(torch.cuda.is_available())

import cv2 as cv
from PIL import Image
import numpy as np

# Read rickroll image using Image
img1 = Image.open('data/landingpad_resize/JPEGImages/IMG_6191_jpg.rf.48d5d1d75f4b29dcf608741b52c1b485.jpg')

img1_arr = np.asarray(img1)

# Reverse RGB to BGR


cv.imshow('Rickroll', img1_arr)
cv.waitKey(0)
cv.destroyAllWindows()