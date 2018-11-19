import sys
import numpy as np
from skimage import io, transform
import csv
import cv2
import os

os.chdir('/mnt/zhanghaoran/paris/paris_train_original/paris_train_original/')
name = os.listdir(os.getcwd())
print(name)

id = 0
with open('/mnt/zhanghaoran/gan_lstm.csv','w') as csvfile:    
    w = csv.writer(csvfile)
    for x in name:
        w.writerow(['/mnt/zhanghaoran/paris/paris_train_original/paris_train_original/'+x, id])
        id+=1
    
    
cv2.destroyAllWindows()
