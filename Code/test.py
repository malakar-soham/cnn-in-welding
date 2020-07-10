import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import os
import random
import tensorflow as tf


h,w = 512,512
num_cases = 3

images = []
labels = []

files = os.listdir('./dataset/images/')
random.shuffle(files)

model = tf.keras.models.load_model('my_model')

lowSevere = 1
midSevere = 2
highSevere = 4

for f in files[0:num_cases]:
    test_img = cv2.imread('./dataset/images/' + f)
    resized_img = cv2.resize(test_img,(w,h))
    resized_img = resized_img/255
    cropped_img = np.reshape(resized_img,
          (1,resized_img.shape[0],resized_img.shape[1],resized_img.shape[2]))

    test_out = model.predict(cropped_img)

    test_out = test_out[0,:,:,0]*1000
    test_out = np.clip(test_out,0,255)

    resized_test_out = cv2.resize(test_out,(test_img.shape[1],test_img.shape[0]))
    resized_test_out = resized_test_out.astype(np.uint16)

    test_img = test_img.astype(np.uint16)

    grey = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    for i in range(test_img.shape[0]):
     for j in range(test_img.shape[1]):
          if(grey[i,j]>150 & resized_test_out[i,j]>40):
            test_img[i,j,1]=test_img[i,j,1] + resized_test_out[i,j]
            resized_test_out[i,j] = lowSevere
          elif(grey[i,j]<100 & resized_test_out[i,j]>40):
            test_img[i,j,2]=test_img[i,j,2] + resized_test_out[i,j]
            resized_test_out[i,j] = highSevere
          elif(resized_test_out[i,j]>40):
            test_img[i,j,0]=test_img[i,j,0] + resized_test_out[i,j]
            resized_test_out[i,j] = midSevere
          else:
            resized_test_out[i,j] = 0

    M = cv2.moments(resized_test_out)
    maxMomentArea = resized_test_out.shape[1]*resized_test_out.shape[0]*highSevere
    print("0th Moment = " , (M["m00"]*100/maxMomentArea), "%")

    test_img = np.clip(test_img,0,255)

    test_img = test_img.astype(np.uint8)

    cv2_imshow(test_img)

    cv2.waitKey(0)
