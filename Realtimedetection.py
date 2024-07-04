import cv2
import time
import urllib.request
import requests
import threading
import json
import random

def thingspeak_post():
    threading.Timer(15,thingspeak_post).start()
    val=random.randint(1,30)
    URl='https://api.thingspeak.com/update?api_key='
    KEY='JYEIVAU740Z935LP'
    HEADER='&field1={}&field2={}&field3={}'.format(str(val),str(val),str(val))
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data=urllib.request.urlopen(NEW_URL)
    print(data)
    
cap=0
vid = cv2.VideoCapture(0)
while(cap!=1): 
    ret, frame = vid.read()  
    cv2.imshow('frame', frame)               
    if cv2.waitKey(1) & 0xFF == ord('s'):        
          try:                
            cv2.imwrite('test.jpg', frame)           
            print(str(1)+"--image saved")  
            cap=1  
          except:
            print("Save error")
    time.sleep(0.1)
              
   


print("Stage 1")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('full_model.h5')

print("Stage 2")

img=image.load_img('test.jpg',target_size=(224,224))
print("Image opened")
print("Stage 3")

x=image.img_to_array(img)
x

x.shape

x=x/255

print("Stage 4")
import numpy as np
x=np.expand_dims(x,axis=0)
img_data=(x)
img_data.shape
print("Stage 5")
model.predict(img_data)
print("Stage 6")
print("finAL")
print(model.predict(img_data))
print("Stage 7")
a=np.argmax(model.predict(img_data), axis=1)

print(a)
print("Stage 8")


# Step 1
score_theory = 60
score_practical = 20

if(a == 0):
    print("Case 1") 
    print("infernce 1")
    print ("infernce2 ")
    thingspeak_post()
elif(a ==1):
    print("Case 2") 
    print("Damaged road")
    print ("infernc2 ")
    thingspeak_post()
elif(a ==2):
    print("Case 3") 
    print("Plain road")
    print ("infernc2 ")
    thingspeak_post()
elif(a ==3):
    
    print("Case 4") 
    print("infernce 1")
    print ("infernc2 ")
    thingspeak_post()   

