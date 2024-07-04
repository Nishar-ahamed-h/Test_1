import numpy as np
import time 
import cv2 
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('full_model.h5')
SS='Datasets\\test\\1\\12.jpg'
img=image.load_img(SS,target_size=(224,224))


imgv = cv2.imread(SS)

kernel = np.ones((5,5),np.float32)/25  
dst = cv2.bilateralFilter(imgv,9,75,75)

#dst = cv.medianBlur(img,5)
plt.subplot(121),
plt.imshow(imgv),
plt.title('Original Input')
                                           
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Processed Image')
plt.xticks([]), plt.yticks([])
plt.show()

x=image.img_to_array(img)
x

x.shape

x=x/255


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=(x)
img_data.shape

model.predict(img_data)

print("finAL")
print(model.predict(img_data))

a=np.argmax(model.predict(img_data), axis=1)

print(a)



# Step 1
score_theory = 60
score_practical = 20
window_name = 'RESULT'     
font = cv2.FONT_HERSHEY_SIMPLEX
org1 = (50, 50)
org2 = (1, 100)
org3 = (1, 125)
fontScale = 0.5
fontScale1 = 1.35
color = (0, 255, 0)
color1 = (0, 255, 255)
thickness = 1

if(a == 0):
    print("pathhole") 
    print("   ")
    print ("...")
    imgv = cv2.putText(imgv, 'pathole', org1, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
##    imgv = cv2.putText(imgv, 'Normal,', org2, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)
##
##    imgv = cv2.putText(imgv, ', Normal', org3, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, imgv)
    
elif(a ==1):
    print("Normal")  
    print(" .")
    print (" ")
    imgv = cv2.putText(imgv, 'Normal', org1, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
##    imgv = cv2.putText(imgv, ' ,', org2, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)

##    imgv = cv2.putText(imgv, ', ,  ', org3, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, imgv)

elif(a ==2):
    print("No pathHole")  
    print(" .")
    print (" ")
    imgv = cv2.putText(imgv, 'pathhole', org1, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
##    imgv = cv2.putText(imgv, ' ,', org2, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)
##
##    imgv = cv2.putText(imgv, ', ,  ', org3, font, 
##                       fontScale, color1, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, imgv)
    

else:
    print("Not applicable ") # not applicable

    
