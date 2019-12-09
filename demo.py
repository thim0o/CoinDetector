#!/usr/bin/env python
# coding: utf-8

# In[37]:


from keras.models import load_model

model_dir = 'D:\\BSc Computer Science\\P2\\Physical Computing\\Project\\modelsWeight\\full_2.h5'

model = load_model(model_dir)
model.summary()


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image      
import random


# In[39]:


cropped_imgs_path = 'D:\\github\\deeplearning\\datasets\\euroCoins\\'

datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        # shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.3,
)

train_generator = datagen.flow_from_directory(
        cropped_imgs_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        shuffle=True,
        subset="training"
)

val_generator = datagen.flow_from_directory(
        cropped_imgs_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset="validation"
)


# In[40]:


def predict_images(img):
#     img = mpimg.imread(path)
#     plt.imshow(img)


#     image = Image.open(path)
    image = img.convert('RGB')
    image = image.resize((224, 224))

    probabilities = model.predict(np.expand_dims(image, axis=0))
    type_list = tuple(zip(val_generator.class_indices.values(), val_generator.class_indices.keys()))

    for i in probabilities.argsort()[0][-6:][::-1]: 
        return probabilities[0][i], type_list[i][1]
#         print(probabilities[0][i], "  :  " , type_list[i][1])
            
# img_dir = 'C:\\Users\\khanh\\Downloads\\IMG_4193_1.jpg'

# img = Image.open(img_dir)
# print(predict_images(img))


# In[41]:


import cv2
import numpy as np
from matplotlib import cm

print(cv2.__version__)


# In[55]:


vid_dir = 'C:\\Users\\khanh\\Downloads\\test5.mp4'

cap = cv2.VideoCapture(vid_dir)

while 1:
    ret, img = cap.read()
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive Thresholding
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Circle detection
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 64,
                                    param1=40, param2=40, minRadius=40,
                                    maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                x, y = i[0], i[1]
                R = i[2]
                R -= 10
                if y - R < 0 or x - R < 0: 
                    continue
                cropped_img = img[y - R : y + R, x - R : x + R]
                converted_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
                prob, className = predict_images(converted_img)
                prob = format(prob * 100, '.2f')
                output = className + ': ' + str(prob) + '%'
                cv2.rectangle(img, (x - R, y - R), (x + R, y + R), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(img, output, (x - R, y - R), font, 1, (0, 0, 255), 4, cv2.LINE_AA)
                
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
            
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




