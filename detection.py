# در این قسمت تعریف کتاب خانه ها مورد نیاز برای اجرای برنامه 
import cv2 as cv
from google.colab.patches import cv2_imshow
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import keras.utils as ku 
import numpy as np
import cv2 
import tensorflow as tf

from keras.models import Model
from keras.applications import VGG16
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,UpSampling2D,GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import SGD


#/////////////// ارابه ای از حروف زبان انگلیسی ایجاد می کنیم 
maxcont=6
arrayalph=['1:a', '2:b', '3:c', '4:d', '5:e', '6:f', '7:g', '8:h', '9:i', '10:j', '11:k', '12:l', '13:m', '14:n', '15:o', '16:p', '17:q', '18:r', '19:s', '20:t', '21:u', '22:v', '23:w', '24:x', '25:y', '26:z']
arrayalph2=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# بخش آموزش که شبکه را ایجاد می کنیم و داده ها را فراخوانی میکنیم و شبکه کانولوشنی را با داده ها آموزش می دهدیم  و ورودی تصاویر را دریافت می کند , ور بر روی شبکه اموزش داده شده predict صورت میدهد
def runmodel(image1):


  (input_train, target_train), (input_test, target_test) = emnist.load_data(type='letters')
 
  #////////////////////////////// لود کردن داده و جداسازی داده آموزشی و تست
  input_train = input_train.reshape((input_train.shape[0], 28, 28, 1))
  input_test = input_test.reshape((input_test.shape[0], 28, 28, 1))

  print(input_train.shape)
  print(input_test.shape)
  

  target_train=ku.to_categorical(target_train)
  target_test=ku.to_categorical(target_test)

  input_valid=input_test[0:4000]
  target_valid=target_test[0:4000]
  input_test=input_test[4000:20800]
  target_test=target_test[4000:20800]



  input_train = input_train.astype('float32')
  input_test = input_test.astype('float32')
  input_valid = input_valid.astype('float32')
 

  cv2_imshow(input_test[1000])
  input_train= input_train / 255.0
  input_test = input_test / 255.0
  input_valid = input_valid / 255.0
  cv2_imshow(input_test[5])
  
  
 
  #/////////////////////////////////////////////////////// Model

  model = Sequential()
  
  model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))  
  model.add(Dropout(0.2))  
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(27, activation='softmax'))
 
  
  # /////////////////////////////////////////////////////compile model

  learning_rate = 2e-5
  # optimizer Adam recommended
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
  model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  history = model.fit(input_train, target_train, epochs=10 ,validation_data=(input_valid, target_valid))

 
  loss_train = history.history['loss']
  loss_val = history.history['val_loss']

  plt.plot(loss_train, 'g')
  plt.plot( loss_val, 'b')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

  loss_train = history.history['accuracy']
  loss_val = history.history['val_accuracy']
      
  plt.plot( loss_train, 'g')
  plt.plot(loss_val, 'b')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()



  #///////////////////////////////////////////////////// test
  _, acc = model.evaluate(input_test, target_test,verbose=0)
  cv2_imshow(image1[0])
  cv2_imshow(input_test[5])
  a=np.array([input_test[1000]])
  preds=model.predict(image1)
  
  pred_class = np.argmax(preds, axis=-1)
  string=" "
  for i in range(0,maxcont,1):
    string=string + arrayalph2[pred_class[i] - 1]
  print(string)
  print(pred_class)
  print(arrayalph)
  print(arrayalph2[pred_class[0] - 1])
  print(acc)
  
#انجام پیش پردازش بر روی داده هاای وروی  و  تبدیل کردن به داده های با سایز های مناسب برای برسی
def resize28_28(img1):
  width = 28
  height = 28
  dim = (width, height)
 
  # resize image
  resized = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
  return resized

#تابع اجرایی که ابتدا حروف پیش پردازش شده را  میخواند و برای انجام تست ها و آموزش وارد  شبکه می کند 
def alphabet1():
    alph=list()
    for i in range(0,maxcont,1):
        im_gray = cv.imread(str(i)+'.png',0)
        cv2_imshow(im_gray)
        r,c=im_gray.shape
        thresh = 128
        im_gray = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY)[1]
        
      #  cv2_imshow(im_gray)
        for i in range(0,r):
          for j in range(0,c):
            im_gray[i][j]=255-im_gray[i][j]

        im_gray = cv2.dilate(im_gray,None,iterations = 1)
        cv2_imshow(im_gray)
        re=resize28_28(im_gray)
        alph.append(re)
    #cv.imwrite("6.png",re)
    testimage=np.array(alph)
    for i in range(0,maxcont,1):
       cv2_imshow(testimage[i])
   
    testimage=testimage.astype('float32')
    testimage=testimage/255
   
    testimage=testimage.reshape((testimage.shape[0], 28, 28, 1))
    print(testimage.shape)
    cv2_imshow(testimage[0])
    runmodel(testimage)
    #print(re.shape)

#شروع
alphabet1()