from vgg1.model1_custom_mini import tinyVGG
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2
import pickle
import random


#to save images in the background,we will use AGG setting of matplotlib
matplotlib.use('Agg')
#path to dataset
dataset='data1'
#path to save model
model_path='model.h5'
#path to save labels
label_path='/'
#path to save plots
plot_path='/'
HP_LR=1e-3
HP_EPOCHS=142
HP_BS=32
HP_IMAGE_DIM=(96,96,3)
data=[]
classes=[]

imagepaths=sorted(list(paths.list_images(dataset)))
#print(imagepaths)

random.seed(42)
random.shuffle(imagepaths)
for imgpath in imagepaths:
        try:
                image=cv2.imread(imgpath)
                image=cv2.resize(image,(96,96))
                image_array=img_to_array(image)
                data.append(image_array)
                label=imgpath.split(os.path.sep)[-2]
                classes.append(label)
        except Exception as e:
                 print(e)

data=np.array(data,dtype='float')/ 255.0
labels=np.array(classes)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)

print("2")

xtrain,xtest,ytrain,ytest=train_test_split(data,labels,test_size=0.3,random_state=42)

aug=ImageDataGenerator(rotation_range=0.25,width_shift_range=0.25,height_shift_range=0.1
	,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

model=tinyVGG.build(height=96,width=96,depth=3,classes=len(lb.classes_))
opt= Adam(lr=HP_LR , decay=HP_LR/HP_EPOCHS)
csv_logger = CSVLogger('train_to_plot.log')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

H=model.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS),validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS,callbacks=[csv_logger])

#for testing accuracy->

preds=model.evaluate(xtest,ytest)
#print("Loss = " +str(preds[0]))
#print("Accuracy = " +str(preds[1]))


model.save('mymodel11.h5')
