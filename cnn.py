from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k 
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.regularizers import l2
class CNN:
  @staticmethod
  def build(height,width,depth,classes):
      model=Sequential()
      input_shape=(height,width,depth)
      channel_dim = -1

      if(k.image_data_format() == 'channels_first'):
        input_shape = (depth,width,height)
        channel_dim = 1

      model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
      model.add(Activation('relu'))
      model.add(Conv2D(32, (3, 3)))
      model.add(MaxPooling2D())
      model.add(BatchNormalization(axis=-1))
      model.add(Dropout(0.50))

     #model.add(Conv2D(32, (3, 3)))
      #model.add(Activation('relu'))
      #model.add(Conv2D(64, (3, 3)))
      #model.add(Activation('relu'))
      #model.add(MaxPooling2D())
      #model.add(BatchNormalization(axis=-1))
      #model.add(Dropout(0.25))

    


      model.add(Conv2D(64, (3, 3), padding='same'))
      model.add(Activation('relu'))
      model.add(Conv2D(128, (3, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D())
      model.add(BatchNormalization(axis=-1))
      model.add(Dropout(0.30))
      
    


      model.add(Conv2D(128, (3, 3),padding='same'))
      model.add(Activation('relu'))
      model.add(Conv2D(256, (3, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D())
      model.add(BatchNormalization(axis=-1))
      model.add(Dropout(0.25))
      model.add(Flatten())
      model.add(Dense(512))#kernel_regularizer=l2(0.01)))
      model.add(Activation('relu'))
      model.add(Dropout(0.50))
      model.add(Dense(activation='sigmoid', units=1))

      print(model.summary())


      return model
