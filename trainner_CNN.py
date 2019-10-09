import tensorflow as tf
from tensorflow  import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Conv2D,Activation,Flatten,Dense,MaxPooling2D,Dropout

import pickle

pickle_in=open("training_img.pickle","rb")
X = pickle.load(pickle_in)

pickle_in=open("training_roll_no.pickle","rb")
y = pickle.load(pickle_in)

X =X/255.0

model = keras.Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('softmax'))


model.compile(loss = 'sparse_categorical_crossentropy',optimizer= 'adam',metrics = ['accuracy'])

model.fit(X,y,epochs=30,validation_split=0.1)

model.summary()
keras.models.save_model(model,"D:\\CNN1.tf")