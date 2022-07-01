
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator



imagegen = ImageDataGenerator( rescale=1/255.0, 
                                validation_split=0.2)

train = imagegen.flow_from_directory("trainingSet/", 
                                        subset="training",
                                        class_mode="categorical", 
                                        shuffle=True, 
                                        batch_size=128, 
                                        target_size=(40, 40),
                                        color_mode="grayscale")
validation = imagegen.flow_from_directory("trainingSet/",
                                        subset="validation", 
                                        class_mode="categorical", 
                                        shuffle=True, 
                                         batch_size=128, 
                                        target_size=(40, 40),
                                        color_mode="grayscale")




model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(40,40,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(11, activation='softmax'))


model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])



model.fit(train,
        batch_size=128, 
        epochs = 5,  
        validation_data =validation, 
        verbose=2)


model.save('digit.model')


