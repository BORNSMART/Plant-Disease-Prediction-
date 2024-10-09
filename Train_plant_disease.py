# Importing libraries
import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
from tensorflow.keras.layers import Dense, Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential 

# Data Preprocessing 

# Training image pre-processing 
training_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\Aditya Dubey\OneDrive\Desktop\Projects\Plant disease detection\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

# Validation image pre-processing 
validation_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\Aditya Dubey\OneDrive\Desktop\Projects\Plant disease detection\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

for x,y in training_set: #see the structure 
    print(x,x.shape)
    print(x,x.shape)
    break

#building model

model= Sequential()
#Building CNN layer

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Dropout(0.25)) #to avoid overfitting

#flattening
model.add(Flatten())
model.add(Dense(units=1500,activation='relu'))
model.add(Dropout(0.40))

#Output Layer
model.add(Dense(units=38,activation='softmax'))

#Compiling model

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Model Training  
training_history=model.fit(x=training_set,validation_data=validation_set,epochs=10 )

#To avoid overshooting 
# 1.Choose smaller learning rate
# 2.There may be a chance of underfitting
# 3.Add more convolution layer to extract more feature from images 

#Model Evaluation

#On training_set
train_loss,train_acc = model.evaluate(training_set) 
print(train_loss,train_acc)

#On validation
val_loss,val_acc = model.evaluate(training_set) 
print(val_loss,val_acc)

#Save model
model.save("trained_model.keras")
training_history.history 

#Recording history in json
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)

##Accuracy Visulization 
epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel("No. of epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()                                                                    
plt.show()