'''
  https://colab.research.google.com/drive/1Iil6MLm56Ql-naLoY6t3rsjYk5T-j7W4?usp=sharing


  Public notebook by Me for proper codes in Collab format

  
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Making and Training of the model

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),#to convert 2D images into 1D vector
    tf.keras.layers.Dense(128,activation='relu'),#converting into 128 neurons with relu activation
    tf.keras.layers.Dropout(0.2),#to prevent overfitting
    tf.keras.layers.Dense(10,activation='softmax')]#A 10 classes model and softmax activation for output probabailites
)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''

We are using adam optimizer for efficient training.
Sparse categorical crossentropy for dealing of multiple
categorical data and accuracy as our metrics to compile the model


'''
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#fitting the model
model.fit(x_train,y_train,epochs=5)     #Epochs=5 means our model will see the data 5 times

'''

Training should look like:


Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9139 - loss: 0.2963
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9572 - loss: 0.1435
Epoch 3/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9667 - loss: 0.1080
Epoch 4/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9726 - loss: 0.0877
Epoch 5/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.9757 - loss: 0.0775
<keras.src.callbacks.history.History at 0x78dfea5bedb0>


'''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Testing the model

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print(f'Test accuracy:{test_acc}')

'''
Output:

313/313 - 1s - 3ms/step - accuracy: 0.9769 - loss: 0.0724
Test accuracy:0.9768999814987183

'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Saving Model

model.save('my_model.h5')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Loading of Model

loaded_model=tf.keras.models.load_model('my_model.h5')


#-----------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXX----------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------------------





















