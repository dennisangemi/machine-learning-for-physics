import tensorflow.keras as keras
import tensorflow as tf
import mnist
import numpy as np
import matplotlib.pyplot as plt
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")
print("Data dimensions:\n")
print("xlrn=",x_train.shape)
print("xtst=",x_test.shape)
print("ylrn=",y_train.shape)
print("ytst=",y_test.shape)
wait=input("Press Enter to continue.")
while True:
 npat=int(input("Give new pattern number or 0 to continue: "))
 print("Ecco il pattern n.",npat)
 image=x_train[npat];
 plt.imshow(image, cmap='gray')
 print("Il valore atteso e': ",y_train[npat]);
 plt.show(block=False)
 plt.pause(0.001) #Per un bug che talvolta capita sull'istruzione sopra
 input("Press any key to continue")
 plt.close()
 if npat==0: break
 
 
x_train=x_train.reshape(60000,784)
x_test=  x_test.reshape(10000,784)
print(x_train.shape)
x_train=x_train/255
x_test =x_test/ 255
num_categories=10
y_train=keras.utils.to_categorical(y_train,num_categories)
y_test =keras.utils.to_categorical(y_test, num_categories)
print("New data dimensions:\n")
print("ylrn=",y_train.shape)
print("ytst=",y_test.shape)
print("New categorical output pattern n.",npat,"=",y_train[npat])
wait = input("Press Enter to continue.")
#creo il modello
model=Sequential()
#aggiungo layer + input
model.add(Dense(units=100,activation='relu',input_shape=(784,)))
#aggiungo uno strato nascosto
model.add(Dense(units=2,activation='relu'))
#output layer
model.add(Dense(units=10,activation='softmax'))
model.summary()
wait = input("Press Enter to continue.")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# Model is being trained on 1875 batches of 32 images each ... (1875*32=60000)
history=model.fit(x_train,y_train,epochs=15,verbose=1,validation_data=(x_test,y_test));
             
# List all data in history
print(history.history.keys())
# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Salvataggio del modello
model.save('marco.mnist.model.h5')
plt.show()
#Clear the Memory
#Before moving on, please execute the following cell to clear up the GPU memory.
#This is required to move on to the next notebook.
#import IPython
#app = IPython.Application.instance()
#app.kernel.do_shutdown(True)
#{​​​​​​​​'status': 'ok', 'restart': True}​​​​​​​​
