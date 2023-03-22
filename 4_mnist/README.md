# MNIST

Obiettivo: costruire un modello che sia in grado di riconoscere carattere numerico manoscritto

Cominciamo importando le librerie necessarie per il corretto funzionamento del codice (tensorflow, keras, matplotlib)


```python
import tensorflow.keras as keras
import tensorflow as tf
import mnist
import numpy as np
import matplotlib.pyplot as plt
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense
```

    2023-03-22 23:25:58.065590: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-03-22 23:26:06.000025: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-03-22 23:26:06.000107: I tensorflow/compiler/xla/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2023-03-22 23:26:26.942192: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2023-03-22 23:26:26.954086: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2023-03-22 23:26:26.954147: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.



```python
# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path = "mnist.npz")

# undestand data dimensions
print("Data dimensions:\n")
print("x_lrn:",x_train.shape)
print("x_tst:",x_test.shape)
print("y_lrn:",y_train.shape)
print("y_tst:",y_test.shape)
```

    Data dimensions:
    
    x_lrn: (60000, 28, 28)
    x_tst: (10000, 28, 28)
    y_lrn: (60000,)
    y_tst: (10000,)


Visualizziamo un'immagine del dataset di learning (per modificare il numero dell'immagine, modifica il valore della variabile `npat`)


```python
npat = 7
print("Ecco il pattern n.", npat)
print("Il valore atteso è: ", y_train[npat]);

image = x_train[npat];
plt.imshow(image, cmap = 'gray')
plt.show(block = False)
```

    Ecco il pattern n. 7
    Il valore atteso è:  3



    
![png](img/output_5_1.png)
    



```python
# flattening
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

print("New dimensions\n")
print("x_train:",x_train.shape)
print("x_test:",x_test.shape)
```

    New dimensions
    
    x_train: (60000, 784)
    x_test: (10000, 784)



```python
# casting
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
```

L'idea è quella di dividere ogni immagine per il max per normalizzare


```python
# normalizzo
print("Before normalization")

x_max = np.amax(x_train, axis = 1, keepdims=True)
print("I valori univoci dell'array dei massimi (train) sono:",np.unique(x_max))
x_train /= x_max

x_max = np.amax(x_test, axis = 1, keepdims=True)
print("I valori univoci dell'array dei massimi (test) sono:",np.unique(x_max))
x_test /= x_max
```

    Before normalization
    I valori univoci dell'array dei massimi (train) sono: [254. 255.]
    I valori univoci dell'array dei massimi (test) sono: [254. 255.]



```python
# rendo categoriche le variabili di output
num_categories = 10
y_train = keras.utils.to_categorical(y_train,num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)
```


```python
print("New categorical output pattern n.",npat,"=",y_train[npat])
```

    New categorical output pattern n. 7 = [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]


## Modello
Adesso creiamo il modello avente due layer di neuroni con funzione di attivazione relu e l'ultimo (output) softmax


```python
#creo il modello
model=Sequential()
#aggiungo layer + input
model.add(Dense(units=100,activation='relu',input_shape=(784,)))
#aggiungo uno strato nascosto
model.add(Dense(units=2,activation='relu'))
#output layer
model.add(Dense(units=10,activation='softmax'))
model.summary()
```

    2023-03-22 23:32:57.256712: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2023-03-22 23:32:57.264288: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)
    2023-03-22 23:32:57.264480: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (dennisangemi): /proc/driver/nvidia/version does not exist
    2023-03-22 23:32:57.288985: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 100)               78500     
                                                                     
     dense_1 (Dense)             (None, 2)                 202       
                                                                     
     dense_2 (Dense)             (None, 10)                30        
                                                                     
    =================================================================
    Total params: 78,732
    Trainable params: 78,732
    Non-trainable params: 0
    _________________________________________________________________



```python
# addestro il modello
model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 15, verbose = 1, validation_data = (x_test, y_test));
```

    Epoch 1/15
    1875/1875 [==============================] - 8s 3ms/step - loss: 1.2873 - accuracy: 0.5738 - val_loss: 0.9213 - val_accuracy: 0.7332
    Epoch 2/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.7910 - accuracy: 0.7763 - val_loss: 0.6709 - val_accuracy: 0.8153
    Epoch 3/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.5769 - accuracy: 0.8434 - val_loss: 0.4989 - val_accuracy: 0.9020
    Epoch 4/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.4445 - accuracy: 0.9126 - val_loss: 0.4228 - val_accuracy: 0.9173
    Epoch 5/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.3762 - accuracy: 0.9240 - val_loss: 0.3887 - val_accuracy: 0.9221
    Epoch 6/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.3354 - accuracy: 0.9312 - val_loss: 0.3796 - val_accuracy: 0.9242
    Epoch 7/15
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.3068 - accuracy: 0.9368 - val_loss: 0.3616 - val_accuracy: 0.9262
    Epoch 8/15
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2836 - accuracy: 0.9406 - val_loss: 0.3575 - val_accuracy: 0.9265
    Epoch 9/15
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2648 - accuracy: 0.9447 - val_loss: 0.3542 - val_accuracy: 0.9294
    Epoch 10/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2494 - accuracy: 0.9474 - val_loss: 0.3561 - val_accuracy: 0.9317
    Epoch 11/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2394 - accuracy: 0.9511 - val_loss: 0.3400 - val_accuracy: 0.9354
    Epoch 12/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2259 - accuracy: 0.9519 - val_loss: 0.3869 - val_accuracy: 0.9322
    Epoch 13/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2186 - accuracy: 0.9548 - val_loss: 0.3482 - val_accuracy: 0.9384
    Epoch 14/15
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2115 - accuracy: 0.9558 - val_loss: 0.3730 - val_accuracy: 0.9375
    Epoch 15/15
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2011 - accuracy: 0.9578 - val_loss: 0.3793 - val_accuracy: 0.9288


## To do


```python
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
```


```python

```
