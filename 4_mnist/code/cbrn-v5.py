#!/usr/bin/env python
# coding: utf-8

# # MNIST semirette (v5)

# In[152]:


# load libraries
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as keraster
import tensorflow as tf
import keras
from   keras.datasets import mnist
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense


# In[164]:


# definisco costante passo
PASSO = 0.5
NUMERO_RETTE = 10




# carica dati mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[251]:


# definisco funzione 'plot_image'
def plot_image(img):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')

# definisci funzione 'show_image'
def show_image(img):
    plt.figure(figsize=(4,4))
    plot_image(img)
    plt.show(block = False)


# definisco funzione trova il centro dell'immagine
def find_center(img):
    # trova i pixel con valore maggiore di 0
    y,x = np.where(img > 0)
    # trova il centro dell'immagine
    center = (int(np.round(np.mean(x),0)), int(np.round(np.mean(y),0)))
    return center

# definisco funzione 'plot_center'
def plot_center(center,color):
    plt.scatter(center[0], center[1], c = color)

# definisco funzione 'generate_lines_from'
def generate_lines_from(point, img, m, passo):

    # with dell'immagine
    l = len(img)-1

    # rinomino point in c
    c = point

    # determino l'intercetta della retta
    q = c[1]-m*c[0]

    # creo due liste vuote
    
    x = np.zeros(np.round(28*10*PASSO).astype(int))
    y = np.zeros(np.round(28*10*PASSO).astype(int))

    # aggiungo il centro all'array x e y
    x[0] = c[0]
    y[0] = c[1]
    
    i=0

    # while y[-1] < 27 e contemporaneamente x[-1] < 27
    while (y[i] < l) and (x[i] <= l):
        i+=1
        x[i]=(x[i-1]+passo)
        y[i]=(m*x[i-1]+q)

    # se x[-1] > 27 o y[-1] > 27 allora elimina l'ultimo elemento di x e y
    if x[i] > l or y[i] > l:
        i-=1

    
    return x[0:i], y[0:i]


# definisco funzione 'get_pixel_values' che ritorna un array con i valori di grigio 
def get_pixel_values(img, x, y):
    
    # inizializza l'array a 0 casta i valori di pixel in interi
    l = len(x)
    
    values = np.zeros(l)
    
    len_img = len(img)
    for i in range(l):
        
        kernel = img[np.round(max(x[i]-1,0)).astype(int):
                     np.round(min(x[i]+1,len_img-1)).astype(int),
                     np.round(max(0,y[i]-1)).astype(int):
                     np.round(min(y[i]+1,len_img-1)).astype(int)]       
        values[i] = np.max(kernel)
        
        
    return values.astype(int)


# definisco funzione per calcolare distanza tra due punti
def calcola_distanza(punto_inizio, punto_fine):
    distanza = np.sqrt((punto_inizio[0] - punto_fine[0])**2 + (punto_inizio[1] - punto_fine[1])**2)
    np.round(distanza, 2)
    return distanza

# definisco funzione 'find intersection'
def find_intersections(img, x, y, c, pixel, soglia_l, soglia_h, show_image_flag):
    numero_intersezioni=0
    distanza_ret=1.5*28
    
    if (show_image_flag == 1):
        # plotto
        plt.figure(figsize=(8,4))
        # set axis limits to 0, 28
        plt.xlim(0, 28)
        plt.ylim(0, 28)
        plt.subplot(1,2,1)
        # plotta x e y sull'immagine
        plt.imshow(img, cmap = 'gray')
        plt.plot(x, y, color = 'red')
        # plotta il centro dell'immagine
        plt.scatter(c[0], c[1], color = 'blue')
        plt.subplot(1,2,2)
        plt.plot(pixel)
        plt.show()
    
    l_p = len(pixel)
    index = 0
    while index < l_p:
        while (index < l_p) and (pixel[index]<soglia_l):
            index+=1
        if (index == l_p):
            break
            
        inizio = index
        index+=1
        #print(index)
        #print(pixel[index])
        while (index < l_p) and (pixel[index]>=soglia_h):
            index+=1
        numero_intersezioni+=1
        m=np.round((inizio + index -1)/2).astype(int)
        if (numero_intersezioni==1):
            punto_inizio = (x[m],y[m])
            distanza_ret = calcola_distanza(punto_inizio,c) 
            #print("distanza centro=",distanza_ret)       
    return numero_intersezioni,distanza_ret

# definisci funzione che plotta le intersezioni
def plot_intersections(coordinate_intersezioni, color):
    for i in range(len(coordinate_intersezioni)):
        plt.scatter(coordinate_intersezioni[i][0], coordinate_intersezioni[i][1], color = color)




# creo funzione che trasforma x_train
def transform_dataset(length,show_image_flag,data,coefficiente_angolare):
    
    #length = len(data)
    #length = 5000
    # inizializzo matrice vuota
    x_transformed = np.zeros((length,2))
    
    for i in range(length):
        print(i+1,"/",length)
        
        img = data[i]
        c = find_center(img)
        x, y = generate_lines_from(c, img, coefficiente_angolare, PASSO)
        pixel = get_pixel_values(img, x, y)
        
        x_transformed[i,0],x_transformed[i,1] = find_intersections(img,x,y,c,pixel,150,150,show_image_flag)
        #input("Press Enter to continue...")
    return x_transformed


# crea funzione 'recognize_digit' per riconoscere il numero che crei il modello sopra
def recognize_digit(shape, n_epochs, n_categories, x_train, y_train, x_test, y_test, path):

    # rendo categoriche le variabili di output
    y_train = keras.utils.to_categorical(y_train, n_categories)
    y_test = keras.utils.to_categorical(y_test, n_categories)

    # normalizzo usando la funzione di keras
    # x_train = keras.utils.normalize(x_train, axis = 1)
    # x_test = keras.utils.normalize(x_test, axis = 1)

    # creo il modello
    model = Sequential()

    # flattening
    # model.add(keras.layers.Flatten(input_shape = (shape,shape)))

    # aggiungo layer + input
    model.add(Dense(units = 100, activation = 'relu', input_shape = (shape,)))

    # aggiungo uno strato nascosto
    model.add(Dense(units = 2, activation = 'relu'))

    # output layer
    model.add(Dense(units = 10, activation = 'softmax'))

    model.summary()

    # addestro il modello
    model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs = n_epochs, verbose = 1, validation_data = (x_test, y_test));

    # Salvataggio del modello
    model.save(path)

    # ritorna history
    return history

# crea funzione per visualizzare i risultati
def plot_results(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(5,5))
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')

    # summarize history for loss
    plt.subplot(2,1,2)
    plt.subplots_adjust(hspace = 0.5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()





val = np.linspace(2*math.pi/NUMERO_RETTE, 2*math.pi, NUMERO_RETTE)

print(val)

ms = np.zeros((NUMERO_RETTE)) # coefficienti angolari
i=0
for alpha in val:
    ms[i] = math.tan(alpha)
    i+=1



length=50
x_train_2 = np.zeros((length,NUMERO_RETTE*2))

a = transform_dataset(length,0,x_train,ms[0])
print(a[np.ix_([0,1],[0,1])])
print(a.shape)
input("enter")
print(a)
print("la prima riga")
print(a[1,:])
print("ciao")
print(a[1:2,1:2])
print("bye")

b = a[1,1]
print(b.shape)

input("enter")
for i in range(NUMERO_RETTE):
    a= transform_dataset(length,0,x_train,ms[i])
    #x_train_2[2*i:(2*i)+1,:] = transform_dataset(length,0,x_train,ms[i])
    a.shape


# invoco funzione transform_dataset
x_train_2 = transform_dataset(length,0,x_train)


# In[115]:


x_test_2 = transform_dataset(1000,0,x_test)


# In[140]:


y_train = y_train[0:5000]
y_test = y_train[0:1000]


# In[150]:


# invoco funzione recognize_digit per creare modello
history = recognize_digit(2, 100, 10, x_train_2, y_train, x_test_2, y_test, "4_mnist/models/np.mnist.model.h6")

# invoco funzione per visualizzare i risultati
plot_results(history)

