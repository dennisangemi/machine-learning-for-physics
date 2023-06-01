#!/usr/bin/env python
# coding: utf-8

# # CBRN
# 
# Metodo delle semirette per il riconoscimento di caratteri numerici manoscritti

# In[2]:


# hide output
# %%capture

# load libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as keraster
import tensorflow as tf
from   keras.datasets import mnist
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense


# In[3]:


# definisco costante passo
PASSO = 0.001

# carica dati mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


# definisco funzione 'plot_image'
def plot_image(img):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')

# definisci funzione 'show_image'
def show_image(img):
    plt.figure(figsize=(4,4))
    plot_image(img)
    plt.show(block = False)



# In[6]:


# trova il centro dell'immagine
def find_center(img):
    # trova i pixel con valore maggiore di 0
    y,x = np.where(img > 0)
    # trova il centro dell'immagine
    center = (int(np.round(np.mean(x),0)), int(np.round(np.mean(y),0)))
    return center

# definisco funzione 'plot_center'
def plot_center(center,color):
    plt.scatter(center[0], center[1], c = color)

def generate_lines_from(point, img, m, passo):

    # with dell'immagine
    l = len(img)-1

    # rinomino point in c
    c = point

    # determino l'intercetta della retta
    q = c[1]-m*c[0]

    # creo due liste vuote
    x = []
    y = []

    # aggiungo il centro all'array x e y
    x.append(c[0])
    y.append(c[1])

    # while y[-1] < 27 e contemporaneamente x[-1] < 27
    while (y[-1] < l) and (x[-1] <= l):
        x.append(x[-1]+passo)
        y.append(m*x[-1]+q)

    # se x[-1] > 27 o y[-1] > 27 allora elimina l'ultimo elemento di x e y
    if x[-1] > l or y[-1] > l:
        x.pop()
        y.pop()

    return x, y




# salvare i valori dei pixel sulla retta individuata dai punti x,y in un array 
def get_pixel_values(img, x, y):
    # inizializza l'array a 0 casta i valori di pixel in interi
    l=len(x)
    values = np.zeros(l)
    for i in range(len(x)):
        #print(np.round(max(x[i]-1,1)).astype(int))
        #print(np.round(min(x[i]+1,l)).astype(int))
        #print(np.round(max(1,y[i]-1)).astype(int))
        #print(np.round(min(y[i]+1,l)).astype(int))
        
        kernel = img[np.round(max(x[i]-1,0)).astype(int):np.round(min(x[i]+1,l-1)).astype(int),np.round(max(0,y[i]-1)).astype(int):np.round(min(y[i]+1,l-1)).astype(int)]       
        values[i] = np.max(kernel)
    return values.astype(int)




# definisco funzione per calcolare distanza tra due punti
def calcola_distanza(punto_inizio, punto_fine):
    distanza = np.sqrt((punto_inizio[0] - punto_fine[0])**2 + (punto_inizio[1] - punto_fine[1])**2)
    np.round(distanza, 2)
    return distanza


def find_intersections(img, x, y, soglia_l, soglia_h, show_image_flag):
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
        print(index)
        print(pixel[index])
        while (index < l_p) and (pixel[index]>=soglia_h):
            index+=1
        numero_intersezioni+=1
        m=np.round((inizio + index -1)/2).astype(int)
        if (numero_intersezioni==1):
            punto_inizio = (x[m],y[m])
            distanza_ret = calcola_distanza(punto_inizio,c) 
            print("distanza centro=",distanza_ret)       
        #print("m=",m)
        #print("inizio=",inizio)
        #print("fine=",index-1)
        #print("coordinata x=",x[np.round(inizio).astype(int)])
        #print("coordinata y=",y[np.round(inizio).astype(int)])
    return numero_intersezioni,distanza_ret


distanze = []



for i in range(1,10):
    img = x_train[i]

    #trova il centro dell'immagine
    c = find_center(img)
    print("Il centro ha coordinate:", c)
    # genera retta invocando generate_lines_from
    x, y = generate_lines_from(c, img, 1.2, PASSO)
    pixel = get_pixel_values(img, x, y)
    ni, d = find_intersections(img,x,y,10,10,1)
    print(ni)
    print(d)

    ni, d = find_intersections(img,x,y,150,150,1)
    print(ni)
    print(d)



# definisci funzione che plotta le intersezioni
def plot_intersections(coordinate_intersezioni, color):
    for i in range(len(coordinate_intersezioni)):
        plt.scatter(coordinate_intersezioni[i][0], coordinate_intersezioni[i][1], color = color)

# non so se la sintassi corretta sia
# plt.scatter(coordinate_intersezioni[i][1], coordinate_intersezioni[i][0], color = color)
# oppure
# plt.scatter(coordinate_intersezioni[i][0], coordinate_intersezioni[i][1], color = color)


# In[ ]:


# quindi basterà fare

for i in range(1,10):
    img = x_train[i]
    
    # calcola centro
    c = find_center(img)
    
    x, y = generate_lines_from(c, img,0.6, PASSO)
    coordinate_intersezioni, distanze = find_intersections(img, x, y, soglia = 10)
    
    # plotta i punti di intersezione sull'immagine insieme alla semiretta (sovrapposta)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap = 'gray')
    plt.plot(x, y, color = 'red')
    plot_center(c, 'blue')
    plot_intersections(coordinate_intersezioni, 'red')
    plt.show()

    print("Le distanze sono: ", distanze)






# plotto img con assi x e y chiamati 'x' e 'y'
plt.figure(figsize=(4,4))
plt.imshow(img, cmap = 'gray')
plt.axis('on')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ecco perchè! l'asse y è rivolto verso il basso. Cosa implica questa osservazione?

# In[ ]:


# test su singola immagine per capire cosa non funziona


c = find_center(img)
print("Coordinate centro:", c)

x, y = generate_lines_from(c, img, m = 0.6)
print("x:",x)
print("y:",y,"\n")

pixel = get_pixel_values(img, x, y)
coordinate, distanze = find_intersections(img,x,y, soglia = 2)

print("i valori dei pixel sulla retta sono:")
print(pixel,"\n")

print("coordinate:")
print(coordinate)

print("\ndistanze:")
print(distanze)

