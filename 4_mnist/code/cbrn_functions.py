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

from   costanti         import *

# definisco funzione 'plot_image'
def plot_image(img):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')

# definisci funzione 'show_image'
def show_image(img):
    plt.figure(figsize=(4,4))
    plot_image(img)
    plt.show()

# definisco funzione per calcolare distanza tra due punti
def calcola_distanza(punto_inizio, punto_fine):
    distanza = np.sqrt((punto_inizio[0] - punto_fine[0])**2 + (punto_inizio[1] - punto_fine[1])**2)
    np.round(distanza, 2)
    return distanza

# definisco funzione trova il centro dell'immagine
def find_center_dmax(img):
    # trova i pixel con valore maggiore di 0
    y,x = np.where(img > SOGLIA_INFERIORE)
    n_punti = len(y)
    
    
    x_min = np.min(x).astype(float)
    x_max = np.max(x).astype(float)
    y_min = np.min(y).astype(float)
    y_max = np.max(y).astype(float)
    pi = (x_min, y_min)
    pf = (x_max, y_max)
    d_max = calcola_distanza(pi,pf)
    # trova il centro dell'immagine
    #center = (int(np.round(np.mean(x),0)), int(np.round(np.mean(y),0)))
    center = (sum(x)/n_punti, sum(y)/n_punti)
    #print(center)
    return center, d_max

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
    
    x = np.zeros(np.round(28*10*passo).astype(int))
    y = np.zeros(np.round(28*10*passo).astype(int))

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
        if x[i] > l or y[i] > l or y[i]<0 or x[i] < 0:
            i-=1
            break

    
    return x[0:i], y[0:i]


# definisco funzione 'get_pixel_values' che ritorna un array con i valori di grigio 
def get_pixel_values(img, x, y):
    
    # inizializza l'array a 0 casta i valori di pixel in interi
    l = len(x)
    
    values = np.zeros(l)
    
    len_img = len(img)
    for i in range(l):
        
        #kernel = img[np.round(max(x[i]-1,0)).astype(int):
        #             np.round(min(x[i]+1,len_img-1),0).astype(int),
        #             np.round(max(0,y[i]-1)).astype(int):
        #             np.round(min(y[i]+1,len_img-1)).astype(int)]    
        kernel = img[np.round(x[i],0).astype(int),np.round(y[i],0).astype(int)]      
        
        values[i] = np.max(kernel)
        
        
    return values.astype(int)




# definisco funzione 'find intersection'
def find_intersections(img, x, y, c, d_max, pixel, soglia_l, soglia_h, show_image_flag):
    numero_intersezioni=0
    distanza_ret=np.ones(2)*1.5*28
    print(pixel)

    
    
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
        if (numero_intersezioni<=2):
            punto = (x[m],y[m])
            distanza_ret[numero_intersezioni-1] = calcola_distanza(punto,c) 
    
    if (show_image_flag == 1):
        # plotto
        plt.figure(figsize=(8,4))
        # set axis limits to 0, 28
        plt.xlim(0, 28)
        plt.ylim(0, 28)
        plt.subplot(1,2,1)
        # plotta x e y sull'immagine
        plt.imshow(img, cmap = 'gray')
        plt.plot(y, x, color = 'red', linewidth=4)
        # plotta il centro dell'immagine
        plt.scatter(c[0], c[1], color = 'blue')
        plt.subplot(1,2,2)
        plt.scatter(x,pixel)
        plt.show()
    
    return numero_intersezioni, distanza_ret[0]/d_max, distanza_ret[1]/d_max

# definisci funzione che plotta le intersezioni
def plot_intersections(coordinate_intersezioni, color):
    for i in range(len(coordinate_intersezioni)):
        plt.scatter(coordinate_intersezioni[i][0], coordinate_intersezioni[i][1], color = color)



# creo funzione che trasforma un dataset
def transform_dataset(length,show_image_flag,data,coefficiente_angolare,passo, dimensioni_ventaglio, epsilon_alpha):
    
    # inizializzo matrice vuota
    x_transformed = np.zeros((length,NFR))
    
    # ciclo sulle immagini dato coefficiente_angolare 
    for i in range(length):
        print(i+1,"/",length)
        
        img = data[i]
        c, d_max = find_center_dmax(img)
        x, y = generate_lines_from(c, img, coefficiente_angolare, passo)
        pixel = get_pixel_values(img, x, y)
        
        alpha_i = math.atan(coefficiente_angolare)
        val = np.linspace(alpha_i-epsilon_alpha, alpha_i+epsilon_alpha, dimensioni_ventaglio)
        
        md0 = 100
        md1 = 100
        
        # ciclo del ventaglio
        for alpha in val:
            ni, d0, d1  = find_intersections(img,x,y,c,d_max,pixel,SOGLIA_INFERIORE,SOGLIA_SUPERIORE,show_image_flag)
            
            # calcoliamo la minima distanza
            if (d0 < md0):
                md0 = d0
            
            # calcoliamo la minima distanza successiva
            if (d1 < md1):
                md1 = d1
            
        x_transformed[i,0] = ni
        x_transformed[i,1] = md0
        x_transformed[i,2] = md1
        
        #input("Press Enter to continue...")
    return x_transformed


# crea funzione 'recognize_digit' per riconoscere il numero che crei il modello sopra
def recognize_digit(shape, n_epochs, n_categories, x_train, y_train, x_test, y_test, path):

    # rendo categoriche le variabili di output
    y_train = keras.utils.to_categorical(y_train, n_categories)
    y_test = keras.utils.to_categorical(y_test, n_categories)


    # creo il modello
    model = Sequential()

    # aggiungo layer + input
    model.add(Dense(units = U_PRIMO_STRATO, activation = 'relu', input_shape = (shape,)))

    # aggiungo uno strato nascosto
    model.add(Dense(units = U_SECONDO_STRATO, activation = 'relu'))

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
def plot_results(history,outputname):
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
    
    # salvo grafico
    plt.savefig(outputname)
    
def importa_mie_immagini(cartella_immagini, nuova_dimensione):

    # inizializza l'array numpy per contenere tutte le immagini ridimensionate
    immagini = np.empty((0, *nuova_dimensione), dtype=np.uint8)
  
    # scorri tutti i file nella cartella specificata
    for nome_file in os.listdir(cartella_immagini):
        # carica l'immagine
        print("ciao")
        percorso_file = os.path.join(cartella_immagini, nome_file)
        img = cv2.imread(percorso_file, cv2.IMREAD_GRAYSCALE)

        # ridimensiona l'immagine
        img_ridimensionata = cv2.resize(img, nuova_dimensione)

        # aggiungi l'immagine all'array numpy
        immagini = np.vstack([immagini, np.expand_dims(img_ridimensionata, axis=0)])

    return immagini


# definisci funzione 'personal_digit_recognizer'
def personal_digit_recognizer(immagini, immagini_trasformate, model_path):
    # carica le immagini
    #immagini = importa_mie_immagini(cartella_immagini, nuova_dimensione)
    
    # carico modello salvato
    model = keras.models.load_model(model_path)

    # applico modello alle mie immagini
    predictions = model.predict(immagini_trasformate)

    plt.figure(figsize=(10, 5))
    for i in range(0, immagini.shape[0]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(immagini[i], cmap = 'gray')
        plt.title(np.argmax(predictions[i]))
        plt.axis('off')
    plt.show()
