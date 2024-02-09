#!/usr/bin/env python
# coding: utf-8

# load libraries
from   cbrn_functions import *
from   costanti       import *

print(NUMERO_RETTE)

# specifica il percorso della cartella contenente le immagini
cartella_immagini = "../img/numbers"

# specifica le dimensioni desiderate
nuova_dimensione = (28, 28)

# importa le mie immagini
my_imgs = importa_mie_immagini(cartella_immagini, nuova_dimensione)
print(my_imgs[6])
#show_image(my_imgs[6])
#input("ciao")

print("shape my_imgs",np.shape(my_imgs))

val = np.linspace(0, 2*math.pi, NUMERO_RETTE)

my_imgs_transformed = np.zeros((len(my_imgs),NUMERO_RETTE*NFR))
my_imgs_transformed = np.asmatrix(my_imgs_transformed)

# trasformo dataset mie immagini
i=0
for alpha in val:
    # calcolo coefficiente angolare
    m = math.tan(alpha)
    
    # creo dataset
    my_imgs_transformed[0:len(my_imgs),NFR*i:(NFR*i)+NFR] = transform_dataset(len(my_imgs),1,my_imgs,m, PASSO, DIMENSIONI_VENTAGLIO, EPSILON_ALPHA)
    
    #input("enter")
    i+=1 

print("shape my_imgs_transformed",np.shape(my_imgs_transformed))

print(my_imgs_transformed)

# applico modello alle mie immagini
personal_digit_recognizer(my_imgs, my_imgs_transformed, "../models/np.mnist.cbrn.model.h7")



