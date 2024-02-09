#!/usr/bin/env python
# coding: utf-8

# load personal libraries
from     cbrn_functions import *
from   costanti         import *

# carica dati mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# creo vettore degli angoli equally spaced
val = np.linspace(0, 2*math.pi, NUMERO_RETTE+1)

val = val[0:NUMERO_RETTE]

# riduco dataset
length_lrn = np.round(len(x_train)/DIVIDENDO,0).astype(int)
length_tst = np.round(len(x_test) /DIVIDENDO,0).astype(int)

# inizializzo matrici
x_train_2 = np.zeros((length_lrn,NUMERO_RETTE*NFR))
x_test_2  = np.zeros((length_tst,NUMERO_RETTE*NFR))
x_train_2 = np.asmatrix(x_train_2)
x_test_2  = np.asmatrix(x_test_2) 

i=0
for alpha in val:
    # calcolo coefficiente angolare
    m = math.tan(alpha)
    
    # creo dataset trasformato
    x_train_2[0:length_lrn,NFR*i:(NFR*i)+NFR] = transform_dataset(length_lrn, 0, x_train,m, PASSO, DIMENSIONI_VENTAGLIO, EPSILON_ALPHA)
    x_test_2 [0:length_tst,NFR*i:(NFR*i)+NFR] = transform_dataset(length_tst, 0, x_test, m, PASSO, DIMENSIONI_VENTAGLIO, EPSILON_ALPHA)
    
    #input("enter")
    i+=1 


y_train = y_train[0:length_lrn]
y_test  = y_test [0:length_tst]

#x_tmp     = x_train_2
#x_train_2 = x_test_2
#x_test_2  = x_tmp

#x_tmp     = y_train
#y_train   = y_test
#y_test    = x_tmp

print("dimension x_train_2",np.shape(x_train_2))
print("dimension x_test_2", np.shape(x_test_2 ))
print("dimension y_train",  np.shape(y_train  ))
print("dimension y_test",   np.shape(y_test   ))

# output

nome_modello = "../models/np.mnist.cbrn.model.h7"

# invoco funzione recognize_digit per creare modello
history = recognize_digit(NUMERO_RETTE*NFR, N_EPOCHE, 10, x_train_2, y_train, x_test_2, y_test, nome_modello)

print("Model exported:",nome_modello)

# invoco funzione per visualizzare i risultati
plot_results(history,'last_plot.png')

