#!/usr/bin/env octave

# definisco costanti
N = 4;


# carico immagine
img = imread("./lena.jpg");

# mostro immagine
imshow(img);

pause

# devo generare con un ciclo for questa cosa
# img(1,1) img(1,2) img (1,3) img(1,4)
# img(2,1) img(2,2) img (2,3) img(2,4)

# inizializzo array temporaneo NxN
temp = zeros(N,N);

# determino lunghezza immagine
l = length(img);

for i=1:l


