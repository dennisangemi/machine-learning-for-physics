% functions


% randu_between
% description: genera valori casuali uniformemente distribuiti
% a: estremo inferiore
% b: estremo superiore
% n: numero di elementi da generare
% output: vettore
function randbet = randu_between(a,b,n)
    randbet = a + (b-a).*rand(n,1);
end
