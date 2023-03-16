% functions


% randu_between
% description: genera valori casuali normalmente distribuiti
% a: estremo inferiore
% b: estremo superiore
% n: numero di elementi da generare
% output: vettore
function randbet = randn_between(a,b,n)
    randbet = a + (b-a).*randn(n,1);
end
