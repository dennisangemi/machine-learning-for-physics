% functions


% rand_between
% a: estremo inferiore
% b: estremo superiore
% n: numero di elementi da generare
% output: vettore
function randbet = rand_between(a,b,n)
    randbet = a + (b-a).*rand(n,1);
end
