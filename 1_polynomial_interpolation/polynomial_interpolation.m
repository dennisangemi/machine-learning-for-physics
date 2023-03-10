%% Polynomial interpolation

% cleaning
clc
clear

% funzione seno
sen = @(x) sin(2*pi*x);

% genero vettori
x = linspace(0,1,100);
y = sen(x);

% plotto funzione seno
figure;
plot(x,y)
xlabel("x")
ylabel("y")
legend("sin(2\pix)")
%%
% genero set di learning
n_lrn = 10;
x_lrn = linspace(0,1,n_lrn);
eps = 0.15;
y_lrn = sin(2*pi*x_lrn);
y_lrn = y_lrn + (-eps + (2.*eps).*rand(n_lrn,1))';

figure
plot(x,y)
hold on
plot(x_lrn,y_lrn,"o")
legend("sin(2\pix)","data")
xlabel("x")
ylabel("y")
hold off
%%
% genero matrice di Vandermonde
V = fliplr(vander(x_lrn))
%% 
% Risolvo il sistema $y = \alpha V$ dove $\alpha$ sono i coefficienti del polinomio 
% cercato: $y = \alpha_1 + \alpha_2 x_1 + \alpha_3 x_2^2+...$
% 
% Alla luce della forma matriciale, è possibile determinare i coefficienti $\alpha$ 
% eseguendo il prodotto righe per colonna tra l'inversa della matrice di Vandermonde 
% e il vettore colonna y

% determino i coefficienti
a = pinv(V)*(y_lrn')

% ottengo il polinomio funzione degli scalari x e m (grado)
poly = @(x,m) (x.^(0:m))*(a(1:m+1));

% over-fitting
z = zeros(1,100);
for i=1:100
    z(i) = poly(x(i),n_lrn-1);
end
%%
figure;
plot(x,z,"r")
hold on
plot(x_lrn,y_lrn,'ob')
plot(x,y,"g")
hold off
legend("polynomial fit", "data", "sin(2\pix)")
xlabel("x")
ylabel("y")
ylim([-1.5 1.5])
xlim([0 1])
%%
% plotting at different M (polynomial order)

% plot M = 0
figure;
plot(x,repelem(a(1),length(x)),"r")
hold on
plot(x_lrn,y_lrn,'ob')
plot(x,y,"g")
hold off
legend("polynomial fit", "data", "sin(2\pix)")
xlabel("x")
ylabel("y")
ylim([-1.5 1.5])
xlim([0 1])
title("M=0")

% plot M = 1
figure;
plot(x,(a(1)+a(2).*x),"r")
hold on
plot(x_lrn,y_lrn,'ob')
plot(x,y,"g")
hold off
legend("polynomial fit", "data", "sin(2\pix)")
xlabel("x")
ylabel("y")
ylim([-1.5 1.5])
xlim([0 1])
title("M=1")

% plot M = 3
figure;
plot(x,(a(1) + (a(2).*x) + a(3).*x.^2 + a(4).*x.^3),"r")
hold on
plot(x_lrn,y_lrn,'ob')
plot(x,y,"g")
hold off
legend("polynomial fit", "data", "sin(2\pix)")
xlabel("x")
ylabel("y")
ylim([-1.5 1.5])
xlim([0 1])
title("M=3")
%%
% learning error

% initializing vectors
learning_error = zeros(1,n_lrn);
y_fit = learning_error;

for j = 1:n_lrn

    % estimate y points
    for i = 1:n_lrn
        y_fit(i) = poly(x_lrn(i),j-1);
    end

    % calculating learning error
    learning_error(j) = sqrt(sum((y_fit-y_lrn).^2));
end

% plotting learning error
plot(0:n_lrn-1,learning_error,"-o")
xlabel("M (grado del polinomio interpolante)")
ylabel("Scarto quadratico medio")
legend("Training")