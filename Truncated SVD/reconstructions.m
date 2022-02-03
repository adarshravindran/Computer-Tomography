%% Truncated SVD

%% part a

N=100;
i=linspace(1,N,100);
h=1/N;
x=(i-(ones(1,100)*0.5))*h;

v=ones(1,100)*0.5;
A=diag(v)+tril(ones(length(v)),-1);

g=A*f(x);

% Noisy data
p=19; % We need to add 19 percent noise
rng(314);
n=randn(length(g),1); % creating a random vector
gdelta=(n/norm(n))*norm(g)*(p/100)+g; 

hold off
plot(g,'DisplayName','g')
hold on
plot(gdelta,'DisplayName','noisy data gdelta')
title('part (a) : plotting g and gdelta')

%% part b

fv=f(x);

%using SVD build in function to get u,v vectors and sigma values.
[U, S, V] = svd(A);

% computing error, by varying K
K=1;
errors=fv*0;
while K<101
    errors(K)=norm(fv-fstar(U,S,V,g,K));
    K=K+1;
end

[~,I]=min(errors);
display(strcat('Optimum value of K is ',num2str(I))) % Optimum value of K is 100, when there is no noise.
% This is as expected

ft=fstar(U,S,V,g,I);

hold off
plot(fv,'DisplayName','Original function')
hold on
plot(ft,'DisplayName','Optimum K reconstruction')
title('TSVD reconstruction')
legend('Location','northeast')


%% part c

fv=f(x);

%using SVD build in function to get u,v vectors and sigma values.
[U, S, V] = svd(A);

% computing error, by varying K
K=1;
errors=fv*0;
while K<101
    errors(K)=norm(fv-fstar(U,S,V,gdelta,K));
    K=K+1;
end

[~,I]=min(errors);

display(strcat('Optimum value of K is ',num2str(I))) % Optimum value of K is 5, when there is no noise.
% When we add noise, K =5 is a good cutoff for a close reconstruction

ft=fstar(U,S,V,g,I);
hold off
plot(fv,'DisplayName','Original function')
hold on
plot(ft,'DisplayName','Optimum K reconstruction')
title('TSVD reconstruction')
legend('Location','northeast')

%% checks
% In this section, we can vary the value of K manually, to see how it
% affects the reconstruction. Higher values of K, introduce oscillations.

fv=f(x);

%using SVD build in function to get u,v vectors and sigma values.
[U, S, V] = svd(A);

f1=fstar(U,S,V,gdelta,40);
f2=fstar(U,S,V,gdelta,5);

hold off
plot(fv)
hold on
plot(f1,'DisplayName','K=15')
plot(f2,'DisplayName','K=5')
legend('Location','best')
% visually, lot of values of K around 10-15 seem to fit well.

%% Functions

% TSVD
function ftsvd = fstar(U,S,V,g,K)
k=1;
tsvd=g*0;
while k<K+1
    tsvd=tsvd+(1/S(k,k))*dot(g,U(:,k))*V(:,k);
    k=k+1;
end
ftsvd=tsvd;
end

% f(x)
function ex_f= f(x)
j=1;
func=x*0;
while j<length(x)+1
    func(j)=exp(-2*x(j))*cos(5*x(j));
    j=j+1;
end
ex_f=func';
end
