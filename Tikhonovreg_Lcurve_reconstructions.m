%% Extra Exercise sheet : Tikhonov regularization and parameter selection
% Adarsh Ravindran ; UHH Student ID : 7479186
%{
A: L^2(0,1) -> L^2(0,1)
Af(x) = int(0,x)(f(t)dt)

f1(x) = sign(x-0.5)
f2(x) = sin(pi*x)
%}


%% (a) Creating noisy data - Given : N=300; To find : gdelta with 5% noise
N=300;
h=1/N;
i=(1:N);
x=(i-0.5)*h;

% Create A
A=h*(tril(ones(N),-1)+0.5*eye(N));
% subplot(1,2,1);
% heatmap(A,'Title','Matrix A heatmap');
%imshow(A);
g=A*f2(x);

% Noisy data
p=5; % We need to add 5 percent noise
rng(1);
n=randn(length(g),1); % creating a random vector
gdelta=(n/norm(n))*norm(g)*(p/100)+g; 

%it satisfies
diff=norm(g-gdelta)/norm(g)-0.05;% =-6.938893903907228e-18 negligible reconstruction error
disp(diff)
plot(g,'DisplayName', 'g=A*f2(x)', 'Color','blue')
hold on
plot(gdelta, 'DisplayName', 'gdelta', 'Color','red')
title('part (a) : Comparing g and gdelta for sin(pi*x)');
legend('location','northwest');
hold off

%% (b) Tikhonov regularization implementation

%{
 From Lecture notes, f* is a unique solution to the regularized
 lease squares problem for alpha >0, and it is given by

 f* = Σ(j=1 till N) {sigma(j)/[sigma(j)^2+alpha]}*<g|u(j)>*v(j)
 where U=[u(1) ... u(N)] , V=[v(1) ... v(N)], and s=diag(sigma(1),...,sigma(N))

 and f* solves
 (A*A + alpha*eye(N))* f* = A*g  (6.14)
%}


% let a be a vector storing the value {sigma(j)/[sigma(j)^2+alpha]}*<g|u(j)>*v(j);
% let alpha = 0, we will try for different values of alpha and check visually
% defining our parameters
N=300;
h=1/N;
i=(1:N);
x=(i-0.5)*h;
A=h*(tril(ones(N),-1)+0.5*eye(N));
g=A*f2(x);

% Noisy data
p=5; % We need to add 5 percent noise
rng(1);
n=randn(length(g),1); % creating a random vector
gdelta=(n/norm(n))*norm(g)*(p/100)+g; 

%using SVD build in function to get u,v vectors and sigma values.
[U, S, V] = svd(A);
% let alpha = 0, we will try for different values of alpha and check visually
% for alpha = 0, f2 and f* will be the same
alpha=0;
diff1 = f(S,U,V,g,alpha)-f2(x);
norm(diff1); % This difference should be zero. We get numerical error of 1.389910604113496e-10
subplot(1,2,1);
hold off
plot(f2(x),'DisplayName', 'f2=sin(pi*x)')
hold on
plot(f(S,U,V,g,alpha),'DisplayName', 'f* with alpha = 0')
title('alpha = 0 reconstruction without noise')
legend('location','south')

% Now we will check for different values of alpha, and for our gdelta, and
% plotting them

alpha=linspace(0.0035,0.004,2000);
diff=zeros(N,length(alpha));
normdiff=zeros(1,length(alpha));

i=1;
while i<length(alpha)+1
    diff(:,i)=f(S,U,V,gdelta,alpha(i)) - f2(x);
    normdiff(i)=norm(diff(:,i));
    i=i+1;
end
[~,I]=min(normdiff); %finding the index of minimal norm, to find corresponding alpha
% This can be done visually too, but it is difficult to see which curve is
% the closest, I found out after trial and error that the alpha lies
% between 0.003 and 0.004

% The optimal alpha has the minimum norm
disp(strcat('optimal alpha is = ', num2str(alpha(I)))) % I got optimal alpha = 0.0037204 approximately
subplot(1,2,2);
hold off
plot(f2(x),'DisplayName','f2=sin(pi*x)')
hold on
plot(f(S,U,V,gdelta,alpha(I)),'DisplayName',strcat('f* with noise, alpha = ', num2str(alpha(I))))
title('Optimal alpha reconstruction with added noise')
sgtitle('part (b) : Tikhonov regularization for sin(pi*x)')
legend('location','south')

%% (c) Choosing the regularization parameter using L-curve

% creating a set of admissible alphas near our optimal alpha with an
% interval of 5 percent
alphaset=linspace(0.00001,0.05,1000);
p=zeros(1,length(alphaset));
q=p;
i=1;
while i<length(alphaset)+1
    p(i)=log(norm(A*f(S,U,V,gdelta,alphaset(i))-gdelta)); % Only noise added data is considered
    q(i)=log(norm(f(S,U,V,gdelta,alphaset(i))));
    i=i+1;
end
hold off
subplot(1,2,1);
plot(p,q,'.');
% We can see the L-curve here
hold on
plot(p,q);
title('L-curve');
plot(p(36),q(36),'r*');

% After visual inspection of the L-curve, I chose the point
% (x,y)=(-1.08901,2.49161). This corresponds to p(36), q(36)
% which in turn corresponds to i=36, and alpha = alphaset(36) = 0.001761401401401

alpha = alphaset(36);
hold off
subplot(1,2,2);
hold on
plot(f2(x),'DisplayName', 'f2=sin(pi*x)')
hold on
plot(f(S,U,V,gdelta,alpha),'DisplayName', strcat('f* with alpha = ',num2str(alpha)))
title('Reconstruction for alpha chosen from L-curve')
sgtitle('part (c) : L-curve reconstruction for sin(pi*x)')
legend('location','south')
%% (d) Repeating all the steps for f1
%%%%%%%%%%%%%%%%%%%%%%% (a) Tikhonov Regularization for f1 %%%%%%%

g=A*f1(x);

% Noisy data
p=5; % We need to add 5 percent noise
rng(1);
n=randn(length(g),1); % creating a random vector
gdelta=(n/norm(n))*norm(g)*(p/100)+g; 

%using SVD build in function to get u,v vectors and sigma values.
[U, S, V] = svd(A);
% let alpha = 0, we will try for different values of alpha and check visually
% for alpha = 0, f1 and f* will be the same
alpha=0;
diff1 = f(S,U,V,g,alpha)-f1(x);
norm(diff1); % This difference should be zero. We get numerical error of 1.927904497943238e-11
subplot(2,2,1);
hold off
plot(f1(x),'DisplayName', 'f2=sin(pi*x)')
hold on
plot(f(S,U,V,g,alpha),'DisplayName', 'f* with alpha = 0')
title('alpha = 0 reconstruction without noise')
legend('location','north')

% Now we will check for different values of alpha, and for our gdelta, and
% plotting them

alpha=linspace(0.0004,0.0006,1000); % This range of alpha is chosen after trial and error
diff=zeros(N,length(alpha));
normdiff=zeros(1,length(alpha));
i=1;
while i<length(alpha)+1
    diff(:,i)=f(S,U,V,gdelta,alpha(i)) - f1(x);
    normdiff(i)=norm(diff(:,i));
    i=i+1;
end
[M,I]=min(normdiff); %finding the index of minimal norm, to find corresponding alpha
% This can be done visually too, but it is difficult to see which curve is
% the closest.

% The optimal alpha has the minimum norm
disp(strcat('optimal alpha is = ', num2str(alpha(I)))) % I got optimal alpha = 0.145 approximately
subplot(2,2,2);
hold off
plot(f1(x),'DisplayName','f1=sign(x-0.5)')
hold on
plot(f(S,U,V,gdelta,alpha(I)),'DisplayName',strcat('f* with noise, alpha = ', num2str(alpha(I))))
title('Optimal alpha reconstruction with added noise')
legend('location','north')

%%%%%%%%%%%%%%%%%%%%%%%% L-curve %%%%%%%%%%%%%%%%%%%%%%%%%
% creating a set of admissible alphas near our optimal alpha with an
% interval of 5 percent
alphaset=linspace(0.00001,0.005,1000);
p=zeros(1,length(alphaset));
q=p;
i=1;
while i<length(alphaset)+1
    p(i)=log(norm(A*f(S,U,V,gdelta,alphaset(i))-gdelta));
    q(i)=log(norm(f(S,U,V,gdelta,alphaset(i))));
    i=i+1;
end
hold off
subplot(2,2,3);
plot(p,q,'.');
% We can see the L-curve here
hold on
plot(p,q);
plot(p(31),q(31),'r*');
title('L-curve');

% After visual inspection of the L-curve, I chose the point
% (x,y)=(-1.47774,2.85925). This corresponds to p(31), q(31)
% which in turn corresponds to i=31, and alpha = alphaset(31) = 1.598498498498499e-04

alpha = alphaset(31);
hold off
subplot(2,2,4);
hold on
plot(f1(x),'DisplayName', 'f1=sign(x-0.5)')
hold on
plot(f(S,U,V,gdelta,alpha),'DisplayName', strcat('f* with alpha = ',num2str(alpha)))
title('Reconstruction for alpha chosen from L-curve')
sgtitle('part (d) : Tikhonov regularization and L-curve reconstruction for sign(x-0.5)')
legend

%% Function which returns f*
function fstar = f(S,U,V,g,alpha)
N=300;
a=zeros(N);
j=1;
while j<301
    a(:,j)=(S(j,j)/((S(j,j)^2)+alpha))*dot(g,U(:,j))*V(:,j);
    j=j+1;
end
i=1;
fstar=zeros(size(a(:,i)));
while i<301
    fstar=fstar+a(:,i);
    i=i+1;
end
end

%% Functions
function ex_f1 = f1(x)
    ex_f1 = sign(x - 0.5)';
end

function ex_f2 = f2(x)
    ex_f2 = sin(pi*x)';
end