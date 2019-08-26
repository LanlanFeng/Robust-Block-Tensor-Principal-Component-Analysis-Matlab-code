clc;
clear all;
close all;
% load data
bb= zeros(200,1);
b= zeros(200,1);
% bbb= zeros(200,1);
load('face');
Y_sizes = size(Y); 
m =Y_sizes(:,1);
n = Y_sizes(:,2);
X = zeros(m,n,4);
X(:,:,1)=Y(:,:,1);
X(:,:,2)=Y(:,:,5);
X(:,:,3)=Y(:,:,6);
X(:,:,4)=Y(:,:,8);
maxP = max(abs(X(:)));

[n1,n2,n3] = size(X);
Xn = X;
rhos = 0.1;
ind = find(rand(n1*n2*n3,1)<rhos);
Xn(ind) = rand(length(ind),1);
Xn_size=size(Xn);
% initialize the parameters
k = 0;
K = 1;
rho =1.4;
mu =0.1;
max_mu = 100;
nIter=10;
tol = 1e-5;
% set block sizes
block_sizes = [24,24,4];
disp('block_sizes:');disp(block_sizes);

levels = size(block_sizes,1);
ms = block_sizes(:,1);
ns = block_sizes(:,2);
vs = block_sizes(:,3);

N1 = max(ms,ns);
lambda1 = 1;
lambdas = 1/sqrt(N1*vs);
L = zeros(size(Xn)); LU = Xn; 
S = zeros(size(Xn)); Y = zeros(size(Xn));
tic
for it = 1:nIter
    err = norm(L(:)-X(:))/norm(X(:));
    bb(it)=err;
    L1 = max(L,0);
    L1 = min(L,maxP);
    psnr = PSNR(X,L1,maxP);
    b(it)=psnr;
    k=k+1;
    Lk = L;
    Sk = S;
    L = blockSVT_tensor(Xn - S -Y/mu, block_sizes,1/ mu);  
    mu = rho*mu;
    S = prox_l1(Xn - L-Y/mu,lambdas/mu);
    Y = Y - mu * (Xn - L - S);
    dY = L+S-Xn;
    chgL = max(abs(Lk(:)-L(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([ chgL chgS max(abs(dY(:))) ]);
     
    if chg < tol
        break;
    end 

end
toc

err = norm(L(:)-X(:))/norm(X(:))
L = max(L,0);
L = min(L,maxP);
psnr = PSNR(X,L,maxP)
[mssim, ssim_map]=ssim(X,L1);
mssim
