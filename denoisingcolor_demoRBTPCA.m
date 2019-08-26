clc;
clear all;
close all;
% load data
n=1;
bbbb= zeros(50,1);
bbb= zeros(50,1);
bb= zeros(50,1);
b= zeros(50,1);
for i=1:n
imageName=strcat(num2str(i),'.jpg');
X = double(imread(imageName));  
X = X/255;
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
max_mu = 500;
nIter=500;
tol = 1e-2;
psnrMAX=0;
% set block sizes
block_sizes = [24,24,3];

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
    k=k+1;
    Lk = L;
    Sk = S;
    L = blockSVT_tensor(Xn - S -Y/mu, block_sizes,1/ mu);  
    mu = min(rho*mu,max_mu);  
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
bbbb(i) = toc;
err = norm(L(:)-X(:))/norm(X(:));
bb(i)=err;
L = max(L,0);
L = min(L,maxP);
psnr = PSNR(X,L,maxP);
[mssim, ssim_map]=ssim(X,L);
bbb(i)=mssim;
b(i)=psnr;
figure(1)
imshow(X/max(X(:)))
set(gca,'position',[0 0 1 1])
figure(2)
imshow(Xn/max(Xn(:)))
set(gca,'position',[0 0 1 1])
figure(3)
imshow(L/max(L(:)))
set(gca,'position',[0 0 1 1])
end
% % disp(b);
% save('psnrblockspase.mat','b');
% save('errblockspase.mat','bb');
% save('ssimblockspase.mat','bbb');
% save('CPUtime_blockspase.mat','bbbb');