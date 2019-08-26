clc;clear;
% load the data
load face.mat
Xn = Y(:,:,:);
X_size = size(Xn); 
% initialize the parameters
k = 0;
K = 1;
mu =0.1;
max_mu = 100;
nIter=10;
tol =2e-1;
% set block sizes
block_sizes = [2,2,64];
disp('block_sizes:');disp(block_sizes);
levels = size(block_sizes,1);
ms = block_sizes(:,1);
ns = block_sizes(:,2);
vs = block_sizes(:,3);
N1 = max(ms,ns);
lambdas = 0.01/sqrt(N1*vs);
% show the original images
figure(1);imshow3(Xn,[],[8,8]);
L =zeros(size(Xn)); LU = Xn; 
S = zeros(size(Xn)); Y = zeros(size(Xn));
tic
for it = 1:nIter
    Lk = L;
    Sk = S;
    L = blockSVT_tensor(Xn - S -Y/mu, block_sizes,1/ mu); 
    mu =1.4*mu;
    S = prox_l1(Xn - L-Y/mu,lambdas/mu);
    figure(4);imshow3(L,[],[levels*8,8]);
    titlef(it);
    drawnow
    figure(3);imshow3(S,[],[levels*8,8]);
    titlef(it);
    drawnow
    Y = Y - mu * (Xn - L - S);
    dY = L+S-Xn;
    chgL = max(abs(Lk(:)-L(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([ chgL chgS max(abs(dY(:))) ]);
     
    if chgL < tol
        break;
    end 
end
toc
