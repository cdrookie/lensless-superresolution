clear;clc
close all

% load functions
addpath(genpath('./utils'))
addpath(genpath('./src')) 
addpath(genpath('./src/func'))
data_path = './data/experiment/';
Time_stamp = '20250114';
Test_stamp = '_1';
File_Name = [data_path,'Filter_Holo_',Time_stamp,Test_stamp,'.mat'];
load(File_Name);
K = size(y,3);      % number of diversity measurements
xxx=1.31;
sig = 1;    % down-sampling ratio
sig2 = 1;
m = size(y(:,:,1),1);    % sensor resolution: m x m
n = m*sig;  % sample resolution: n x n

x = zeros(n,n);
x11 = zeros(m,m);
% physical parameters
params.pxsize11 = 1.67e-3;                   % pixel size (mm)
params.pxsize = params.pxsize11/sig;                   % pixel size (mm)
params.wavlen = NaN(K);
for i = 1:16
params.wavlen(i) = 0.625e-3;                 % wavelength (mm)
end
for i = 17:32
params.wavlen(i) = 0.525e-3;                 % wavelength (mm)
end
for i = 33:48
params.wavlen(i) = 0.470e-3;                 % wavelength (mm)
end
params.method = 'Angular Spectrum';     % numerical method

params.dist = xxx;                      % imaging distance (mm)
params.dist2 = xxx;  
params.dist3 = xxx; 

dist = NaN([1,K]);   
for k = 1:16
    dist(k) = params.dist;
end
for k = 17:32
    dist(k) = params.dist2;
end
for k = 33:48
    dist(k) = params.dist3;
end
sub_x = -params.pxsize11/2:0.167e-3:params.pxsize11/2;
sub_y = -params.pxsize11/2:0.167e-3:params.pxsize11/2;
[Xsub,Ysub] = meshgrid(sub_x,sub_y);
len_sub_pixel = length(sub_x);
u = zeros(len_sub_pixel,len_sub_pixel,K);
xsub = NaN(1,K);
ysub = NaN(1,K);
for i = 1:len_sub_pixel
    for j = 1:len_sub_pixel
       u(i,j,:) = subpixle1(sub_x(i),sub_x(j),y,m,m,params.pxsize11,K);
    end
end

for i = 1:K
   [p,q] = find(u(:,:,i)==min(min(u(:,:,i))));
   xsub(i) = sub_x(p);
   ysub(i) = sub_x(q);
   disp(['xsub=',num2str(xsub(i)),'  ,ysub=',num2str(ysub(i))])
%    legend('AWF','WF','WFi')
end
   surf(Xsub,Ysub,u(:,:,2));
% figure
% hold on,semilogy(sub_x,u,'linewidth',1.5,'color','b');
% legend('subpixle')


% zero-pad the object to avoid convolution artifacts
% kernelsize = params.dist*params.wavlen/params.pxsize/2; % diffraction kernel size
nullpixels = 0;

Q11  = @(x,k) propagate(x,dist(k),params.pxsize11,params.wavlen(k),params.method);
Q  = @(x,k) propagate(x,dist(k),params.pxsize,params.wavlen(k),params.method);
QH11 = @(x,k) propagate(x,-dist(k),params.pxsize11,params.wavlen(k),params.method);
QH = @(x,k) propagate(x,-dist(k),params.pxsize,params.wavlen(k),params.method);

% Q  = @(x,k) propagate(x.*mask(:,:,k), params.dist,params.pxsize,params.wavlen,params.method);
% QH = @(x,k) propagate(x,-params.dist,params.pxsize,params.wavlen,params.method).*conj(mask(:,:,k));
C  = @(x) imgcrop(x,nullpixels);
CT = @(x) zeropad(x,nullpixels);
% A11  = @(x,k) C(Q11(x,k));
% A  = @(x,k) C(Q(x,k));
A11  = @(x,k) Q11(x,k);
AH11  = @(x,k) QH11(x,k);
A  = @(x,k) Q(x,k);
AH = @(x,k) QH(x,k);

% generate data

%%
% =========================================================================
% Pixel super-resolution phase retrieval algorithms
% =========================================================================

% define a rectangular region for computing the errors
region.x1 = nullpixels+1;
region.x2 = nullpixels+n;
region.y1 = nullpixels+1;
region.y2 = nullpixels+n;

% algorithm settings
x_init = zeros(size(x));   % initial guess
x_init(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n) = 1;
% x_init = y_back(:,:,K);

lam = 1e-3;             % regularization parameter
gam = 2;                % step size (see the paper for details)
n_iters = 5;           % number of iterations (main loop)
n_subiters = 5;         % number of iterations (TV denoising)

% options
opts.verbose = true;                                % display status during the iterations
opts.errfunc = @(z) relative_error_2d(z,x,region);
% opts.errfunc = @(z,k) relative_error_2d(S((A(z,k))^2,sig),y(:,:,k),region);  % user-defined error metrics
opts.threshold = 1e-3;                              % threshold for step size update (for incremental algorithms)
opts.eta = 2;                                       % step size decrease ratio (for incremental algorithms)

% function handles
myF     = @(x) F(x,y,A,K,params,sig);                          % fidelity function 
mydF    = @(x) dF(x,y,A,AH,K,params,sig);                      % gradient of the fidelity function
mydFk   = @(x,k) dFk(x,y,A,AH,k,params,sig);                   % gradient of the fidelity function with respect to the k-th measurement
myR     = @(x) normTV(x,lam);                           % regularization function
myproxR = @(x,gamma) proxTV(x,gamma,lam,n_subiters);    % proximal operator for the regularization function

% run the algorithm
% [x_awf,J_awf,E_awf,runtimes_awf] = AWF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);     % AWF (accelerated Wirtinger flow)
% [x_wf, J_wf, E_wf, runtimes_wf ] = WF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);      % WF (Wirtinger flow)
[x_WFi,J_WFi,E_WFi,runtimes_WFi] = WFi(x_init,myF,mydFk,myR,myproxR,gam,n_iters,K,opts,xsub,ysub,params.pxsize,sig);  % WFi (Wirtinger flow with incremental updates)
% [x_WFi2,J_WFi2,E_WFi2,runtimes_WFi2] = WFi2(x_init,myF,mydFk,myR,myproxR,gam,n_iters,K,opts);  % WFi (Wirtinger flow with incremental updates)

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
% x_awf_crop = x_awf(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);
% x_wf_crop  = x_wf(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);
x_WFi_crop = x_WFi(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);
% x_WFi_crop2 = x_WFi2(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);

% visualize the reconstructed images
figure
% subplot(1,2,1),imshow(abs(x_awf_crop),[]);colorbar
% title(['Accelerated WF (Obj. Val. = ', num2str(J_awf(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
% % subplot(2,3,2),imshow(abs(x_wf_crop), []);colorbar
% title(['WF (Obj. Val. = ', num2str(J_wf(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
% subplot(1,2,1),imshow(abs(x_WFi_crop),[0.6/sig,1/sig]);colorbar
subplot(1,2,1),imshow(abs(x_WFi_crop),[]);colorbar
title(['Incremental WF (Obj. Val. = ', num2str(J_WFi(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
% subplot(1,3,2),imshow(abs(x_WFi_crop2),[0.6/sig,1/sig]);colorbar
% title(['Incremental WF (Obj. Val. = ', num2str(J_WFi(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
% subplot(1,2,2),imshow(angle(x_awf_crop),[]);colorbar
% subplot(2,3,5),imshow(angle(x_wf_crop), []);colorbar
% subplot(1,3,3),imshow(abs(x_WFi_crop-x_WFi_crop2),[]);colorbar
subplot(1,2,2),imshow(angle(x_WFi_crop),[]);colorbar
set(gcf,'unit','normalized','position',[0.15,0.2,0.7,0.6])
cell_phase = angle(x_WFi_crop);
cell_num = '1';
File_Name = ['cell_num_',cell_num];
save(File_Name,'cell_phase');

figure
% semilogy(0:n_iters,J_awf,'linewidth',1.5,'color','r');
% hold on,semilogy(0:n_iters,J_wf,'linewidth',1.5,'color','g');
hold on,semilogy(0:n_iters,J_WFi,'linewidth',1.5,'color','b');
legend('Error','WF','WFi')
figure;mesh(angle(x_WFi_crop));
%%
% =========================================================================
% Auxiliary functions
% =========================================================================

function v = F(x,y,A,K,params,sig)
% =========================================================================
% Data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - K   : Total measurement number.
%           - sig : Down-sampling ratio.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
v = 0;
for k = 1:K
    v = v + 1/K*Fk(x,y,A,k,params,sig);
end

end


function v = Fk(x,y,A,k,params,sig)
% =========================================================================
% Data-fidelity function w.r.t. the k-th measurement.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - k   : Measurement number of interest.
%           - sig : Down-sampling ratio.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
% v = 1/2 * norm2(sqrt(S(abs(A(x,k)).^2,sig)) - sqrt(y(:,:,k)))^2;
v = 1/2 * norm2(sqrt(S(abs(A(x.*exp(1i*angle(x)*(params.wavlen(1)/params.wavlen(k) - 1)),k)).^2,sig)) - sqrt(y(:,:,k)))^2;

function n = norm2(x)   % calculate the l2 vector norm
n = norm(x(:),2);
end

end


function g = dF(x,y,A,AH,K,params,sig)
% =========================================================================
% Gradient of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
%           - K   : Total measurement number.
%           - sig : Down-sampling ratio.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
g = zeros(size(x));
for k = 1:K
    g = g + 1/K*dFk(x,y,A,AH,k,params,sig);
end

end

% function g = dFk(x,y,A,AH,k,params,sig)
function g = dFk(x,y,A,AH,k,params,sig)
% =========================================================================
% Gradient of the data-fidelity function w.r.t. the k-th measurement.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
%           - k   : Measurement number of interest.
%           - sig : Down-sampling ratio.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
u = A(x.*exp(1i*angle(x)*(params.wavlen(1)/params.wavlen(k) - 1)),k);
a = sqrt(S(abs(u).^2,sig));
g = 1/2 * AH(u.*ST((1./a).*(a - sqrt(y(:,:,k))), sig), k);
% v = A(x,k);
% v1 = sqrt(S(abs(v).^2,sig));
% v = (1./v1).*(abs(v1) - sqrt(y(:,:,k))) .* exp(1i*angle(S(v,sig)));
% v = conj(x).* AH(ST(v,sig),k);
% g = (-1i/4) *((params.wavlen(1)/params.wavlen(k) + 1)* v + (-params.wavlen(1)/params.wavlen(k) + 1)*conj(v));
end


function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);

end


function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);

end