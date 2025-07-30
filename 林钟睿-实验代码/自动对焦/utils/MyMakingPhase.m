function [Phase]=MyMakingPhase(Nx,Ny,z,lambda,delta)

k=1/lambda;

X=[ceil(-Nx/2):1:ceil(Nx/2-1)]'.*(1/(Nx*delta));
Y=[ceil(-Ny/2):1:ceil(Ny/2-1)].*(1/(Ny*delta));

kx=repmat(X,1,Ny);
ky=repmat(Y,Nx,1);
kp=sqrt(kx.^2+ky.^2);

term=k.^2-kp.^2;
term(term<0)=0;

Phase=exp(1i*2*pi*z*sqrt(term));