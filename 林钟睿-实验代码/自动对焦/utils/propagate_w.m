function [ w_o ] = propagate_w( w_i, k, s, phase, method )
% ----------------------------------------------------------------
% This function numerically simulates the free-space propagation of a 
% complex wavefield.
% -----------------------------------------------------------------
%   INPUT   w_i    : Input complex wavefield
%           dist   : Propagation distance
%           pxsize : Pixel size
%           wavlen : Wavelength
%           mtehod : Numerical method ('Fresnel' or 'Angular Spectrum')
%   OUTPUT  w_o    : Output wavefield after propagation
% -----------------------------------------------------------------                                                                                                                   function [ w_o ] = propagate( w_i, dist, pxsize, wavlen, method )
H = phase(:,:,s,k);

inputFT = fftshift(fft2(w_i));

if(method == 'F')   
    H = H;
elseif(method == 'B')
    H = conj(H);
end


w_o = ifft2(fftshift(inputFT.*H));

end

