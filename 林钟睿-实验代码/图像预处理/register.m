function [xoffSet,yoffSet] = register(sub_onion,sub_peppers,k)
c = normxcorr2(sub_onion,sub_peppers(:,:,k));
% figure
% mesh(c);
% shading flat
% offset found by correlation
[ypeak,xpeak] = find(c==max(c(:)));
yoffSet = ypeak-size(sub_onion,1);
xoffSet = xpeak-size(sub_onion,2);