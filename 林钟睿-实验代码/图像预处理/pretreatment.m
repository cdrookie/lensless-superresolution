close all
clear;clc;
Time = '20250114';
Test = '_1';
Test2 = '_2';
format_choose = 1;
Picture_format = ['.bmp'];

x0= 2764;
y0 = 3872;
K = 48;

ini_r = NaN(x0,y0,K);
ini_g = NaN(x0,y0,K);
ini_b = NaN(x0,y0,K);


start_y = 1359;
start_x = 2777;
show_size = 600;
ini_y = NaN(show_size,show_size,K);
r = NaN(show_size,show_size,K);
g = NaN(show_size,show_size,K);
b = NaN(show_size,show_size,K);
y = NaN(show_size,show_size,K);

hologram_r = NaN(x0,y0,7);
hologram_g = NaN(x0,y0,7);
hologram_b = NaN(x0,y0,7);
zero_padding = zeros(x0,y0);
background_r = NaN(x0,y0,7);
background_g = NaN(x0,y0,7);
background_b = NaN(x0,y0,7);
% 
for k=1:16
    background_r(:,:,k) = double(abs(imread(['./',Time,Test,'/back/','/r/',num2str(k),Picture_format(format_choose,:)]))); 
    ini_r(:,:,k) = double(abs(imread(['./',Time,Test,'/hologram/','/r/',num2str(k),Picture_format(format_choose,:)])));
    zero_padding((background_r(:,:,k)==0)&(ini_r(:,:,k)==0)) = 1; 
    background_r(:,:,k) = background_r(:,:,k)+zero_padding;
    hologram_r(:,:,k) = ini_r(:,:,k)./background_r(:,:,k);
end

figure;imshow(abs(hologram_r(:,:,1)),[],'border','tight')

for k=1:16
    background_g(:,:,k) = double(abs(imread(['./',Time,Test,'/back/','/g/',num2str(k),Picture_format(format_choose,:)]))); 
    ini_g(:,:,k) = double(abs(imread(['./',Time,Test,'/hologram/','/g/',num2str(k),Picture_format(format_choose,:)])));
    zero_padding((background_g(:,:,k)==0)&(ini_g(:,:,k)==0)) = 1; 
    background_g(:,:,k) = background_g(:,:,k)+zero_padding;
    hologram_g(:,:,k) = ini_g(:,:,k)./background_g(:,:,k);
end

for k=1:16
    background_b(:,:,k) = double(abs(imread(['./',Time,Test,'/back/','/b/',num2str(k),Picture_format(format_choose,:)]))); 
    ini_b(:,:,k) = double(abs(imread(['./',Time,Test,'/hologram/','/b/',num2str(k),Picture_format(format_choose,:)])));
    zero_padding((background_b(:,:,k)==0)&(ini_b(:,:,k)==0)) = 1; 
    background_b(:,:,k) = background_b(:,:,k)+zero_padding;
    hologram_b(:,:,k) = ini_b(:,:,k)./background_b(:,:,k);
end

ini_y(:,:,1) = hologram_r(start_y:(start_y+show_size-1),start_x:(start_x+show_size-1),1);
figure;imshow(abs(ini_y(:,:,1)),[],'border','tight')

xoffset_rt = NaN(1,16);
xoffset_gt = NaN(1,16);
xoffset_bt = NaN(1,16);
yoffset_rt = NaN(1,16);
yoffset_gt = NaN(1,16);
yoffset_bt = NaN(1,16);
for k = 1:16
    [xoffset_r,yoffset_r] = register(ini_y(:,:,1),hologram_r,k);
    [xoffset_g,yoffset_g] = register(ini_y(:,:,1),hologram_g,k);
    [xoffset_b,yoffset_b] = register(ini_y(:,:,1),hologram_b,k);
    disp(['xoffset_r=',num2str(xoffset_r), ' yoffset_r=',num2str(yoffset_r)])
    disp(['xoffset_g=',num2str(xoffset_g), ' yoffset_g=',num2str(yoffset_g)])
    disp(['xoffset_b=',num2str(xoffset_b), ' yoffset_b=',num2str(yoffset_b)])
    r(:,:,k) = hologram_r(yoffset_r:(yoffset_r+show_size-1),xoffset_r:(xoffset_r+show_size-1),k);
    g(:,:,k) = hologram_g(yoffset_g:(yoffset_g+show_size-1),xoffset_g:(xoffset_g+show_size-1),k);
    b(:,:,k) = hologram_b(yoffset_b:(yoffset_b+show_size-1),xoffset_b:(xoffset_b+show_size-1),k);
    xoffset_rt(k) =  xoffset_r;
    xoffset_gt(k) =  xoffset_g;
    xoffset_bt(k) =  xoffset_b;
    yoffset_rt(k) =  yoffset_r;
    yoffset_gt(k) =  yoffset_g;
    yoffset_bt(k) =  yoffset_b;
end
xoffset = [xoffset_rt,xoffset_gt,xoffset_bt];
yoffset = [yoffset_rt,yoffset_gt,yoffset_bt];


for i = 1:16
    y(:,:,i) = r(:,:,i);
end
for i = 17:32
    y(:,:,i) = g(:,:,i-4);
end
for i = 33:48
    y(:,:,i) = b(:,:,i-8);
end



for k = 1:K
figure;imshow(abs(y(:,:,k)),[],'border','tight')
end


File_Name = ['Filter_Holo_',Time,Test];
save(File_Name,'y');
