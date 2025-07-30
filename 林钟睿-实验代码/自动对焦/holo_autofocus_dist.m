clear;clc
close all
% load functions
addpath(genpath('./utils'))
% addpath(genpath('../src'))
% addpath(genpath('./MyMonoFunction'))
%% Holography Paprameters
Time = '20250114';
Test = '_1';
% data_path = '../data/experiment/';
File_Name = ['Filter_Holo_',Time,Test,'.mat'];
%加载滤除背景的全息图
load(File_Name);
% y = y(:,1:3000,:);
% figure;imshow(y(:,:,1),[0.2,2.9],'border','tight')
start_y = 1;
start_x = 1;
show_size = 600;
% y = y(start_y:(start_y+show_size),start_x:(start_x+show_size),:);
figure;imshow(abs(y(:,:,1)),[],'border','tight')

upsampling_ratio = 1;
% psr_y = zeros(size(y,1)*upsampling_ratio,size(y,2)*upsampling_ratio,size(y,3));
% for i = 1:size(y,3)
%     upsample_g = imresize(y(:,:,i),[size(y,1)*upsampling_ratio,size(y,2)*upsampling_ratio],'bicubic');
%     upsample_g(upsample_g<0) = 0;
% %     psr_y(:,:,i) = gpuArray(upsample_g);
%     psr_y(:,:,i) = upsample_g;
% end
% figure;imshow(plotdatacube(gather(psr_y)),[],'border','tight')
% y = psr_y;
%% parameter
params.dwavlen = [0.625,0.525,0.470];
[row,col,K]=size(y);
params.pxsize = 1.67/upsampling_ratio;% pixel pitch (um)
%Autofocusing distance parameter
% params.dist = 440;    %定义样品到ccd的距离(um)
% params.dist = 1380:1:1400;    %定义样品到ccd的距离(um)
params.dist = 1200:10:1700;    %定义样品到ccd的距离(um)
find_num = size(params.dist,2);
res_sharpness = zeros([find_num,4]);
auto_channel = 9;
%% Reconstructed by back-propagation
pic_num = 1;
figure;
for i = 1:find_num
    dphase = (MyMakingPhase(row,col,params.dist(i),params.dwavlen(3),params.pxsize));
    re = propagate_w(y(:,:,auto_channel), 1, 1, dphase, 'B');
    res_sharpness(i,1) = Sharpness_Cal(re, 'ToG');
    res_sharpness(i,2) = Sharpness_Cal(re, 'LAP');
    res_sharpness(i,3) = Sharpness_Cal(re, 'TI');
    res_sharpness(i,4) = NoG(re);
    imshow(abs(re),[]);
%     imshow(angle(re),[]);%,colorbar;
%     imshow(abs(re(start_y:(start_y+show_size),start_x:(start_x+show_size))),[]);
%     imshow(imresize(abs(re(start_y:(start_y+show_size),start_x:(start_x+show_size))),[512,512],'nearest'),[]);
    str=[num2str(params.dist(i))];
    text(30,30,str,'HorizontalAlignment','center','VerticalAlignment','middle','background','white')
%     m(i)=getframe;
%     pause(0.09);     %%%%    暂停时间
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if pic_num == 1
    imwrite(I,map,'test2.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
    imwrite(I,map,'test2.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    pic_num = pic_num + 1;

end

% movie(m,1,2);
%% Visual the focus value
figure; plot(params.dist,MaxMinNorm(res_sharpness(:,1),[0,1]),'r');
hold on,plot(params.dist,MaxMinNorm(res_sharpness(:,2),[0,1]),'g');
hold on,plot(params.dist,MaxMinNorm(res_sharpness(:,4),[0,1]),'b');
legend('ToG','LAP','NOG')

