function [sharpness_cal] = NoG(re) %kernel_size,std,threshold

% 创建滤波器
% W = fspecial('gaussian',[60,60],20); 
% G = mat2gray(imfilter(holo, W, 'replicate'));
% figure;
% subplot(1,2,1); imshow(mat2gray(holo),[]);colorbar; title('原始图像');
% subplot(1,2,2); imshow(G,[]);colorbar;title('滤波后图像');
% 3.固定阈值分割方法
% sup=imbinarize(G,0.9);
% Diff_row_sup = abs((sup - circshift(sup,[1 0]))).^2;
% Diff_col_sup = abs((sup - circshift(sup,[0 1]))).^2;
% Gr_sup = Diff_row_sup+ Diff_col_sup;
% [r, c] = find(Gr_sup == 1);
% cut=re.*sup;
% figure,imshow(sup,[]);colorbar;title('固定阈值分割方法（取经验值为0.9）');
% Diff_row_1 = abs((re - circshift(re,[-1 -1]))).^2;
% Diff_row_2 = abs((re - circshift(re,[0 -1]))).^2;
% Diff_row_3 = abs((re - circshift(re,[1 1]))).^2;
Diff_row_4 = abs((re - circshift(re,[1 0]))).^2;
% Diff_col_1 = abs((re - circshift(re,[1 -1]))).^2;
Diff_col_2 = abs((re - circshift(re,[0 1]))).^2;
% Diff_col_3 = abs((re - circshift(re,[-1 1]))).^2;
% Diff_col_4 = abs((re - circshift(re,[-1 0]))).^2;
% Gr = Diff_row_1+Diff_row_2+Diff_row_3+Diff_row_4+Diff_col_1+Diff_col_2+Diff_col_3+Diff_col_4;
% Gr(sub2ind(size(Gr),r,c))=0;
Gr = Diff_row_4+Diff_col_2;
[~,S,~] = svd(Gr);
n=size(S,1);
% sharpness_cal=trace(S)/n;
sharpness_cal=trace(S)/n;
% disp(sharpness_cal);
end

