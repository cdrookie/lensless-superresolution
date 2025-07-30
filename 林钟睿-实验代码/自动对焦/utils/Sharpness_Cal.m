function result = Sharpness_Cal(Image_Complex, mode) 

[row,col]=size(Image_Complex);
matrix_select = zeros([row,col]);
matrix_select(2:row-1,2:col-1) = 1;

if strcmp(mode,'ToG')
    Diff_row = abs((Image_Complex - circshift(Image_Complex,[1 0]))).^2;
    Diff_col = abs((Image_Complex - circshift(Image_Complex,[0 1]))).^2;
    Diff = sqrt(Diff_row.*matrix_select+ Diff_col.*matrix_select);
    S = sqrt(std2(Diff)/mean(mean(Diff)));
end 

Image_Real = abs(Image_Complex);

if strcmp(mode,'Tum')
    S = sqrt(std2(Diff)/mean(mean(Diff)));
end 

if strcmp(mode,'VAR')
    S = sum(sum((Image_Real - mean(Image_Real(:))).^2 ))/row/col;
end 
if strcmp(mode,'GRA')
    Diff_row = (Image_Real - circshift(Image_Real,[1 0])).^2;
    Diff_col = (Image_Real - circshift(Image_Real,[0 1])).^2;
    Diff = sqrt(Diff_row.*matrix_select+ Diff_col.*matrix_select);
    S = sum(Diff(:))/row/col;
end 
if strcmp(mode,'LAP')
    kenel = [0,1,0;1,-4,1;0,1,0];
    Diff = conv2(Image_Real,kenel,'same');
    S = sum(sum((Diff.*matrix_select).^2))/row/col;
end 
if strcmp(mode,'LAP_ToG')
    kenel = [0,1,0;1,-4,1;0,1,0];
    Diff = conv2(Image_Real,kenel,'same');
    Diff = (Diff.*matrix_select).^2;
    S = sqrt(std2(Diff)/mean(mean(Diff)));
%     S = sum(sum((Diff.*matrix_select).^2))/row/col;
end 
if strcmp(mode,'TI')
    S = sqrt(std2(Image_Real.^2)/mean(mean(Image_Real.^2)));
end 
if strcmp(mode,'SOBEL')
    F2 = double(Image_Real);        
    U = double(Image_Real); 
    uSobel = zeros(row,col);
    for i = 2:row - 1   %sobel±ßÔµ¼ì²â
        for j = 2:col - 1
            Gx = (U(i+1,j-1) + 2*U(i+1,j) + F2(i+1,j+1)) - (U(i-1,j-1) + 2*U(i-1,j) + F2(i-1,j+1));
            Gy = (U(i-1,j+1) + 2*U(i,j+1) + F2(i+1,j+1)) - (U(i-1,j-1) + 2*U(i,j-1) + F2(i+1,j-1));
            uSobel(i,j) = sqrt(Gx^2 + Gy^2); 
        end
    end 
%     figure;imshow(uSobel,[]);
    S = sum(uSobel(:))/row/col;
end
result = gather(S);
% result = S;

