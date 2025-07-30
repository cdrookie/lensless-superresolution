function automated_hologram_processing(data_folder, crop_size)
% AUTOMATED_HOLOGRAM_PROCESSING 全自动全息图处理脚本
% 输入参数：
%   data_folder: 包含全息图数据的文件夹路径
%   crop_size: 裁剪大小 (默认为600)
%
% 功能：
% 1. 自动预处理图像（包含自动裁剪）
% 2. 自动对焦（对三种颜色分别进行）
% 3. 使用获得的焦点值进行重建算法
%
% 使用示例：
% automated_hologram_processing('20250114_1', 600)

clear; clc; close all;

% 默认参数设置
if nargin < 2
    crop_size = 600;
end

% 添加必要的路径
script_path = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_path, '图像预处理')))
addpath(genpath(fullfile(script_path, '自动对焦', 'utils')))
addpath(genpath(fullfile(script_path, '重建算法', 'utils')))
addpath(genpath(fullfile(script_path, '重建算法', 'src')))
addpath(genpath(fullfile(script_path, '重建算法', 'src', 'func')))

fprintf('开始自动化全息图处理...\n');

%% 第一步：预处理和自动裁剪
fprintf('步骤1: 图像预处理和自动裁剪...\n');
[y, best_crop_x, best_crop_y] = preprocess_with_auto_crop(data_folder, crop_size);

%% 第二步：自动对焦
fprintf('步骤2: 自动对焦...\n');
optimal_distances = auto_focus_all_channels(y);

%% 第三步：重建算法
fprintf('步骤3: 运行重建算法...\n');
reconstruction_result = run_reconstruction_algorithm(y, optimal_distances);

fprintf('自动化处理完成！\n');
fprintf('最佳裁剪位置: x=%d, y=%d\n', best_crop_x, best_crop_y);
fprintf('最佳焦点距离: 红色=%.2f, 绿色=%.2f, 蓝色=%.2f\n', ...
    optimal_distances(1), optimal_distances(2), optimal_distances(3));

end

%% 子函数：带自动裁剪的预处理
function [y, best_crop_x, best_crop_y] = preprocess_with_auto_crop(data_folder, crop_size)
    
    Picture_format = '.bmp';
    
    % 获取图像尺寸（假设从第一张图像获得）
    sample_img = imread(['./', data_folder, '/hologram/r/1', Picture_format]);
    [x0, y0] = size(sample_img);
    
    % 修改为12张图像（每种颜色4张）
    K = 12;
    
    ini_r = NaN(x0, y0, 4);
    ini_g = NaN(x0, y0, 4);
    ini_b = NaN(x0, y0, 4);
    
    hologram_r = NaN(x0, y0, 4);
    hologram_g = NaN(x0, y0, 4);
    hologram_b = NaN(x0, y0, 4);
    
    background_r = NaN(x0, y0, 4);
    background_g = NaN(x0, y0, 4);
    background_b = NaN(x0, y0, 4);
    
    % 读取红色通道图像（4张）
    for k = 1:4
        background_r(:,:,k) = double(abs(imread(['./', data_folder, '/back/r/', num2str(k), Picture_format]))); 
        ini_r(:,:,k) = double(abs(imread(['./', data_folder, '/hologram/r/', num2str(k), Picture_format])));
        zero_padding = zeros(x0, y0);
        zero_padding((background_r(:,:,k)==0)&(ini_r(:,:,k)==0)) = 1; 
        background_r(:,:,k) = background_r(:,:,k) + zero_padding;
        hologram_r(:,:,k) = ini_r(:,:,k) ./ background_r(:,:,k);
    end
    
    % 读取绿色通道图像（4张）
    for k = 1:4
        background_g(:,:,k) = double(abs(imread(['./', data_folder, '/back/g/', num2str(k), Picture_format]))); 
        ini_g(:,:,k) = double(abs(imread(['./', data_folder, '/hologram/g/', num2str(k), Picture_format])));
        zero_padding = zeros(x0, y0);
        zero_padding((background_g(:,:,k)==0)&(ini_g(:,:,k)==0)) = 1; 
        background_g(:,:,k) = background_g(:,:,k) + zero_padding;
        hologram_g(:,:,k) = ini_g(:,:,k) ./ background_g(:,:,k);
    end
    
    % 读取蓝色通道图像（4张）
    for k = 1:4
        background_b(:,:,k) = double(abs(imread(['./', data_folder, '/back/b/', num2str(k), Picture_format]))); 
        ini_b(:,:,k) = double(abs(imread(['./', data_folder, '/hologram/b/', num2str(k), Picture_format])));
        zero_padding = zeros(x0, y0);
        zero_padding((background_b(:,:,k)==0)&(ini_b(:,:,k)==0)) = 1; 
        background_b(:,:,k) = background_b(:,:,k) + zero_padding;
        hologram_b(:,:,k) = ini_b(:,:,k) ./ background_b(:,:,k);
    end
    
    % 自动寻找最佳裁剪区域
    [best_crop_x, best_crop_y] = find_best_crop_region(hologram_r(:,:,1), crop_size);
    
    % 使用找到的最佳位置进行裁剪和配准
    reference_img = hologram_r(best_crop_y:(best_crop_y+crop_size-1), best_crop_x:(best_crop_x+crop_size-1), 1);
    
    r = NaN(crop_size, crop_size, 4);
    g = NaN(crop_size, crop_size, 4);
    b = NaN(crop_size, crop_size, 4);
    
    % 对每张图像进行配准和裁剪
    for k = 1:4
        [xoffset_r, yoffset_r] = register(reference_img, hologram_r, k);
        [xoffset_g, yoffset_g] = register(reference_img, hologram_g, k);
        [xoffset_b, yoffset_b] = register(reference_img, hologram_b, k);
        
        r(:,:,k) = hologram_r(yoffset_r:(yoffset_r+crop_size-1), xoffset_r:(xoffset_r+crop_size-1), k);
        g(:,:,k) = hologram_g(yoffset_g:(yoffset_g+crop_size-1), xoffset_g:(xoffset_g+crop_size-1), k);
        b(:,:,k) = hologram_b(yoffset_b:(yoffset_b+crop_size-1), xoffset_b:(xoffset_b+crop_size-1), k);
    end
    
    % 组装最终的y矩阵（12张图像）
    y = NaN(crop_size, crop_size, K);
    for i = 1:4
        y(:,:,i) = r(:,:,i);        % 红色：1-4
    end
    for i = 5:8
        y(:,:,i) = g(:,:,i-4);      % 绿色：5-8
    end
    for i = 9:12
        y(:,:,i) = b(:,:,i-8);      % 蓝色：9-12
    end
    
    % 保存预处理结果
    time_stamp = strsplit(data_folder, '_');
    File_Name = ['Filter_Holo_', time_stamp{1}, '_', time_stamp{2}];
    save(File_Name, 'y');
    
    fprintf('预处理完成，已保存到 %s.mat\n', File_Name);
end

%% 子函数：自动寻找最佳裁剪区域
function [best_x, best_y] = find_best_crop_region(img, crop_size)
    [height, width] = size(img);
    
    % 确保裁剪区域不超出图像边界
    max_x = width - crop_size + 1;
    max_y = height - crop_size + 1;
    
    % 设置搜索步长（可以调整以平衡速度和精度）
    step_size = 50;
    
    best_score = 0;
    best_x = 1;
    best_y = 1;
    
    % 在图像中搜索最佳裁剪区域
    for y = 1:step_size:max_y
        for x = 1:step_size:max_x
            % 裁剪当前区域
            crop_region = img(y:(y+crop_size-1), x:(x+crop_size-1));
            
            % 计算该区域的信号强度（可以使用方差、梯度等指标）
            score = calculate_signal_strength(crop_region);
            
            if score > best_score
                best_score = score;
                best_x = x;
                best_y = y;
            end
        end
    end
    
    fprintf('找到最佳裁剪位置: x=%d, y=%d, 信号强度=%.4f\n', best_x, best_y, best_score);
end

%% 子函数：计算信号强度
function score = calculate_signal_strength(img)
    % 使用多种指标的组合来评估信号强度
    
    % 1. 方差（信号变化程度）
    variance_score = var(img(:));
    
    % 2. 梯度强度
    [Gx, Gy] = gradient(img);
    gradient_score = mean(sqrt(Gx(:).^2 + Gy(:).^2));
    
    % 3. 频域能量
    fft_img = fft2(img);
    freq_energy = sum(abs(fft_img(:)).^2);
    
    % 综合评分
    score = variance_score * 0.4 + gradient_score * 0.4 + freq_energy * 0.2;
end

%% 子函数：自动对焦（所有通道）
function optimal_distances = auto_focus_all_channels(y)
    
    % 全息图参数
    params.dwavlen = [0.625, 0.525, 0.470]; % 红、绿、蓝波长(um)
    [row, col, K] = size(y);
    params.pxsize = 1.67; % 像素尺寸(um)
    
    % 对焦距离搜索范围
    dist_range = 1200:5:1700; % 距离范围(um)
    find_num = length(dist_range);
    
    % 为三种颜色分别进行自动对焦
    auto_channels = [1, 5, 9]; % 红、绿、蓝的代表通道
    channel_names = {'红色', '绿色', '蓝色'};
    optimal_distances = zeros(1, 3);
    
    for color_idx = 1:3
        fprintf('正在对焦 %s 通道...\n', channel_names{color_idx});
        
        auto_channel = auto_channels(color_idx);
        wavelength = params.dwavlen(color_idx);
        
        res_sharpness = zeros(find_num, 1);
        
        % 对每个距离计算清晰度
        for i = 1:find_num
            dphase = MyMakingPhase(row, col, dist_range(i), wavelength, params.pxsize);
            re = propagate_simple(y(:,:,auto_channel), dphase, 'B');
            res_sharpness(i) = NoG(re);
        end
        
        % 找到最佳焦点
        [~, max_idx] = max(res_sharpness);
        optimal_distances(color_idx) = dist_range(max_idx);
        
        % 绘制对焦曲线
        figure;
        plot(dist_range, res_sharpness, 'LineWidth', 2);
        xlabel('距离 (μm)');
        ylabel('清晰度 (NOG)');
        title(sprintf('%s通道自动对焦结果', channel_names{color_idx}));
        grid on;
        hold on;
        plot(optimal_distances(color_idx), res_sharpness(max_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        legend('清晰度曲线', sprintf('最佳焦点: %.1f μm', optimal_distances(color_idx)));
        
        fprintf('%s通道最佳焦点: %.1f μm\n', channel_names{color_idx}, optimal_distances(color_idx));
    end
end

%% 子函数：运行重建算法
function reconstruction_result = run_reconstruction_algorithm(y, optimal_distances)
    
    K = size(y, 3); % 现在是12张图像
    sig = 1; % 下采样比例
    m = size(y(:,:,1), 1); % 传感器分辨率
    n = m * sig; % 样本分辨率
    
    x = zeros(n, n);
    
    % 物理参数
    params.pxsize11 = 1.67e-3; % 像素尺寸 (mm)
    params.pxsize = params.pxsize11 / sig;
    params.wavlen = NaN(K, 1);
    
    % 为12张图像设置波长
    for i = 1:4
        params.wavlen(i) = 0.625e-3; % 红色
    end
    for i = 5:8
        params.wavlen(i) = 0.525e-3; % 绿色
    end
    for i = 9:12
        params.wavlen(i) = 0.470e-3; % 蓝色
    end
    
    params.method = 'Angular Spectrum';
    
    % 设置距离（转换为mm）
    params.dist = optimal_distances(1) * 1e-3;   % 红色
    params.dist2 = optimal_distances(2) * 1e-3;  % 绿色  
    params.dist3 = optimal_distances(3) * 1e-3;  % 蓝色
    
    dist = NaN(1, K);
    for k = 1:4
        dist(k) = params.dist;
    end
    for k = 5:8
        dist(k) = params.dist2;
    end
    for k = 9:12
        dist(k) = params.dist3;
    end
    
    % 子像素配准
    sub_x = -params.pxsize11/2:0.167e-3:params.pxsize11/2;
    sub_y = -params.pxsize11/2:0.167e-3:params.pxsize11/2;
    [Xsub, Ysub] = meshgrid(sub_x, sub_y);
    len_sub_pixel = length(sub_x);
    u = zeros(len_sub_pixel, len_sub_pixel, K);
    xsub = NaN(1, K);
    ysub = NaN(1, K);
    
    for i = 1:len_sub_pixel
        for j = 1:len_sub_pixel
           u(i,j,:) = subpixle1_modified(sub_x(i), sub_x(j), y, m, m, params.pxsize11, K);
        end
    end
    
    for i = 1:K
       [p, q] = find(u(:,:,i) == min(min(u(:,:,i))));
       xsub(i) = sub_x(p);
       ysub(i) = sub_x(q);
    end
    
    % 传播算子
    nullpixels = 0;
    Q = @(x,k) propagate(x, dist(k), params.pxsize, params.wavlen(k), params.method);
    QH = @(x,k) propagate(x, -dist(k), params.pxsize, params.wavlen(k), params.method);
    A = @(x,k) Q(x,k);
    AH = @(x,k) QH(x,k);
    
    % 算法设置
    region.x1 = nullpixels + 1;
    region.x2 = nullpixels + n;
    region.y1 = nullpixels + 1;
    region.y2 = nullpixels + n;
    
    x_init = zeros(size(x));
    x_init(nullpixels+1:nullpixels+n, nullpixels+1:nullpixels+n) = 1;
    
    lam = 1e-3;
    gam = 2;
    n_iters = 5;
    n_subiters = 5;
    
    opts.verbose = true;
    opts.errfunc = @(z) relative_error_2d(z, x, region);
    opts.threshold = 1e-3;
    opts.eta = 2;
    
    % 函数句柄
    myF = @(x) F(x, y, A, K, params, sig);
    mydFk = @(x,k) dFk(x, y, A, AH, k, params, sig);
    myR = @(x) normTV(x, lam);
    myproxR = @(x,gamma) proxTV(x, gamma, lam, n_subiters);
    
    % 运行算法
    [x_WFi, J_WFi, E_WFi, runtimes_WFi] = WFi(x_init, myF, mydFk, myR, myproxR, gam, n_iters, K, opts, xsub, ysub, params.pxsize, sig);
    
    % 裁剪结果
    x_WFi_crop = x_WFi(nullpixels+1:nullpixels+n, nullpixels+1:nullpixels+n);
    
    % 显示结果
    figure
    subplot(1,2,1), imshow(abs(x_WFi_crop), []); colorbar
    title(['重建振幅 (目标值 = ', num2str(J_WFi(end),'%4.3f'), ')'], 'interpreter', 'latex', 'fontsize', 14)
    subplot(1,2,2), imshow(angle(x_WFi_crop), []); colorbar
    title('重建相位', 'interpreter', 'latex', 'fontsize', 14)
    set(gcf, 'unit', 'normalized', 'position', [0.15, 0.2, 0.7, 0.6])
    
    % 保存结果
    cell_phase = angle(x_WFi_crop);
    cell_num = '1';
    File_Name = ['cell_num_', cell_num];
    save(File_Name, 'cell_phase');
    
    % 绘制收敛曲线
    figure
    semilogy(0:n_iters, J_WFi, 'linewidth', 1.5, 'color', 'b');
    xlabel('迭代次数');
    ylabel('目标函数值');
    title('算法收敛曲线');
    grid on;
    
    reconstruction_result = struct('amplitude', abs(x_WFi_crop), 'phase', angle(x_WFi_crop), 'objective', J_WFi);
end

%% 辅助函数（从原代码复制）
function [xoffSet, yoffSet] = register(sub_onion, sub_peppers, k)
    c = normxcorr2(sub_onion, sub_peppers(:,:,k));
    [ypeak, xpeak] = find(c == max(c(:)));
    yoffSet = ypeak - size(sub_onion, 1);
    xoffSet = xpeak - size(sub_onion, 2);
end

function v = F(x, y, A, K, params, sig)
    v = 0;
    for k = 1:K
        v = v + 1/K * Fk(x, y, A, k, params, sig);
    end
end

function v = Fk(x, y, A, k, params, sig)
    v = 1/2 * norm2(sqrt(S(abs(A(x.*exp(1i*angle(x)*(params.wavlen(1)/params.wavlen(k) - 1)),k)).^2,sig)) - sqrt(y(:,:,k)))^2;
    
    function n = norm2(x)
        n = norm(x(:), 2);
    end
end

function g = dFk(x, y, A, AH, k, params, sig)
    u = A(x.*exp(1i*angle(x)*(params.wavlen(1)/params.wavlen(k) - 1)), k);
    a = sqrt(S(abs(u).^2, sig));
    g = 1/2 * AH(u.*ST((1./a).*(a - sqrt(y(:,:,k))), sig), k);
end

%% 简化的传播函数
function w_o = propagate_simple(w_i, phase, method)
    inputFT = fftshift(fft2(w_i));
    
    if strcmp(method, 'F')   
        H = phase;
    elseif strcmp(method, 'B')
        H = conj(phase);
    end
    
    w_o = ifft2(fftshift(inputFT .* H));
end
