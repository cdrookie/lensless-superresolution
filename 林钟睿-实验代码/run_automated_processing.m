% 自动化全息图处理使用示例
% 
% 使用前请确保：
% 1. 数据文件夹结构正确（参见README.md）
% 2. 所有必要的工具函数都在相应目录中
% 
% 数据文件夹结构示例：
%    20250114_1/
%    ├── back/
%    │   ├── r/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)
%    │   ├── g/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)
%    │   └── b/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)
%    └── hologram/
%        ├── r/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)
%        ├── g/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)
%        └── b/ (1.bmp, 2.bmp, 3.bmp, 4.bmp)

clear; clc; close all;

%% 用户设置区域 - 请根据需要修改以下参数
% =================================================================

% 设置数据文件夹名称（请修改为您的实际文件夹名）
data_folder = '20250114_1';  

% 设置裁剪大小（建议范围：400-800像素）
crop_size = 600;  

% 可选：调整对焦搜索范围（单位：微米）
% 如果默认范围不合适，可以取消注释并修改下面的行
% focus_range = [1200, 1700];  % [最小距离, 最大距离]

% =================================================================

fprintf('=== 自动化全息图处理系统 ===\n');
fprintf('数据文件夹: %s\n', data_folder);
fprintf('裁剪大小: %d × %d 像素\n', crop_size, crop_size);
fprintf('开始处理...\n\n');

% 检查数据文件夹是否存在
if ~exist(data_folder, 'dir')
    error('错误：找不到数据文件夹 "%s"。请检查文件夹名称和路径。', data_folder);
end

% 检查必要的子文件夹是否存在
required_folders = {
    fullfile(data_folder, 'back', 'r'), 
    fullfile(data_folder, 'back', 'g'), 
    fullfile(data_folder, 'back', 'b'),
    fullfile(data_folder, 'hologram', 'r'), 
    fullfile(data_folder, 'hologram', 'g'), 
    fullfile(data_folder, 'hologram', 'b')
};

for i = 1:length(required_folders)
    if ~exist(required_folders{i}, 'dir')
        error('错误：找不到必要的子文件夹 "%s"。请检查文件夹结构。', required_folders{i});
    end
end

% 运行自动化处理
try
    tic;
    automated_hologram_processing(data_folder, crop_size);
    processing_time = toc;
    
    fprintf('\n=== 处理完成 ===\n');
    fprintf('总处理时间: %.1f 秒\n', processing_time);
    fprintf('生成的文件:\n');
    fprintf('  - Filter_Holo_%s.mat (预处理数据)\n', strrep(data_folder, '_', ''));
    fprintf('  - cell_num_1.mat (重建相位)\n');
    fprintf('  - 多个图形窗口 (结果可视化)\n');
    
catch ME
    fprintf('\n=== 处理失败 ===\n');
    fprintf('错误信息: %s\n', ME.message);
    fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
    fprintf('\n可能的解决方案:\n');
    fprintf('1. 检查数据文件夹路径和结构是否正确\n');
    fprintf('2. 确保所有.bmp文件都存在（每个子文件夹4个文件）\n');
    fprintf('3. 检查图像文件是否损坏\n');
    fprintf('4. 确保有足够的内存和磁盘空间\n');
    fprintf('5. 查看README.md获取详细说明\n');
end
