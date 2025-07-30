% 快速测试脚本 - 检查环境和依赖
% 这个脚本用于验证所有必要的函数和文件是否存在

clear; clc; close all;

fprintf('=== 自动化全息图处理系统 - 环境检查 ===\n\n');

%% 检查MATLAB版本
matlab_version = version('-release');
fprintf('MATLAB版本: %s\n', matlab_version);

%% 检查当前工作目录
current_dir = pwd;
fprintf('当前工作目录: %s\n', current_dir);

%% 检查必要的文件夹是否存在
required_folders = {
    '图像预处理',
    '自动对焦',
    '重建算法',
    fullfile('自动对焦', 'utils'),
    fullfile('重建算法', 'utils'),
    fullfile('重建算法', 'src'),
    fullfile('重建算法', 'src', 'func')
};

fprintf('\n检查文件夹结构:\n');
all_folders_exist = true;
for i = 1:length(required_folders)
    if exist(required_folders{i}, 'dir')
        fprintf('  ✓ %s\n', required_folders{i});
    else
        fprintf('  ✗ %s (缺失)\n', required_folders{i});
        all_folders_exist = false;
    end
end

%% 检查关键函数文件
required_functions = {
    fullfile('图像预处理', 'register.m'),
    fullfile('自动对焦', 'utils', 'MyMakingPhase.m'),
    fullfile('自动对焦', 'utils', 'NoG.m'),
    fullfile('自动对焦', 'utils', 'Sharpness_Cal.m'),
    fullfile('重建算法', 'utils', 'propagate.m'),
    fullfile('重建算法', 'utils', 'S.m'),
    fullfile('重建算法', 'utils', 'ST.m'),
    fullfile('重建算法', 'utils', 'subpixle1.m'),
    fullfile('重建算法', 'utils', 'relative_error_2d.m'),
    fullfile('重建算法', 'src', 'WFi.m'),
    fullfile('重建算法', 'src', 'func', 'normTV.m'),
    fullfile('重建算法', 'src', 'func', 'proxTV.m')
};

fprintf('\n检查关键函数文件:\n');
all_functions_exist = true;
for i = 1:length(required_functions)
    if exist(required_functions{i}, 'file')
        fprintf('  ✓ %s\n', required_functions{i});
    else
        fprintf('  ✗ %s (缺失)\n', required_functions{i});
        all_functions_exist = false;
    end
end

%% 检查主脚本文件
main_scripts = {
    'automated_hologram_processing.m',
    'run_automated_processing.m'
};

fprintf('\n检查主脚本文件:\n');
all_scripts_exist = true;
for i = 1:length(main_scripts)
    if exist(main_scripts{i}, 'file')
        fprintf('  ✓ %s\n', main_scripts{i});
    else
        fprintf('  ✗ %s (缺失)\n', main_scripts{i});
        all_scripts_exist = false;
    end
end

%% 测试路径添加
fprintf('\n测试路径添加:\n');
try
    script_path = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(script_path, '图像预处理')))
    addpath(genpath(fullfile(script_path, '自动对焦', 'utils')))
    addpath(genpath(fullfile(script_path, '重建算法', 'utils')))
    addpath(genpath(fullfile(script_path, '重建算法', 'src')))
    addpath(genpath(fullfile(script_path, '重建算法', 'src', 'func')))
    fprintf('  ✓ 路径添加成功\n');
catch ME
    fprintf('  ✗ 路径添加失败: %s\n', ME.message);
end

%% 测试关键函数是否可调用
fprintf('\n测试关键函数可用性:\n');
test_functions = {
    'MyMakingPhase', 'NoG', 'register', 'propagate', 'S', 'ST'
};

for i = 1:length(test_functions)
    if exist(test_functions{i}, 'file') == 2
        fprintf('  ✓ %s 可用\n', test_functions{i});
    else
        fprintf('  ✗ %s 不可用\n', test_functions{i});
    end
end

%% 总结
fprintf('\n=== 检查总结 ===\n');
if all_folders_exist && all_functions_exist && all_scripts_exist
    fprintf('✓ 环境检查通过！系统已准备就绪。\n');
    fprintf('您可以运行 run_automated_processing.m 开始处理。\n');
else
    fprintf('✗ 环境检查未通过。请确保所有必要的文件和文件夹都存在。\n');
    if ~all_folders_exist
        fprintf('  - 缺少必要的文件夹\n');
    end
    if ~all_functions_exist
        fprintf('  - 缺少必要的函数文件\n');
    end
    if ~all_scripts_exist
        fprintf('  - 缺少主脚本文件\n');
    end
end

fprintf('\n检查完成。\n');
