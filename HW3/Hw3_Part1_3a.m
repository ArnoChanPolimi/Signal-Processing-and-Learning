%% %%%%  Hw3-Part1-3)a  %%%%%%%%%
clc; clear; close all;

% 加载数据
load('GoogleDataset.mat');  % 确保数据文件存在

% 归一化后的数据合并
data_KPI = [Cost'; Click'; Conversion']';
data_KPI = (data_KPI - mean(data_KPI, 2)) ./ std(data_KPI, 0, 2); % 归一化

[num_samples, num_features] = size(data_KPI); % 数据维度 (20 列)


% 只关注 Click KPI 进行预测
click_KPI = Click; 

% 计算 Pearson 互相关矩阵
corr_matrix = corrcoef(click_KPI);

%% 1. 互相关热力图
figure;
imagesc(corr_matrix);  % 画相关性矩阵
colorbar;              % 添加颜色条
colormap(jet);         % 设置颜色映射
clim([-1 1]);         % 颜色范围 (-1 to 1)
title('Click KPI 互相关矩阵');
xlabel('Track Index');
ylabel('Track Index');

% **Step: 添加数值标注**
[num_rows, num_cols] = size(corr_matrix);
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'k');
    end
end

%% 2. 计算 Click KPI 的 19×20 线性预测系数矩阵
beta_matrix = zeros(num_features - 1, num_features);  
Y_pred = zeros(num_samples, num_features);  

for target_col = 1:num_features
    % 选择其他 19 列作为预测变量
    predictor_idx = setdiff(1:num_features, target_col);
    X = data_KPI(:, predictor_idx); % 预测变量
    Y = data_KPI(:, target_col);    % 目标变量

    % 计算线性回归系数 β (19×1)
    beta = (X' * X) \ (X' * Y);
    beta_matrix(:, target_col) = beta; % 存储系数

    % 计算预测值
    Y_pred(:, target_col) = X * beta;
end

%% 3. 输出 Click KPI 的线性回归系数矩阵
fprintf('Click KPI 线性回归系数矩阵 β (19x20):\n');
disp(beta_matrix);

%% 4. 绘制 Click KPI 的 20 张真实 vs. 预测值的对比图
figure;
for i = 1:num_features
    subplot(4, 5, i); % 4 行 5 列布局
    hold on;
    plot(1:num_samples, data_KPI(:, i), 'b', 'LineWidth', 1.5); % 真实值
    plot(1:num_samples, Y_pred(:, i), 'r--', 'LineWidth', 1.5); % 预测值
    title(['Click - Column ', num2str(i)]);
    xlabel('时间索引');
    ylabel('Clicks');
    legend('真实', '预测');
    grid on;
    hold off;
end
title('Click KPI 真实 vs. 预测值对比');
