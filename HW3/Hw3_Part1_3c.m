%% %%%%  Hw3-Part1-3)c  %%%%%%%%%
% clear; clc; close all;

% 加载数据
load('GoogleDataset.mat');  % 确保数据文件存在

% 选取 Click 指标的控制 KPI（1:19列）
All_KPI = [Conversion',Click',Cost']';  
num_tracks = size(All_KPI, 2); % 轨迹数量（19个企业）
All_KPI = (All_KPI - mean(All_KPI, 2)) ./ std(All_KPI, 0, 2);

% **Step 6: 估计广告开始时间 T_K**
target_idx = 20;  % 目标 track: track 20
predictor_idx = [2,5,12,13,14,18,19];  % 预测变量（track 1~19）
%predictor_idx = [1:19];  % 预测变量（track 1~19）


% 构建数据集
X_train = All_KPI(:, predictor_idx);  % 预测变量（所有其他 track）
y_train = All_KPI(:, target_idx);  % 目标 track 20

% 线性回归估计参数
beta = (X_train' * X_train) \ (X_train' * y_train);
y_pred = X_train * beta; % 预测结果

% **Step 7: 计算误差**
prediction_error = abs(y_train - y_pred);  % 计算误差

% **Step 8: 平滑误差曲线**
window_size = 50;  % 设定滑动窗口大小
smoothed_error = movmean(prediction_error, window_size);  % 计算滑动均值

% **Step 9: 计算误差变化率**
error_diff = diff(smoothed_error);  % 计算误差的一阶差分

% **Step 10: 识别最长的线性上升段**
rising_start = [];  % 记录最长上升段的起点
max_rising_length = 0;  % 记录最长上升段的长度
current_length = 0;
current_start = 1;  % 记录当前上升段的起点

for t = 2:length(error_diff)
    if error_diff(t-1) > 0  % 误差在上升
        current_length = current_length + 1;
    else  % 误差下降或持平
        if current_length > max_rising_length  % 记录最长上升段
            max_rising_length = current_length;
            rising_start = current_start;
        end
        current_length = 0;
        current_start = t + 1;
    end
end

% 确定最终 T_K
T_K_est = rising_start;

% **Step 11: 画出误差随时间的变化**
figure;
hold on;
plot(smoothed_error, 'b', 'LineWidth', 1.5);
xline(T_K_est, 'g', 'LineWidth', 2); % 估计的 T_K
title(['Estimated T_K = ', num2str(T_K_est)]);
xlabel('Time (Days)');
ylabel('Smoothed Prediction Error');
legend('Smoothed Error', 'Estimated T_K');
hold off;

% **Step 12: 画出真实 Click vs 预测 Click**
figure;
hold on;
plot(y_train, 'b', 'LineWidth', 1.5);  % 真实值
plot(y_pred, 'r--', 'LineWidth', 1.5); % 预测值
xline(T_K_est, 'g', 'LineWidth', 2); % 估计的 T_K
title(['Click KPI Prediction - Estimated T_K = ', num2str(T_K_est)]);
xlabel('Time (Days)');
ylabel('Clicks');
legend('Actual Clicks', 'Predicted Clicks', 'Estimated T_K');
hold off;