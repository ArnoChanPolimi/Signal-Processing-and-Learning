%%%%%%%%%%%% Hw3 - Part1.1  %%%%%%%%%%%%%%%%%
%%%%%%%%%  GooooooooooooooooD %%%%%%%%%%%%%%%%
clc; clear; close all;
data = load('GoogleDataset.mat');  % 提取 Click / Cost / Conversion

X = data.Click;  % 选择要分析的KPI数据（也可换成 Cost 或 Conversion）
target_col = 20;       % 第20列是目标公司

T20_cut = 397;           % 广告激活时间点

ws_list = [10, 20, 30, 50, 80];
gap_list = [1, 3, 5, 10];
result = zeros(length(ws_list), length(gap_list));

for i = 1:length(ws_list)
    for j = 1:length(gap_list)
        [~, ~, mse_val, ~] = causal_linear_predictor(X, 20, ws_list(i), gap_list(j), T20_cut);
        result(i, j) = mse_val;
    end
end

function [true_vals, pred_vals, mse_val, alpha] = causal_linear_predictor(X, target_col, window_size, pred_gap, T_cut)
% 功能：
%   用过去的真实 KPI 数据训练一个固定的线性预测器，
%   在广告后时刻只能用已估计的结果进行递推预测
% 输入：
%   X           : N x K 的KPI数据
%   target_col  : 要预测的目标列
%   window_size : 用于预测的历史长度
%   pred_gap    : 要预测的未来间隔
%   T_cut       : 广告激活时间点（不能使用之后真实数据训练或预测）
%
% 输出：
%   true_vals   : 原始真实 KPI（用于对比）
%   pred_vals   : 估计的预测值（广告后递推）
%   mse_val     : 广告前验证的 MSE
%   alpha       : 学到的预测系数（固定）

% Extract target column
x = X(:, target_col);
N = length(x);
true_vals = x;

% Build training data
X_train = [];
Y_train = [];

% Ensure sufficient training samples
if T_cut <= window_size + pred_gap
    error('T_cut is too small. Must satisfy T_cut > window_size + pred_gap');
end

% Collect training samples: using true values
for n = window_size : (T_cut - pred_gap)
    x_window = x(n:-1:n - window_size + 1);   % reverse-order window
    x_target = x(n + pred_gap);              % the target point to predict

    X_train = [X_train; x_window'];
    Y_train = [Y_train; x_target];
end

% Training phase: solve for optimal linear coefficients (weights)
alpha = (X_train' * X_train) \ (X_train' * Y_train);  % fixed alpha

% Initialize container for predicted values
pred_vals = nan(N, 1);

% Phase 1: when t + pred_gap ≤ T_cut, use true values for prediction
for n = window_size : (T_cut - pred_gap)
    x_input = x(n:-1:n - window_size + 1);
    x_hat = alpha' * x_input;
    pred_vals(n + pred_gap) = x_hat;
end

% Phase 2: after the ad starts, recursive prediction (only using estimated values)
for n = (T_cut - window_size + 1) : (N - pred_gap)

    x_input = zeros(window_size, 1);  % initialize input vector

    for i = 0:(window_size - 1)
        idx = n - i;

        if idx <= T_cut
            x_input(i+1) = x(idx);          % before ad: use true values
        else
            x_input(i+1) = pred_vals(idx);  % after ad: use predicted values
        end
    end

    x_hat = alpha' * x_input;
    pred_vals(n + pred_gap) = x_hat;
end

% MSE evaluation only over verifiable region (before ad)
valid_idx = (window_size + pred_gap):(T_cut);
mse_val = mean((pred_vals(valid_idx) - x(valid_idx)).^2);

% Visualization
figure;
plot(x, 'b', 'LineWidth', 1.5); hold on;
plot(pred_vals, 'r--', 'LineWidth', 1.5);
xline(T_cut, '--k', 'Ad Activation Point', 'LineWidth', 1.5);
title(sprintf('Linear Prediction (W=%d, Gap=%d), MSE=%.4f', window_size, pred_gap, mse_val));
xlabel('Time / Day'); ylabel('KPI Value');
legend('True Values', 'Predicted Values'); grid on;

end
