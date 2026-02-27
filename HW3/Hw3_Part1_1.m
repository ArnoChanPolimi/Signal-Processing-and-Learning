clc; clear; close all;
data = load('GoogleDataset.mat');  % 加载数据

X = data.Click;  % 读取完整 524x20 数据
[N, K] = size(X);  % N=524, K=20

%% 计算瞬时线性预测系数矩阵 Beta (20x19)
Beta = zeros(K, K-1);  % 20 行，每行有 19 个系数
Y_pred = zeros(N, K);  % 存储所有预测值

for i = 1:K
    % 选取当前列为目标变量，去除自身后作为输入
    X_i = X(:, i);  % 目标变量
    X_minus_i = X(:, setdiff(1:K, i));  % 其余 19 列
    
    % 计算最小二乘估计: Beta_i = (X^T X)^(-1) X^T Y
    beta_i = (X_minus_i' * X_minus_i) \ (X_minus_i' * X_i);
    Beta(i, :) = beta_i';  % 存储系数
    
    % 计算预测值
    Y_pred(:, i) = X_minus_i * beta_i;
end

%% 计算相关性矩阵
corr_matrix = corr(X);

%% 绘制相关性热力图（带数值标注）
figure;
imagesc(corr_matrix);
colorbar;
clim([-1, 1]);
title('变量互相关矩阵');
xlabel('变量索引');
ylabel('变量索引');
xticks(1:K);
yticks(1:K);
for i = 1:K
    for j = 1:K
        text(j, i, sprintf('%.2f', corr_matrix(i, j)), 'FontSize', 10, 'HorizontalAlignment', 'center');
    end
end

%% 输出线性预测系数矩阵 Beta
fprintf('线性预测系数矩阵 (20x19):\n');
disp(Beta);

%% 绘制真实值 vs. 预测值对比图
figure;
for i = 1:K
    subplot(4, 5, i);
    plot(1:N, X(:, i), 'b', 'LineWidth', 1.5); hold on;
    plot(1:N, Y_pred(:, i), 'r--', 'LineWidth', 1.5);
    title(sprintf('变量 %d 预测 vs. 真实', i));
    xlabel('样本索引');
    ylabel('值');
    legend('真实值', '预测值');
    grid on;
end
hold off;
