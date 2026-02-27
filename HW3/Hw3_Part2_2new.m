%% Hw3-Part2-2 %%

clc; clear; close all;

%% 参数设置
N = 800;            % 总数据点数
T0 = 400;           % 真实突变点
rho_true = 0.8;     % AR(1) 真实系数
phi_true = pi/4;    % AR(2) 极点角度
sigma_alpha = 1;    % AR(1) 噪声标准差
sigma_beta = 1;     % AR(2) 噪声标准差
M = 1000;             % Monte Carlo 实验次数

% 真实 AR(2) 系数
a1_true = rho_true * cos(phi_true);
a2_true = -rho_true^2 / 4;

%% 自定义函数：手动估计 AR(1) 参数
function [a1, sigma2] = estimate_ar1(x)
    N = length(x);
    if N < 2
        a1 = 0;
        sigma2 = inf;
        return;
    end
    % 计算自相关 R0 和 R1
    R0 = sum(x(1:N).^2) / N;
    R1 = sum(x(2:N) .* x(1:N-1)) / (N-1);
    a1 = R1 / R0;                   % AR(1) 系数
    residuals = x(2:N) - a1 * x(1:N-1);
    sigma2 = sum(residuals.^2) / (N-1); % 噪声方差
end

%% 自定义函数：手动估计 AR(2) 参数
function [a1, a2, sigma2] = estimate_ar2(x)
    N = length(x);
    if N < 3
        a1 = 0; a2 = 0; sigma2 = inf;
        return;
    end
    % 计算自相关 R0, R1, R2
    R0 = sum(x(1:N).^2) / N;
    R1 = sum(x(2:N) .* x(1:N-1)) / (N-1);
    R2 = sum(x(3:N) .* x(1:N-2)) / (N-2);
    % 构建 Yule-Walker 方程矩阵
    R_matrix = [R0, R1; R1, R0];
    r_vector = [R1; R2];
    % 解线性方程组
    a = R_matrix \ r_vector;
    a1 = a(1);
    a2 = a(2);
    % 计算残差和噪声方差
    residuals = x(3:N) - a1*x(2:N-1) - a2*x(1:N-2);
    sigma2 = sum(residuals.^2) / (N-2);
end

%% Monte Carlo 仿真
mse_values = zeros(M, 1);

for mc = 1:M
    % 生成数据
    x = zeros(N, 1);
    
    % 生成 AR(1) 数据 (n ≤ T0)
    x(1) = sqrt(sigma_alpha) * randn;
    for n = 2:T0
        x(n) = rho_true * x(n-1) + sqrt(sigma_alpha) * randn;
    end
    
    % 生成 AR(2) 数据 (n > T0)
    x(T0+1) = sqrt(sigma_beta) * randn;
    x(T0+2) = sqrt(sigma_beta) * randn;
    for n = T0+3:N
        x(n) = a1_true * x(n-1) + a2_true * x(n-2) + sqrt(sigma_beta) * randn;
    end

    %% 遍历所有可能的突变点 T_hat，计算 RSS
    RSS = inf(N, 1); % 初始化为无穷大
    
    for T_hat = 2:N-2
        % 前段 AR(1) 参数估计
        x_front = x(1:T_hat);
        [a1_ar1, ~] = estimate_ar1(x_front);
        
        % 前段残差计算
        if T_hat >= 2
            residuals_front = x_front(2:T_hat) - a1_ar1 * x_front(1:T_hat-1);
            RSS1 = sum(residuals_front.^2);
        else
            RSS1 = inf;
        end
        
        % 后段 AR(2) 参数估计
        x_back = x(T_hat+1:end);
        [a1_ar2, a2_ar2, ~] = estimate_ar2(x_back);
        
        % 后段残差计算
        if length(x_back) >= 3
            residuals_back = x_back(3:end) - a1_ar2*x_back(2:end-1) - a2_ar2*x_back(1:end-2);
            RSS2 = sum(residuals_back.^2);
        else
            RSS2 = inf;
        end
        
        % 总 RSS
        RSS(T_hat) = RSS1 + RSS2;
    end
    
    % 找到最小 RSS 对应的 T_hat
    [~, T_hat_opt] = min(RSS(2:N-2));
    T_hat_opt = T_hat_opt + 1; % 补偿索引偏移
    
    % 记录 MSE
    mse_values(mc) = (T_hat_opt - T0)^2;
end

%% 计算均方误差 (MSE)
MSE = mean(mse_values);
fprintf('任务2 (参数未知) 的 MSE: %.4f\n', MSE);

%% 绘制最后一次实验的信号和 RSS 曲线
figure;
subplot(2,1,1);
plot(x, 'b', 'LineWidth', 1.5); hold on;
xline(T0, '--r', 'LineWidth', 2, 'Label', '真实 T_0');
xline(T_hat_opt, '--g', 'LineWidth', 2, 'Label', '预测 T_0');
title('原始数据与检测结果');
xlabel('时间 n'); ylabel('x[n]');
legend('信号', '真实 T_0', '预测 T_0', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(2:N-2, RSS(2:N-2), 'b', 'LineWidth', 1.5); hold on;
plot(T0, RSS(T0), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(T_hat_opt, RSS(T_hat_opt), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
title('残差平方和 (RSS) vs 候选断点 T''');
xlabel('T'''); ylabel('RSS');
legend('RSS', '真实 T_0', '预测 T_0', 'Location', 'best');
grid on;
