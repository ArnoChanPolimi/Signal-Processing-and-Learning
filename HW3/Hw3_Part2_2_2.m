clc; clear; close all;

%% 参数设置
N = 400;            % 总数据点数
T0 = 200;           % 真实突变点
rho_true = 0.8;     % AR(1) 真实系数
phi_true = pi/4;    % AR(2) 极点角度
sigma_alpha = 1;    % AR(1) 噪声标准差
sigma_beta = 1;     % AR(2) 噪声标准差
M = 50;             % Monte Carlo 实验次数

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
    % 计算自相关 R0 和 R1（无偏估计）
    R0 = sum(x(1:N).^2) / N;
    R1 = sum(x(2:N) .* x(1:N-1)) / (N-1);
    a1 = R1 / R0;                   % Yule-Walker 方程解
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
    % 计算自相关 R0, R1, R2（无偏估计）
    R0 = sum(x(1:N).^2) / N;
    R1 = sum(x(2:N) .* x(1:N-1)) / (N-1);
    R2 = sum(x(3:N) .* x(1:N-2)) / (N-2);
    % 构建 Yule-Walker 方程矩阵
    R_matrix = [R0, R1; R1, R0];
    r_vector = [R1; R2];
    a = R_matrix \ r_vector;        % 解线性方程组
    a1 = a(1);
    a2 = a(2);
    residuals = x(3:N) - a1*x(2:N-1) - a2*x(1:N-2);
    sigma2 = sum(residuals.^2) / (N-2);
end

%% Monte Carlo 仿真
% 预分配存储变量
mse_T0 = zeros(M, 1);
rho_ar1_ests = zeros(M, 1);     % AR(1) 的 ρ 估计值
a1_ar2_ests = zeros(M, 1);      % AR(2) 的 a1 估计值
a2_ar2_ests = zeros(M, 1);      % AR(2) 的 a2 估计值
sigma_alpha_ests = zeros(M,1);  % AR(1) 噪声方差估计值
sigma_beta_ests = zeros(M,1);   % AR(2) 噪声方差估计值
rho_ar2_ests = zeros(M,1);      % 从 AR(2) 反推的 ρ
phi_ests = zeros(M,1);          % 从 AR(2) 反推的 φ

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
    RSS = inf(N, 1);
    
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
        
        RSS(T_hat) = RSS1 + RSS2;
    end
    
    % 找到最小 RSS 对应的 T_hat
    [~, T_hat_opt] = min(RSS(2:N-2));
    T_hat_opt = T_hat_opt + 1; % 补偿索引偏移
    mse_T0(mc) = (T_hat_opt - T0)^2;
    
    %% 保存参数估计值
    % 前段 AR(1) 的参数
    x_front = x(1:T_hat_opt);
    [rho_ar1_ests(mc), sigma_alpha_ests(mc)] = estimate_ar1(x_front);
    
    % 后段 AR(2) 的参数
    x_back = x(T_hat_opt+1:end);
    [a1_ar2_ests(mc), a2_ar2_ests(mc), sigma_beta_ests(mc)] = estimate_ar2(x_back);
    
    % 从 AR(2) 系数反推 ρ 和 φ
    if a2_ar2_ests(mc) < 0
        rho_ar2 = 2*sqrt(-a2_ar2_ests(mc));
        rho_ar2_ests(mc) = rho_ar2;
        if abs(a1_ar2_ests(mc)/rho_ar2) <= 1
            phi_ests(mc) = acos(a1_ar2_ests(mc)/rho_ar2);
        else
            phi_ests(mc) = NaN; % 无效值
        end
    else
        rho_ar2_ests(mc) = NaN;
        phi_ests(mc) = NaN;
    end
end

%% 过滤无效值
valid_idx = ~isnan(rho_ar2_ests) & ~isnan(phi_ests);
rho_ar1_valid = rho_ar1_ests(valid_idx);
a1_ar2_valid = a1_ar2_ests(valid_idx);
a2_ar2_valid = a2_ar2_ests(valid_idx);
rho_ar2_valid = rho_ar2_ests(valid_idx);
phi_valid = phi_ests(valid_idx);
sigma_alpha_valid = sigma_alpha_ests(valid_idx);
sigma_beta_valid = sigma_beta_ests(valid_idx);

%% 输出统计结果
fprintf('=== AR(1) 参数估计结果 ===\n');
fprintf('ρ 均值 = %.4f ± %.4f (真实值: %.4f)\n', mean(rho_ar1_valid), std(rho_ar1_valid), rho_true);
fprintf('噪声方差 σ_α² 均值 = %.4f ± %.4f (真实值: %.4f)\n', mean(sigma_alpha_valid), std(sigma_alpha_valid), sigma_alpha^2);

fprintf('\n=== AR(2) 参数估计结果 ===\n');
fprintf('a1 均值 = %.4f ± %.4f (真实值: %.4f)\n', mean(a1_ar2_valid), std(a1_ar2_valid), a1_true);
fprintf('a2 均值 = %.4f ± %.4f (真实值: %.4f)\n', mean(a2_ar2_valid), std(a2_ar2_valid), a2_true);
fprintf('噪声方差 σ_β² 均值 = %.4f ± %.4f (真实值: %.4f)\n', mean(sigma_beta_valid), std(sigma_beta_valid), sigma_beta^2);
fprintf('从 AR(2) 反推的 ρ 均值 = %.4f ± %.4f (理论应与 AR(1) 的 ρ 一致)\n', mean(rho_ar2_valid), std(rho_ar2_valid));
fprintf('从 AR(2) 反推的 φ 均值 = %.4f ± %.4f rad (真实值: %.4f rad)\n', mean(phi_valid), std(phi_valid), phi_true);

%% 可视化参数分布
figure;

% AR(1) 的 ρ 估计分布
subplot(2,3,1);
histogram(rho_ar1_valid, 'Normalization', 'pdf', 'BinWidth', 0.02);
xline(rho_true, 'r--', 'LineWidth', 2);
title('AR(1) 的 ρ 估计分布');
xlabel('ρ'); ylabel('密度');

% AR(2) 的 a1 估计分布
subplot(2,3,2);
histogram(a1_ar2_valid, 'Normalization', 'pdf', 'BinWidth', 0.02);
xline(a1_true, 'r--', 'LineWidth', 2);
title('AR(2) 的 a1 估计分布');
xlabel('a1'); ylabel('密度');

% AR(2) 的 a2 估计分布
subplot(2,3,3);
histogram(a2_ar2_valid, 'Normalization', 'pdf', 'BinWidth', 0.02);
xline(a2_true, 'r--', 'LineWidth', 2);
title('AR(2) 的 a2 估计分布');
xlabel('a2'); ylabel('密度');

% 从 AR(2) 反推的 ρ 分布
subplot(2,3,4);
histogram(rho_ar2_valid, 'Normalization', 'pdf', 'BinWidth', 0.02);
xline(rho_true, 'r--', 'LineWidth', 2);
title('从 AR(2) 反推的 ρ 分布');
xlabel('ρ'); ylabel('密度');

% 从 AR(2) 反推的 φ 分布
subplot(2,3,5);
histogram(phi_valid, 'Normalization', 'pdf', 'BinWidth', 0.05);
xline(phi_true, 'r--', 'LineWidth', 2);
title('从 AR(2) 反推的 φ 分布');
xlabel('φ (rad)'); ylabel('密度');

% AR(1) 和 AR(2) 的 ρ 对比
subplot(2,3,6);
scatter(rho_ar1_valid, rho_ar2_valid, 'filled');
xlabel('AR(1) 估计的 ρ');
ylabel('AR(2) 反推的 ρ');
title('两段 ρ 估计值对比');
hold on;
plot([0,1], [0,1], 'r--'); % 理想情况下的对角线
axis equal;