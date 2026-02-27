%%%%%%%%----参数定义赋值--1.2 -----%%%%%%%%
    a0 = 1;
    N_values = [100, 500, 1024]; % 样本数量
    N_values_1 = linspace(20, 200, 10);
    omega_values = pi * [1/4, 1/2, 1/(2^0.5), 0.81];% [1/4, 1/2, 1/sqrt(2), 0.81]; % 频率值
    SNR_dB = -20:5:40; % 信噪比范围（单位：dB）
    rho_values = [0, 0.5, 0.99]; % 协方差相关系数
    sigma_w = 1;  % 噪声标准差
    L = 2000;  % Monte Carlo 样本数量
y1 = generate_correlated_noise_and_evaluate_mse(N_values_1, rho_values, sigma_w, L); % project 1.1

%% %%%%%%%%%%%%%%%%%%  ----------1.1 Noise generation--------%%%%%%%
function Project_1_1 = generate_correlated_noise_and_evaluate_mse(N_values, rho_values, sigma_w, L)
    Project_1_1 = 1;
    %%%%%%%%%%%%%%%%%%%%  ----------参数定义--------%%%%%%%
    % N_values = linspace(20, 200, 10);
    % rho_values = [0, 0.5, 0.99];
    % sigma_w = 1;  % 噪声标准差
    % L = 1000;  % Monte Carlo 样本数量
    
    % 初始化结果
    MSE_results = zeros(length(N_values), length(rho_values));
    
    %%%%%%%%%%%%-------主循环--------%%%%%%%
    for i = 1:length(N_values)
        N = N_values(i);
        for j = 1:length(rho_values)
            rho = rho_values(j);
    
            % 生成协方差矩阵
            Cww = generate_covariance_matrix(N, sigma_w, rho);
    
            % 生成相关噪声
            w = generate_correlated_noise_Cww(Cww, L);
    
            % 估计样本协方差矩阵
            Chat = estimate_sample_covariance(w);
    
            % 计算 MSE
            MSE_results(i, j) = 10 * log10(compute_mse(Chat, Cww));
        end
    end
    
    %%%%%%%% ---- 绘图----- %%%%%%%
    figure(1);
    for j = 1:length(rho_values)
        plot(N_values, MSE_results(:, j), 'DisplayName', ['\rho = ' num2str(rho_values(j))],'LineWidth', 2);
        hold on;
    end
    hold off;
    xlabel('Sample size (N)');
    ylabel('MSE / dB');
    legend;
    title(sprintf('1.1 Noise generation\n MSE vs Sample size for different rho values'));
    grid on;
    Project_1_1 = 0;
end
    %% %%%%%%%%%%%%%%%  ----- 2.全局函数-自定义 -----%%%%%%%%%%%%%%%%%%1.1
function Cww = generate_covariance_matrix(N, sigma_w, rho)
% 生成 N x N 的协方差矩阵
    Cww = zeros(N, N);
    for i = 1:N
        for j = 1:N
            Cww(i, j) = sigma_w^2 * rho^abs(i - j);
        end
    end
end

function w = generate_correlated_noise_Cww(Cww, L)
% 使用协方差矩阵 Cww 和样本数 L 生成相关噪声
N = size(Cww, 1);
z = randn(N, L);  % 生成标准正态分布噪声
% A = myCholesky(Cww);  % Cholesky 分解, 此项目禁止使用
A = sqrt_covariance_matrix(Cww);
w = A * z;  % 生成相关噪声
end

function A = sqrt_covariance_matrix(Cww)
% 进行特征值分解
[Q, Lambda] = fun_eig(Cww); % Q是特征向量矩阵，Lambda是特征值矩阵
A = Q * sqrt(Lambda); % Cww=A * A^T
end

function Chat = estimate_sample_covariance(w)
% w 是 N x L 的噪声矩阵，返回样本协方差矩阵
[N, L] = size(w);
w_mean = mean(w);              % 按列计算样本均值
w_centered = w - w_mean;          % 每列减去均值
Chat = (w_centered * w_centered') / L;  % 样本协方差矩阵
end

function mse = compute_mse(Chat, Cww)
% 计算样本协方差矩阵 Chat 和理论协方差矩阵 Cww 之间的 MSE
mse = mean((Chat(:) - Cww(:)).^2);
end

function [Q, Lambda] = fun_eig(Cww)
% 求Cww的特征向量矩阵和特征值矩阵
[Q, Lambda] = eig(Cww); % Q是特征向量矩阵，Lambda是特征值矩阵
end