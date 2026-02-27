%% %%%%%  SP&L-WORKSHOP-Homework  %%%%%%%%%%%%%% 22/10/2024 %%%%
%%%%%%%%%%%%%%%%%%%%  ----------01/12/2024 edit--------%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  ----------18/07/2025 edit--------%%%%%%%%%
function main()
    addpath('utils');
    %%%%%%%%----参数定义赋值--1.2 -----%%%%%%%%
    a0 = 1;
    N_values = [100, 500, 1024]; % 样本数量
    N_values_1 = linspace(20, 200, 10);
    omega_values = pi * [1/4, 1/2, 1/(2^0.5), 0.81];% [1/4, 1/2, 1/sqrt(2), 0.81]; % 频率值
    SNR_dB = -20:5:40; % 信噪比范围（单位：dB）
    rho_values = [0, 0.5, 0.9]; % 协方差相关系数
    sigma_w = 1;  % 噪声标准差
    L = 2000;  % Monte Carlo 样本数量

    % 定义滤波器 H(z) 的系数
    filter_coeffs.b = [2];
    filter_coeffs.a = [1, 0.9];
    num_simulations = 10; % 仿真次数, 用于求MSE的平均

%% %%%%%% ---- 分为4个部分分别计算 ------%%%%%%
% y1 = generate_correlated_noise_and_evaluate_mse(N_values_1, rho_values, sigma_w, L); % project 1.1
y2_1 = Fun_2_1(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations); % project 1.2.1
% y2_2 = Fun_2_2(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations); % project 1.2.2
% y2_3 = Fun_2_3(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations); % project 1.2.3
end
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
    title(sprintf('1.1 Noise generation\n MSE vs Sample size for different \rho values'));
    grid on;
    Project_1_1 = 0;
end

%% %%%%%%%%%%%%%%  ------1.2 Frequency estimation -------%%%%%%%%%%%%%%%%

function Project_1_2_1 = Fun_2_1(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations)
Project_1_2_1 = 1;   

    % 主程序：完成频率估计任务
    
    %% %%%%%%%%%----- 1.2.1 h = delta; ρ = 0 ------%%%%%%%%
    h_case = "delta";
    rho = rho_values(1)
   
    % 初始化存储结果的矩阵
    all_mse_results = zeros(num_simulations, length(N_values), length(SNR_dB));

    for omega = omega_values
        % 多次仿真求均值
        for sim = 1:num_simulations
            sim_results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0);
            all_mse_results(sim, :, :) = sim_results.MSE; % 存储每次仿真结果
        end

        % 计算均值MSE
        avg_mse = mean(all_mse_results, 1); % 对第1维度求均值

        % 保存最终结果
        results.MSE_mean = squeeze(avg_mse); % 删除多余维度
        results.CRB = sim_results.CRB; % CRB不需要重复仿真，直接保存

        % 绘制结果
        figure;
        hold on;
        % 定义颜色表
        color_map = lines(length(N_values)); % 使用 MATLAB 的内置颜色表, 目的: 让同样的N值下的MSE和CRB颜色一致

        for i = 1:length(N_values)
            % 绘制 MSE 曲线，使用指定颜色
            plot(SNR_dB, 10*log10(results.MSE_mean(i, :)), ...
                'Color', color_map(i, :), 'DisplayName', sprintf('Mean MSE (N=%d)', N_values(i)),'LineWidth', 2);

            % 绘制 CRB 曲线，使用相同颜色，并设置为虚线
            plot(SNR_dB, 10*log10(results.CRB(i, :)), '--', ...
                'Color', color_map(i, :), 'DisplayName', sprintf('CRB (N=%d)', N_values(i)),'LineWidth', 2);
        end
        Project_1_2_1 = 0; 
        hold off;
        xlabel('SNR (dB)');
        ylabel('MSE (dB)');
        title(sprintf('1.2.1 Frequency Estimation (\\omega=%.2f, \\rho=%.2f, h=%s)', omega, rho, h_case));
        legend show;
    end

end

function Project_1_2_2 = Fun_2_2(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations)
    Project_1_2_2 = 1;
    %% %%%%% ------1.2.2 h = fliter; ρ = 0 -------- %%%%%%%%%% 绘制 MSE 和 CRB 曲线
    h_case = "filter";
    rho = rho_values(1)
    for omega = omega_values
        % 多次仿真求均值
        for sim = 1:num_simulations
            sim_results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0);
            all_mse_results(sim, :, :) = sim_results.MSE; % 存储每次仿真结果
        end

        % 计算均值MSE
        avg_mse = mean(all_mse_results, 1); % 对第1维度求均值

        % 保存最终结果
        results.MSE_mean = squeeze(avg_mse); % 删除多余维度
        results.CRB = sim_results.CRB; % CRB不需要重复仿真，直接保存

        % 绘制结果
        figure;
        hold on;
        % 定义颜色表
        color_map = lines(length(N_values)); % 使用 MATLAB 的内置颜色表, 目的: 让同样的N值下的MSE和CRB颜色一致

        for i = 1:length(N_values)
            % 绘制 MSE 曲线，使用指定颜色
            plot(SNR_dB, 10*log10(results.MSE_mean(i, :)), ...
                'Color', color_map(i, :), 'DisplayName', sprintf('Mean MSE (N=%d)', N_values(i)),'LineWidth', 2);

            % 绘制 CRB 曲线，使用相同颜色，并设置为虚线
            plot(SNR_dB, 10*log10(results.CRB(i, :)), '--', ...
                'Color', color_map(i, :), 'DisplayName', sprintf('CRB (N=%d)', N_values(i)),'LineWidth', 2);
        end
        Project_1_2_2 = 0; % 目的: 严重是否正确输出
        hold off;
        xlabel('SNR (dB)');
        ylabel('MSE (dB)');
        title(sprintf('1.2.2 Frequency Estimation (\\omega=%.2f, \\rho=%.2f, h=%s)', omega, rho, h_case));
        legend show;
    end

end

function Project_1_2_3 = Fun_2_3(a0, N_values, omega_values, SNR_dB, rho_values, filter_coeffs, num_simulations)
Project_1_2_3 = 1;
    %% %%%%%%%%%----- 1.2.3.1, h = delta; ρ = 0.9 ------%%%%%%%%
    h_case = "delta";
    rho = rho_values(2)
    % 参数定义

    % 初始化存储结果的矩阵
    all_mse_results = zeros(num_simulations, length(N_values), length(SNR_dB));

    for omega = omega_values
        % 多次仿真求均值
        for sim = 1:num_simulations
            sim_results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0);
            all_mse_results(sim, :, :) = sim_results.MSE; % 存储每次仿真结果
        end

        % 计算均值MSE
        avg_mse = mean(all_mse_results, 1); % 对第1维度求均值

        % 保存最终结果
        results.MSE_mean = squeeze(avg_mse); % 删除多余维度
        results.CRB = sim_results.CRB; % CRB不需要重复仿真，直接保存

        % 绘制结果
        figure;
        hold on;
        % 定义颜色表
        color_map = lines(length(N_values)); % 使用 MATLAB 的内置颜色表, 目的: 让同样的N值下的MSE和CRB颜色一致
        
        for i = 1:length(N_values)
            % 绘制 MSE 曲线，使用指定颜色
            plot(SNR_dB, 10*log10(results.MSE_mean(i, :)), ...
                'Color', color_map(i, :), 'DisplayName', sprintf('Mean MSE (N=%d)', N_values(i)),'LineWidth', 2);

            % 绘制 CRB 曲线，使用相同颜色，并设置为虚线
            plot(SNR_dB, 10*log10(results.CRB(i, :)), '--', ...
                'Color', color_map(i, :), 'DisplayName', sprintf('CRB (N=%d)', N_values(i)),'LineWidth', 2);
        end
        hold off;
        xlabel('SNR (dB)');
        ylabel('MSE (dB)');
        title(sprintf('1.2.3.1 Frequency Estimation (\\omega=%.2f, \\rho=%.2f, h=%s)', omega, rho, h_case));
        legend show;
    end

    %% %%%%%%%%%----- 1.2.3.2 h = filter; ρ = 0.9 ------%%%%%%%%
    h_case = "filter";
    rho = rho_values(2)
    % 参数定义

    % 初始化存储结果的矩阵
    all_mse_results = zeros(num_simulations, length(N_values), length(SNR_dB));

    for omega = omega_values
        % 多次仿真求均值
        for sim = 1:num_simulations
            sim_results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0);
            all_mse_results(sim, :, :) = sim_results.MSE; % 存储每次仿真结果
        end

        % 计算均值MSE
        avg_mse = mean(all_mse_results, 1); % 对第1维度求均值

        % 保存最终结果
        results.MSE_mean = squeeze(avg_mse); % 删除多余维度
        results.CRB = sim_results.CRB; % CRB不需要重复仿真，直接保存

        % 绘制结果
        figure;
        hold on;
        % 定义颜色表
        color_map = lines(length(N_values)); % 使用 MATLAB 的内置颜色表, 目的: 让同样的N值下的MSE和CRB颜色一致

        for i = 1:length(N_values)
            % 绘制 MSE 曲线，使用指定颜色
            plot(SNR_dB, 10*log10(results.MSE_mean(i, :)), ...
                'Color', color_map(i, :), 'DisplayName', sprintf('Mean MSE (N=%d)', N_values(i)),'LineWidth', 2);

            % 绘制 CRB 曲线，使用相同颜色，并设置为虚线
            plot(SNR_dB, 10*log10(results.CRB(i, :)), '--', ...
                'Color', color_map(i, :), 'DisplayName', sprintf('CRB (N=%d)', N_values(i)),'LineWidth', 2);
        end
        Project_1_2_3 = 0; 
        hold off;
        xlabel('SNR (dB)');
        ylabel('MSE (dB)');
        title(sprintf('1.2.3.2 Frequency Estimation (\\omega=%.2f, \\rho=%.2f, h=%s)', omega, rho, h_case));
        legend show;
    end     

end

%% %%%%%% ------  1.全局函数-自定义 ------ %%%%%%%%%1.2
    function results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0)
        % 仿真频率估计的 MSE
        results = struct();
        results.SNR_dB = SNR_dB;
        results.MSE = zeros(length(N_values), length(SNR_dB));
        results.CRB = zeros(length(N_values), length(SNR_dB));
    
        for i = 1:length(N_values)
            N = N_values(i);
            for j = 1:length(SNR_dB)
                snr_linear = 10^(SNR_dB(j) / 10);
                sigma_w2 = a0 / (2 * snr_linear);
    
                % 生成观测信号
                x = generate_signal(N, omega, sigma_w2, rho);
    
                % 滤波处理（根据 h_case）
                if h_case == "filter"
                    x = filter_signal(x, filter_coeffs);
                end
    
                % 估计频率
                omega_estimates = estimate_frequency_fft(x, N);
    
                % 计算 MSE 和 CRB
                results.MSE(i, j) = mean((omega_estimates - omega).^2);
                results.CRB(i, j) = compute_crb(N, snr_linear);
            end
        end
    end

    function crb = compute_crb(N, snr_linear)
        % 计算 Cramér-Rao Bound
        % crb = 9/(4 * pi^2 *snr_linear * N*(N-1)*(2*N-1)); % f0
        % crb = 9/( snr_linear * N*(N-1)*(2*N-1)); % f0
        crb = 6 / (snr_linear * N*(N^2-1));%*(2*N-1)); % 近似之后的 % ω0
    end
    
    function x = generate_signal(N, omega, sigma_w2, rho)
        % 生成带噪信号
        A = 1; % 信号幅度
        phi_0 = 2 * pi * rand(); % 随机初相位
        n = (0:N-1)';
        s = A * cos(omega * n + phi_0); % 正弦信号
    
        % 生成噪声
        w = generate_correlated_noise_sigma(N, sigma_w2, rho);
        x = s + w; % 叠加噪声后的信号
    end
    
    function w = generate_correlated_noise_sigma(N, sigma_w2, rho)
        % 生成相关噪声
        Cww = sigma_w2 * rho.^(abs((1:N)' - (1:N))); % 协方差矩阵
        A = sqrt_covariance_matrix(Cww);              
        g = randn(N, 1); % 独立高斯噪声
        w = A * g; % 生成相关噪声
    end

    function x_filtered = filter_signal(x, filter_coeffs)
        % 对信号进行滤波
        x_filtered = filter(filter_coeffs.b, filter_coeffs.a, x);
    end

    function omega_estimates = estimate_frequency_fft(x, N)
    % 基于 FFT 和最大似然方法估计频率
    % 输入参数：
    %   x - 输入信号
    %   N - 信号长度
    %   N_fft - 增加后的FFT点数（零填充的长度）
    % 输出参数：
    %   omega_estimates - 精确估计的频率
    N_fft = 4 * N;

    % 1. 使用 FFT 初步估计频率
    X = fft(x, N_fft);  % 对信号 x 做 FFT，N_fft > N
    magnitude_spectrum = abs(X(1:floor(N_fft/2)));  % 取正频部分的幅值
    [~, max_idx] = max(magnitude_spectrum);  % 找到最大值索引
    f_fft = (max_idx - 1) / N_fft;  % 对应频率（Hz）
    omega_fft = 2 * pi * f_fft ;  % 对应角频率（rad/s）

    % 使用最大似然法在附近精确搜索频率
    search_range = linspace(omega_fft - pi/N_fft, omega_fft + pi/N_fft, 500);  % 搜索范围
    likelihoods = zeros(length(search_range), 1);

    for k = 1:length(search_range)
        omega = search_range(k);
        likelihoods(k) = sum(x .* cos((0:N-1)' * omega));
    end

    [~, max_idx_refined] = max(likelihoods);
    omega_estimates = search_range(max_idx_refined);  % 精确估计的频率
end


    % function omega_estimates = estimate_frequency_fft(x, N)
    %     % 基于 FFT 和最大似然方法估计频率
    %     % 输入参数：
    %     %   x - 输入信号
    %     %   N - 信号长度
    %     %   fs - 采样频率
    %     % 输出参数：
    %     %   omega_estimates - 精确估计的频率
    % 
    %     % 1. 使用 FFT 初步估计频率
    %     X = fft(x, N);  % 对信号 x 做 FFT
    %     magnitude_spectrum = abs(X(1:floor(N/2)));  % 取正频部分的幅值
    %     [~, max_idx] = max(magnitude_spectrum);  % 找到最大值索引
    %     f_fft = (max_idx - 1) / N;  % 对应频率（Hz）
    %     omega_fft = 2 * pi * f_fft ;  % 对应角频率（rad/s）
    % 
    %     %使用最大似然法在附近精确搜索频率
    %     search_range = linspace(omega_fft - pi/N, omega_fft + pi/N, 500);  % 搜索范围
    %     likelihoods = zeros(length(search_range), 1);
    % 
    %     for k = 1:length(search_range)
    %         omega = search_range(k);
    %         likelihoods(k) = sum(x .* cos((0:N-1)' * omega));
    %     end
    % 
    %     [~, max_idx_refined] = max(likelihoods);
    %     omega_estimates = search_range(max_idx_refined);  % 精确估计的频率
    % end



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



%% %%%%%%%%%--------下述函数目前未使用------------%%%%%%%%%%%%%%%%
% function [lambda, V] = algebraicEigenvalue(Cww)
%     % 求解代数特征方程 |λI - Cww| = 0
%     % Cww：输入的协方差矩阵
% 
%     % 步骤1：计算特征值
%     syms lambda;
%     I = eye(size(Cww)); % 单位矩阵
%     eqn = det(lambda*I - Cww) == 0; % 特征方程
%     lambda_sol = solve(eqn, lambda); % 解方程，得到特征值
% 
%     % 步骤2：计算对应的特征向量
%     lambda = double(lambda_sol); % 转换为数值类型
%     V = zeros(length(lambda), length(lambda)); % 存储特征向量
% 
%     for i = 1:length(lambda)
%         % 对于每个特征值，解 (Cww - λI)v = 0
%         A = Cww - lambda(i)*I;
%         [~, S, V_temp] = svd(A);
%         V(:, i) = V_temp(:, end); % 最后一个列向量是特征向量
%     end
% end


%% %%%%%%% ------- 1.3Frequency modulation ------- %%%%%%%%%%%
% 参数定义
fs = 22000;             % 采样频率 (Hz)
N = 22000;              % 采样点数
n = (0:N-1)';           % 时间索引, 将其转置成列向量
a = 1;                  % 信号幅度
sigma_w = 0.1;          % 噪声标准差
phi = 0;                % 初始相位
window_size = 512;      % 窗口大小
overlap = window_size/2;        % 窗口重叠
nfft = 4*window_size;     % nfft 是傅里叶变换点数
% 瞬时频率范围
gamma = (3*pi/4 - pi/8) / (2 * (N-1)); % 由题目中频率范围确定
omega_n = 2 * gamma * n;              % 瞬时频率 w = 2γn
x = a * cos(gamma * n.^2 + phi) + sigma_w * randn(size(n)); % 调制信号

% 计算分段的起点索引
step_size = window_size - overlap; % 每步的非重叠点数
num_segments = floor((N - overlap) / step_size); % 分段数
segments = zeros(window_size, num_segments);

% 创建加权的 Hann 窗口函数
beta = 0.5; % 调节平滑系数（0 接近 Hann 窗，1 接近矩形窗）
m = 0:(window_size-1);
modified_hann = beta + (1 - beta) * (0.5 - 0.5 * cos(2 * pi * m / (window_size - 1)));
window_function = modified_hann';
% window_function = hann(window_size);

% 对每段信号进行分窗和 FFT
stft_result = zeros(nfft, num_segments); % 初始化 STFT 结果
time_vector = zeros(1, num_segments);    % 存储时间索引
for i = 1:num_segments
    start_idx = (i-1) * step_size + 1; % 当前段起点
    end_idx = start_idx + window_size - 1; % 当前段终点
    
    % 截取信号并加窗
    segment = x(start_idx:end_idx) .* window_function;
    segments(:, i) = segment;
    
    % 对信号段进行 FFT
    stft_result(:, i) = fft(segment, nfft);
    
    % 保存时间点
    time_vector(i) = (start_idx + end_idx) / 2 / fs;
end

% 计算频率轴
frequency_vector = (0:nfft-1) * fs / nfft;

% 取频谱的幅值并转换为 dB
spectrogram_magnitude = 20 * log10(abs(stft_result));

% 绘制时频图
% surf(time_vector, frequency_vector, spectrogram_magnitude);
figure;
imagesc(time_vector, frequency_vector(1:nfft), spectrogram_magnitude(1:nfft, :));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;
