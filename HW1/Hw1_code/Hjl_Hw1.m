% clc;
% clear all;
% %% ========= Project 1.1 ============
% % 生成不同ρ和N下的相关高斯噪声，并比较样本协方差与真实协方差的MSE
% 
% % 参数定义
% rho_list = [0, 0.5, 0.99];             % 相关性参数ρ
% N_list = round(linspace(20, 200, 10)); % 不同样本长度N
% L = 100;                              % Monte Carlo 样本数
% sigma_w = 1;                           % 噪声标准差
% 
% 
% N_value = [100, 500, 1024];
% SNR_dB_list= -30: 5: 40;
% %a0=1;
% omega0_list = pi*[1/4, 1/2, 1/sqrt(2), 0.81];
% phi0 = 0;
% 
% filter_coeffs.b = [2];%滤波器系数
% filter_coeffs.a = [1, 0.9];
% 
% 
% %% Homework1 : 1.1 
% %SPL_Hw1_1_1_MSEvsN_with_diff_rho(rho_list, N_list, L, sigma_w);
% % SPL_Hw1_1_2( sigma_w, phi0, omega0_list, N_value, SNR_dB_list, rho_list, filter_coeffs, L);
% SPL_Hw1_1_2_3( sigma_w, omega0_list, N_value, SNR_dB_list, filter_coeffs, L );
% 
% 
% 
% 
% 
% %% 1.2
% function SPL_Hw1_1_2( sigma_w, phi0, omega0_list, N_value, SNR_dB_list, rho_list, filter_coeffs, L)
% 
% rho = rho_list(1);
% % filter_case       = "delta";               % "delta" 或 "filter_H"
% filter_case       = "filter_H";
% % 对每个 ω₀ 生成一张图
% for idxOmega = 1:length(omega0_list)
%     omega0 = omega0_list(idxOmega);       % 固定当前 ω₀
% 
%     % 预分配
%     MSE_result_dB = zeros(length(N_value), length(SNR_dB_list));
%     CRB_dB        = zeros(length(N_value), length(SNR_dB_list));
%     omega_hat     = zeros(1, L);
% 
%     % 新建 figure
%     figure; hold on; grid on;
%     colors = lines(length(N_value));
% 
%     % 对每个 N 计算并绘制 MSE/CRB
%     for idxN = 1:length(N_value)
%         N = N_value(idxN);
% 
%         % 1) 计算 CRB
%         for j = 1:length(SNR_dB_list)
%             snr_db     = SNR_dB_list(j);
%             snr_linear = 10^(snr_db/10);
%             CRB_dB(idxN, j) = 10*log10( generate_CRB(snr_linear, N) );
%         end
% 
%         % 2) Monte Carlo 估计 MSE
%         for j = 1:length(SNR_dB_list)
%             snr_db     = SNR_dB_list(j);
%             snr_linear = 10^(snr_db/10);
%             a0         = sqrt(2 * sigma_w^2 * snr_linear);
% 
%             for l = 1:L
%                 phi0 = 2*pi*rand;            % 或随机：2*pi*rand
%                 n    = 0:N-1;
%                 x    = a0 * cos(omega0 * n + phi0);
%                 w    = generate_w(N, sigma_w, rho);
%                 y    = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
%                 omega_hat(l) = estimate_frequency_fft_interp(y, N);
%             end
%             MSE_result_dB(idxN, j) = generate_mse_dB(omega_hat, omega0, L);
%         end
% 
%         % 3) 绘图：实线→MSE，虚线→CRB
%         plot(SNR_dB_list, MSE_result_dB(idxN, :), '-',  'LineWidth', 2, ...
%              'Color', colors(idxN,:), ...
%              'DisplayName', sprintf('N=%d, MSE', N));
%         plot(SNR_dB_list, CRB_dB(idxN, :),       '--', 'LineWidth', 2, ...
%              'Color', colors(idxN,:), ...
%              'DisplayName', sprintf('N=%d, CRB', N));
%     end
% 
%     % 坐标轴与标题
%     xlabel('SNR (dB)'); ylabel('MSE / CRB (dB)');
%     title(sprintf('\\omega_0 = %.2f\\pi, \\rho = %.2f: MSE vs SNR for Different N', ...
%       omega0/pi, rho));
%     legend('Location','best');
%     hold off;
% end
% 
% end
% 
% 
% 
% 
% %% 1.2.3 rh0=0.9， filter=h
% 
% function SPL_Hw1_1_2_3( sigma_w, omega0_list, N_value, SNR_dB_list, filter_coeffs, L)
% %rho = rho_list(1);
% rho = 0.9;
%  filter_case       = "delta";               % "delta" 或 "filter_H"
% % filter_case       = "filter_H";
% % 对每个 ω₀ 生成一张图
% for idxOmega = 1:length(omega0_list)
%     omega0 = omega0_list(idxOmega);
% 
%     % 预分配
%     MSE_dB = zeros(length(N_value), length(SNR_dB_list));
%     CRB_dB = zeros(length(N_value), length(SNR_dB_list));
%     omega_hat = zeros(1, L);
% 
%     figure; hold on; grid on;
%     colors = lines(length(N_value));
% 
%     for idxN = 1:length(N_value)
%         N = N_value(idxN);
%         %rho = 0.9;
% 
%         % % ------ 1) 预先构造 i_minus1, Cww, Toeplitz 加速 ------
%         % i_minus1 = (0:N-1)';                      % [0;1;…;N-1]
%         % r = sigma_w^2 * rho.^(0:N-1);            % 自相关序列
%         % Cww = toeplitz(r);                       % N×N 协方差矩阵
% 
%         % ------ 2) 每个 SNR 下计算 CRB ------
%         for j = 1:length(SNR_dB_list)
%             snr_db     = SNR_dB_list(j);
%             snr_linear = 10^(snr_db/10);
% 
%             % 信号幅度 a0 满足 a0^2/(2σ_w^2) = snr
%             a0 = sqrt(2 * sigma_w^2 * snr_linear);
% 
% 
%             CRB_dB(idxN,j) = compute_crb_correlated_noise(a0, sigma_w, rho, N);
%         end
% 
%         % ------ 3) Monte Carlo 估计 MSE ------
%         for j = 1:length(SNR_dB_list)
%             snr_db     = SNR_dB_list(j);
%             snr_linear = 10^(snr_db/10);
%             a0         = sqrt(2 * sigma_w^2 * snr_linear);
% 
%             for l = 1:L
%                 phi0 = 2*pi*rand;
%                 n    = 0:N-1;
%                 % x    = a0*cos(omega0*n + phi0);
%                 w    = generate_w(N, sigma_w, rho);
%                 y    = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
%                 omega_hat(l) = estimate_frequency_fft_interp(y, N);
%             end
%             MSE_dB(idxN,j) = generate_mse_dB(omega_hat, omega0, L);
%         end
% 
%         % ------ 4) 绘图：先画 CRB，再画 MSE，使实线可见 ------
%         plot(SNR_dB_list, CRB_dB(idxN,:), '--', 'LineWidth', 2, ...
%              'Color', colors(idxN,:), ...
%              'DisplayName', sprintf('N=%d, CRB', N));
%         hM = plot(SNR_dB_list, MSE_dB(idxN,:), '-', 'LineWidth', 2, ...
%              'Color', colors(idxN,:), ...
%              'DisplayName', sprintf('N=%d, MSE', N));
%         uistack(hM, 'top');  % 将 MSE 实线置于最上层
%     end
% 
%     xlabel('SNR (dB)');
%     ylabel('MSE / CRB (dB)');
%     title(sprintf('\\omega_0 = %.2f\\pi, \\rho = %.2f', omega0/pi, rho));
%     legend('Location','best');
%     hold off;
% end
% end
% 
% %% SPL homework 1: 1.1 MSE vs. N in different rho
% function SPL_Hw1_1_1_MSEvsN_with_diff_rho(rho_list, N_list, L, sigma_w)
% 
% MSE_result = zeros(length(rho_list), length(N_list));
% 
% % 主循环
% for i = 1:length(rho_list)
%     rho = rho_list(i);
%     for j = 1:length(N_list)
%         N = N_list(j);
% 
%         % 1. 构造协方差矩阵
%         Cww = generate_covariance_matrix(N, sigma_w, rho);
% 
%         % 2. 生成相关噪声样本
%         W = generate_noise(Cww, L);
% 
%         % 3. 计算样本协方差（带减均值）
%         Chat = generate_sample_covariance(W);
% 
%         % 4. 计算 MSE 并转换为 dB
%         MSE_result(i, j) = 10 * log10(compute_mse(Chat, Cww));
%     end
% end
% 
% % 绘图
% figure;
% hold on;
% for i = 1:length(rho_list)
%     plot(N_list, MSE_result(i, :), '-o', 'LineWidth', 2, ...
%         'DisplayName', ['\rho = ', num2str(rho_list(i))]);
% end
% xlabel('Sample size N');
% ylabel('MSE (dB)');
% title('Project 1.1: MSE vs N for different \rho');
% legend('Location', 'northeast');
% grid on;
% hold off;
% 
% end
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%  自定义函数  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% 1.1
% %% === 生成理论协方差矩阵 Cww ===
% function Cww = generate_covariance_matrix(N, sigma_w, rho)
%     Cww = zeros(N, N);
%     for i = 1:N
%         for j = 1:N
%             Cww(i, j) = sigma_w^2 * rho^abs(i - j);
%         end
%     end
% end
% 
% %% === 生成噪声样本 W，维度 N×L ===
% function W = generate_noise(Cww, L)
%     N = size(Cww, 1);
%     [Q, Lambda] = eig(Cww);
% 
%     Lambda_sqrt = sqrt(Lambda);
% 
%     % 构造生成矩阵 A
%     A = Q * Lambda_sqrt;
% 
%     % 白噪声
%     Z = randn(N, L);
% 
%     % 生成相关噪声
%     W = A * Z;
% end
% 
% 
% %% === 样本协方差矩阵 Chat ===
% function Chat = generate_sample_covariance(W)
%     [~, L] = size(W);
%     mean_vector = mean(W);             % 按列求均值
%     W_centered = W - mean_vector;         % 每列减均值
%     Chat = (W_centered * W_centered')/L;
% end
% 
% 
% %% === 计算 MSE (dB) ===
% function mse = compute_mse(Chat, Cww)
%     mse = mean((Chat(:) - Cww(:)).^2);
% end
% 
% 
% %% 1.2
% %% 计算 CRB
% function CRB = generate_CRB(SNR_linear, N)
% CRB = 6/(SNR_linear*(N*(N-1)*(2*N - 1)));
% end
% 
% %% 计算CRB - rho \neq 0
% function CRB_dB = compute_crb_correlated_noise(a0, sigma_w, rho, N)
%     % 构造索引向量 i = [0; 1; ...; N-1]
%     i = (0:N-1)';
% 
%     % 构造协方差矩阵 Cww
%     r = sigma_w^2 * rho.^(0:N-1);  % 自相关序列
%     Cww = toeplitz(r);            % Toeplitz 结构
% 
%     % 计算 Fisher 信息
%     v = Cww \ i;                  % 等价于 inv(Cww)*i
%     J = (a0^2 / 2) * (i' * v);    % Fisher 信息
% 
%     CRB_linear = 1 / J;           % CRB
%     CRB_dB = 10 * log10(CRB_linear);
% end
% 
% 
% %% 生成信号
% function y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0)
% 
% n = 0:N-1;
% x = a0*cos (omega0*n + phi0);
% 
% if strcmp(filter_case, "filter_H")
%     x = filter(filter_coeffs.b, filter_coeffs.a, x);
% elseif strcmp(filter_case, "delta")
%     % do nothing
% else
%     error("Unknown filter_case");
% end
% 
% x = x (:);
% y = x + w;
% end
% 
% 
% %% 生成噪声
% function w = generate_w(N, sigma_w, rho)
%     % 输出: N×1 高斯噪声，协方差满足 [Cww]_{i,j} = sigma_w^2 * rho^|i-j|
% 
%     % 构造协方差矩阵
%     Cww = zeros(N, N);
%     for i = 1:N
%         for j = 1:N
%             Cww(i,j) = sigma_w^2 * rho^abs(i-j);
%         end
%     end
% 
%     % 特征值分解生成协方差平方根矩阵
%     [Q, Lambda] = eig(Cww);
%     Lambda(Lambda < 0) = 0;  % 数值稳定
%     A = Q * sqrt(Lambda);
% 
%     % 生成 w ∈ ℝ^{N×1}
%     w = A * randn(N, 1);
% end
% 
% 
% 
% 
% 
% 
% 
% %% quadratic interpolation
% function omega_hat = estimate_frequency_fft_interp(y, N)
%     N_fft = 4*N;
%     Y = abs(fft(y, N_fft));
% 
%     [~, k_vec] = max(Y);
%     k = k_vec(1);  % 只取第一个最大值的位置，防止 k 是向量
% 
%     if k == 1 || k == N_fft
%         omega_hat = 2*pi*(k-1)/N_fft;  % 边界情况不插值
%         return;
%     end
% 
%     alpha = Y(k-1);
%     beta  = Y(k);
%     gamma = Y(k+1);
% 
%     delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma);
%     k_hat = k + delta;
% 
%     omega_hat = 2*pi*(k_hat - 1)/N_fft;
% end
% 
% %% mse
% function mse_dB = generate_mse_dB(omega_hat, omega0, L)
%         for l = 1 : L
%             error_list(l) = (omega_hat(l) - omega0).^2;
%         end
%         mse = mean(error_list);
%         mse_dB = 10 * log10(mse);
% end
% 
% 

clc; clear all; close all;

%% 参数设置
rho_list = [0, 0.5, 0.9];
N_list = [100, 500, 1024];
SNR_dB_list = -30:5:40;
omega0_list = pi * [1/4, 1/2, 1/sqrt(2), 0.81];
L = 100;
sigma_w = 1;
filter_coeffs.b = [1];        % 设置为1: 不加滤波器
filter_coeffs.a = [1];
filter_case = "delta";        % 信号不再加滤波器！

%% 主程序入口
SPL_Hw1_1_1_MSEvsN_with_diff_rho(rho_list, round(linspace(20, 200, 10)), 100, sigma_w);
SPL_Hw1_1_2(sigma_w, 0, omega0_list, N_list, SNR_dB_list, rho_list, filter_coeffs, L, filter_case);
SPL_Hw1_1_2_3(sigma_w, omega0_list, N_list, SNR_dB_list, filter_coeffs, L, filter_case);

%% ==== 1.1 MSE vs N under different \rho ====
function SPL_Hw1_1_1_MSEvsN_with_diff_rho(rho_list, N_list, L, sigma_w)
    MSE_result = zeros(length(rho_list), length(N_list));
    for i = 1:length(rho_list)
        rho = rho_list(i);
        for j = 1:length(N_list)
            N = N_list(j);
            Cww = generate_covariance_matrix(N, sigma_w, rho);
            W = generate_noise(Cww, L);
            Chat = generate_sample_covariance(W);
            MSE_result(i, j) = 10 * log10(compute_mse(Chat, Cww));
        end
    end
    figure; hold on;
    for i = 1:length(rho_list)
        plot(N_list, MSE_result(i, :), '-o', 'LineWidth', 2);
    end
    xlabel('Sample size N'); ylabel('MSE (dB)');
    title('1.1: Sample Covariance MSE vs N');
    legend(arrayfun(@(r) sprintf('\\rho=%.2f', r), rho_list, 'UniformOutput', false));
    grid on; hold off;
end

%% ==== 1.2: MSE vs CRB, rho = 0 ====
function SPL_Hw1_1_2(sigma_w, phi0, omega0_list, N_list, SNR_dB_list, rho_list, filter_coeffs, L, filter_case)
    rho = rho_list(1);
    for idxOmega = 1:length(omega0_list)
        omega0 = omega0_list(idxOmega);
        figure; hold on; grid on;
        colors = lines(length(N_list));
        for idxN = 1:length(N_list)
            N = N_list(idxN);
            MSE_result_dB = zeros(1, length(SNR_dB_list));
            CRB_dB = zeros(1, length(SNR_dB_list));
            Cww = generate_covariance_matrix(N, sigma_w, rho);
            W_all = generate_noise(Cww, L);
            for j = 1:length(SNR_dB_list)
                snr_linear = 10^(SNR_dB_list(j)/10);
                a0 = sqrt(2 * sigma_w^2 * snr_linear);
                omega_hat = zeros(1, L);
                parfor l = 1:L
                    phi0 = 2*pi*rand;
                    w = W_all(:, l);
                    y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
                    omega_hat(l) = estimate_frequency_fft_interp(y, N);
                end
                mse = mean((omega_hat - omega0).^2);
                MSE_result_dB(j) = 10*log10(mse);

                % ✅ 只有非滤波器+白噪声才使用 CRB
                if strcmp(filter_case, "delta") && rho == 0
                    CRB = 6/(snr_linear*N*(N-1)*(2*N-1));
                    CRB_dB(j) = 10*log10(CRB);
                else
                    CRB_dB(j) = NaN;
                end
            end
            plot(SNR_dB_list, MSE_result_dB, '-', 'LineWidth', 2, 'Color', colors(idxN,:));
            if all(~isnan(CRB_dB))
                plot(SNR_dB_list, CRB_dB, '--', 'LineWidth', 2, 'Color', colors(idxN,:));
            end
        end
        xlabel('SNR (dB)'); ylabel('MSE / CRB (dB)');
        title(sprintf('1.2: MSE vs CRB, \\omega_0 = %.2f\\pi', omega0/pi));
        legend(arrayfun(@(n) sprintf('N=%d', n), N_list, 'UniformOutput', false));
        hold off;
    end
end

%% ==== 1.2.3: MSE vs CRB, rho = 0.9 ====
function SPL_Hw1_1_2_3(sigma_w, omega0_list, N_list, SNR_dB_list, filter_coeffs, L, filter_case)
    rho = 0.9;
    for idxOmega = 1:length(omega0_list)
        omega0 = omega0_list(idxOmega);
        figure; hold on; grid on;
        colors = lines(length(N_list));
        for idxN = 1:length(N_list)
            N = N_list(idxN);
            i = (0:N-1)';
            r = sigma_w^2 * rho.^(0:N-1);
            Cww = toeplitz(r);
            W_all = generate_noise(Cww, L);
            MSE_dB = zeros(1, length(SNR_dB_list));
            CRB_dB = zeros(1, length(SNR_dB_list));
            for j = 1:length(SNR_dB_list)
                snr_linear = 10^(SNR_dB_list(j)/10);
                a0 = sqrt(2 * sigma_w^2 * snr_linear);
                omega_hat = zeros(1, L);
                parfor l = 1:L
                    phi0 = 2*pi*rand;
                    w = W_all(:, l);
                    y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
                    omega_hat(l) = estimate_frequency_fft_interp(y, N);
                end
                mse = mean((omega_hat - omega0).^2);
                MSE_dB(j) = 10*log10(mse);

                if strcmp(filter_case, "delta")
                    v = Cww \ i;
                    J = (a0^2 / 2) * (i' * v);
                    CRB = 1 / J;
                    CRB_dB(j) = 10*log10(CRB);
                else
                    CRB_dB(j) = NaN;
                end
            end
            plot(SNR_dB_list, MSE_dB, '-', 'LineWidth', 2, 'Color', colors(idxN,:));
            if all(~isnan(CRB_dB))
                plot(SNR_dB_list, CRB_dB, '--', 'LineWidth', 2, 'Color', colors(idxN,:));
            end
        end
        xlabel('SNR (dB)'); ylabel('MSE / CRB (dB)');
        title(sprintf('1.2.3: MSE vs CRB, \\omega_0 = %.2f\\pi, \\rho = %.1f', omega0/pi, rho));
        legend(arrayfun(@(n) sprintf('N=%d', n), N_list, 'UniformOutput', false));
        hold off;
    end
end

%% ==== 支持函数 ====
function Cww = generate_covariance_matrix(N, sigma_w, rho)
    Cww = sigma_w^2 * rho .^ abs((0:N-1)' - (0:N-1));
end

function W = generate_noise(Cww, L)
    [Q, Lambda] = eig(Cww);
    Lambda = max(diag(Lambda), 0);
    A = Q * diag(sqrt(Lambda));
    W = A * randn(size(Cww,1), L);
end

function Chat = generate_sample_covariance(W)
    [~, L] = size(W);
    Wc = W - mean(W, 2);
    Chat = (Wc * Wc') / L;
end

function mse = compute_mse(Chat, Cww)
    mse = mean((Chat(:) - Cww(:)).^2);
end

function y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0)
    n = 0:N-1;
    x = a0 * cos(omega0 * n + phi0);
    if strcmp(filter_case, "filter_H")
        x = filter(filter_coeffs.b, filter_coeffs.a, x);
    end
    y = x(:) + w(:);
end

function omega_hat = estimate_frequency_fft_interp(y, N)
    N_fft = 4*N;
    Y = abs(fft(y, N_fft));
    [~, k] = max(Y);
    if k == 1 || k == N_fft
        omega_hat = 2*pi*(k-1)/N_fft;
        return;
    end
    alpha = Y(k-1); beta = Y(k); gamma = Y(k+1);
    delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma);
    k_hat = k + delta;
    omega_hat = 2*pi*(k_hat - 1)/N_fft;
end
