clc; clear all; close all;

%% 参数设置
rho_list = [0, 0.5, 0.9];
N_list = [100, 500, 1024];
SNR_dB_list = -30:5:40;
omega0_list = pi * [1/4, 1/2, 1/sqrt(2), 0.81];
L = 100;  % Monte Carlo 次数
sigma_w = 1;
filter_coeffs.b = [2];
filter_coeffs.a = [1, 0.9];

%% ==== 主程序入口 ====
SPL_Hw1_1_1_MSEvsN_with_diff_rho(rho_list, round(linspace(20, 200, 10)), 100, sigma_w);
SPL_Hw1_1_2(sigma_w, omega0_list, N_list, SNR_dB_list, rho_list, filter_coeffs, L);
SPL_Hw1_1_2_3(sigma_w, omega0_list, N_list, SNR_dB_list, filter_coeffs, L);

%% ==== 1.1 ====
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

%% ==== 1.2 ====
function SPL_Hw1_1_2(sigma_w, omega0_list, N_list, SNR_dB_list, rho_list, filter_coeffs, L)
    rho = rho_list(1);
    filter_case = "filter_H";
    for idxOmega = 1:length(omega0_list)
        omega0 = omega0_list(idxOmega);
        figure; hold on; grid on;
        colors = lines(length(N_list));
        for idxN = 1:length(N_list)
            N = N_list(idxN);
            MSE_result_dB = zeros(1, length(SNR_dB_list));
            CRB_dB = zeros(1, length(SNR_dB_list));
            for j = 1:length(SNR_dB_list)
                snr_linear = 10^(SNR_dB_list(j)/10);
                a0 = sqrt(2 * sigma_w^2 * snr_linear);
                omega_hat = zeros(1, L);
                for l = 1:L
                    phi0 = 2*pi*rand;
                    w = sigma_w * randn(N, 1);  % 白噪声直接生成
                    y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
                    omega_hat(l) = estimate_frequency_fft_interp(y, N);
                end
                mse = mean( angle(exp(1j*(omega_hat - omega0))).^2 );
                MSE_result_dB(j) = 10*log10(mse);
                CRB = 6/(snr_linear * N * (N^2 - 1));
                CRB_dB(j) = 10*log10(CRB);
            end
            plot(SNR_dB_list, MSE_result_dB, '-', 'LineWidth', 2, 'Color', colors(idxN,:));
            plot(SNR_dB_list, CRB_dB, '--', 'LineWidth', 2, 'Color', colors(idxN,:));
        end
        xlabel('SNR (dB)'); ylabel('MSE / CRB (dB)');
        title(sprintf('1.2: MSE vs CRB, \\omega_0 = %.2f\\pi', omega0/pi));
        legend(arrayfun(@(n) sprintf('N=%d', n), N_list, 'UniformOutput', false));
        hold off;
    end
end

%% ==== 1.2.3 ====
function SPL_Hw1_1_2_3(sigma_w, omega0_list, N_list, SNR_dB_list, filter_coeffs, L)
    rho = 0.9;
    filter_case = "delta";
    for idxOmega = 1:length(omega0_list)
        omega0 = omega0_list(idxOmega);
        figure; hold on; grid on;
        colors = lines(length(N_list));
        for idxN = 1:length(N_list)
            N = N_list(idxN);
            i = (0:N-1)';
            r = sigma_w^2 * rho.^(0:N-1);
            Cww = toeplitz(r);
            [Q, Lambda] = eig(Cww);
            A = Q * sqrt(max(Lambda, 0));
            MSE_dB = zeros(1, length(SNR_dB_list));
            CRB_dB = zeros(1, length(SNR_dB_list));
            for j = 1:length(SNR_dB_list)
                snr_linear = 10^(SNR_dB_list(j)/10);
                a0 = sqrt(2 * sigma_w^2 * snr_linear);
                omega_hat = zeros(1, L);
                for l = 1:L
                    phi0 = 2*pi*rand;
                    w = A * randn(N,1);  % 相关噪声
                    y = generate_y(N, omega0, phi0, w, filter_coeffs, filter_case, a0);
                    omega_hat(l) = estimate_frequency_fft_interp(y, N);
                end
                mse = mean( angle(exp(1j*(omega_hat - omega0))).^2 );
                MSE_dB(j) = 10*log10(mse);
                v = Cww \ i;
                J = (a0^2 / 2) * (i' * v);
                CRB = 1 / J;
                CRB_dB(j) = 10*log10(CRB);
            end
            plot(SNR_dB_list, MSE_dB, '-', 'LineWidth', 2, 'Color', colors(idxN,:));
            plot(SNR_dB_list, CRB_dB, '--', 'LineWidth', 2, 'Color', colors(idxN,:));
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
    A = Q * sqrt(max(Lambda, 0));
    W = A * randn(size(Cww,1), L);
end

function Chat = generate_sample_covariance(W)
    Wc = W - mean(W, 2);
    Chat = (Wc * Wc') / size(W,2);
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
