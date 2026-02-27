clear; clc;

Fun_2_1();
Fun_2_2();
Fun_2_3();


% ==== 参数设置 ====
function Fun_2_1()
% 项目 1.2.1 - rho=0, h=delta, 多N值对比

a0 = 1;
N_values = [100 500 1024];
omega = pi/4;
SNR_dB = -20:5:40;
rho = 0;
h_case = "delta";
filter_coeffs.b = [2];
filter_coeffs.a = [1, 0.9];
num_simulations = 400;
oversampling = 64;

results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0, oversampling, num_simulations);

% 绘图
figure; hold on;
color_map = lines(length(N_values));
for i = 1:length(N_values)
    plot(SNR_dB, 10*log10(results.MSE(i,:)), '-', 'Color', color_map(i,:), ...
        'DisplayName', sprintf('MSE (N=%d)', N_values(i)), 'LineWidth', 2);
    plot(SNR_dB, 10*log10(results.CRB(i,:)), '--', 'Color', color_map(i,:), ...
        'DisplayName', sprintf('CRB (N=%d)', N_values(i)), 'LineWidth', 2);
end
xlabel('SNR (dB)'); ylabel('MSE (dB)');
title('1.2.1: h=\delta, \rho=0');
legend show; grid on;
end

function Fun_2_2()
% 项目 1.2.2 - rho=0, h=filter, 多N值对比

a0 = 1;
N_values = [100 500 1024];
omega = pi/4;
SNR_dB = -20:5:40;
rho = 0;
h_case = "filter";
filter_coeffs.b = [2];
filter_coeffs.a = [1, 0.9];
num_simulations = 400;
oversampling = 64;

results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0, oversampling, num_simulations);

% 绘图
figure; hold on;
color_map = lines(length(N_values));
for i = 1:length(N_values)
    plot(SNR_dB, 10*log10(results.MSE(i,:)), '-', 'Color', color_map(i,:), ...
        'DisplayName', sprintf('MSE (N=%d)', N_values(i)), 'LineWidth', 2);
    plot(SNR_dB, 10*log10(results.CRB(i,:)), '--', 'Color', color_map(i,:), ...
        'DisplayName', sprintf('CRB (N=%d)', N_values(i)), 'LineWidth', 2);
end
xlabel('SNR (dB)'); ylabel('MSE (dB)');
title('1.2.2: h=filter, \rho=0');
legend show; grid on;
end

function Fun_2_3()
% 项目 1.2.3 - rho=0.9, 对比 h=delta 和 h=filter

a0 = 1;
N_values = [100 500 1024];             % 固定 N
omega = pi/4;
SNR_dB = -20:5:40;
rho = 0.9;
filter_coeffs.b = [2];
filter_coeffs.a = [1, 0.9];
num_simulations = 400;
oversampling = 64;

h_cases = ["delta", "filter"];
colors = lines(2);

figure; hold on;

for k = 1:2
    h_case = h_cases(k);
    results = simulate_estimation(N_values, SNR_dB, omega, rho, h_case, filter_coeffs, a0, oversampling, num_simulations);

    plot(SNR_dB, 10*log10(results.MSE(1,:)), '-', ...
        'Color', colors(k,:), 'LineWidth', 2, 'DisplayName', sprintf('MSE (%s)', h_case));
    
    plot(SNR_dB, 10*log10(results.CRB(1,:)), '--', ...
        'Color', colors(k,:), 'LineWidth', 2, 'DisplayName', sprintf('CRB (%s)', h_case));
end

xlabel('SNR (dB)');
ylabel('MSE (dB)');
title('1.2.3: \rho=0.9, h=\delta vs h=filter');
legend show; grid on;
end


function results = simulate_estimation(N_values, SNR_dB, omega_true, rho, h_case, filter_coeffs, a0, oversampling, num_simulations)
results = struct();
results.MSE = zeros(length(N_values), length(SNR_dB));
results.CRB = zeros(length(N_values), length(SNR_dB));

for i = 1:length(N_values)
    N = N_values(i);
    for j = 1:length(SNR_dB)
        snr_linear = 10^(SNR_dB(j)/10);
        sigma_w2 = a0 / (2 * snr_linear);
        omega_estimates = zeros(num_simulations, 1);
        for run = 1:num_simulations
            x = generate_signal(N, omega_true, sigma_w2, rho);
            if h_case == "filter"
                x = filter_signal(x, filter_coeffs);
            end
            omega_estimates(run) = estimate_frequency_fft(x, N, oversampling);
        end
        results.MSE(i, j) = mean((omega_estimates - omega_true).^2);
        results.CRB(i, j) = compute_crb(N, snr_linear);
    end
end
end


function omega_estimate = estimate_frequency_fft(x, N, oversampling)
N_fft = oversampling * N;
X = fft(x, N_fft);
P = abs(X(1:N_fft/2)).^2;
[~, idx_max] = max(P);

if idx_max <= 1 || idx_max >= length(P) - 1
    omega_estimate = 2 * pi * (idx_max - 1) / N_fft;
else
    alpha = P(idx_max - 1);
    beta = P(idx_max);
    gamma = P(idx_max + 1);
    delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma);
    f_refined = (idx_max - 1 + delta) / N_fft;
    omega_estimate = 2 * pi * f_refined;
end
end


function x = generate_signal(N, omega, sigma_w2, rho)
A = 1;
phi_0 = 2 * pi * rand();
n = (0:N-1)';
s = A * cos(omega * n + phi_0);
Cww = sigma_w2 * rho.^(abs((1:N)' - (1:N)));
A_cov = sqrtm(Cww);
w = A_cov * randn(N, 1);
x = s + w;
end

function x_filtered = filter_signal(x, filter_coeffs)
x_filtered = filter(filter_coeffs.b, filter_coeffs.a, x);
end

function crb = compute_crb(N, snr_linear)
crb = 6 / (snr_linear * N * (N - 1) * (2 * N - 1));
end

