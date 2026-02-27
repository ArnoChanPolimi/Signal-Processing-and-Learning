%% %%%%%  SP&L-WORKSHOP-Homework (Optimized Version) %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  ---------- Frequency Estimation with Parabolic Regression -------- %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  ---------- Date: 2025-07-28 -------- %%%%%%%%%

clear all;
close all;
clc;

%% Global Parameters
a0 = 1;                            % Signal amplitude
N_values = [100, 500, 1024];       % Sample sizes
N_values_1 = linspace(20, 200, 10);% For covariance evaluation
omega_values = pi * [1/4, 1/2, 1/sqrt(2), 0.81]; % Frequencies
SNR_dB = -20:5:40;                 % SNR range [dB]
rho_values = [0, 0.5, 0.99];       % Correlation coefficients
sigma_w = 1;                       % Noise std deviation
L = 2000;                          % Monte Carlo samples
num_simulations = 100;             % Number of simulations for MSE averaging
Oversampling = 8;                  % FFT oversampling factor

% Filter coefficients (H(z) = 2/(1+0.9z^{-1}))
filter_coeffs.b = [2];
filter_coeffs.a = [1, 0.9];

%% Main Execution
% =============== Part 1.1: Noise Generation ===============
results_1_1 = generate_correlated_noise_and_evaluate_mse(N_values_1, rho_values, sigma_w, L);

% =============== Part 1.2: Frequency Estimation ===============
% --- 1.2.1: No filter, white noise ---
results_1_2_1 = run_frequency_estimation(a0, N_values, omega_values, SNR_dB, ...
                                         rho_values(1), "delta", filter_coeffs, ...
                                         num_simulations, Oversampling);

% --- 1.2.2: With filter, white noise ---
results_1_2_2 = run_frequency_estimation(a0, N_values, omega_values, SNR_dB, ...
                                         rho_values(1), "filter", filter_coeffs, ...
                                         num_simulations, Oversampling);

% --- 1.2.3: Correlated noise (ρ=0.9) ---
% Without filter
results_1_2_3a = run_frequency_estimation(a0, N_values, omega_values, SNR_dB, ...
                                          rho_values(3), "delta", filter_coeffs, ...
                                          num_simulations, Oversampling);

% With filter
results_1_2_3b = run_frequency_estimation(a0, N_values, omega_values, SNR_dB, ...
                                          rho_values(3), "filter", filter_coeffs, ...
                                          num_simulations, Oversampling);

% =============== Part 1.3: Frequency Modulation ===============
% [Implementation would go here based on project requirements]

%% %%%%%%%%%%%%%%%%%%  ---------- 1.1 Noise Generation -------- %%%%%%%%
function results = generate_correlated_noise_and_evaluate_mse(N_values, rho_values, sigma_w, L)
    fprintf('Running Part 1.1: Noise Generation...\n');
    
    % Initialize results
    MSE_results = zeros(length(N_values), length(rho_values));
    CRB_samples = zeros(length(N_values), length(rho_values));
    
    % Main loop
    for i = 1:length(N_values)
        N = N_values(i);
        fprintf('Processing N=%d...\n', N);
        
        for j = 1:length(rho_values)
            rho = rho_values(j);
            
            % Generate covariance matrix
            Cww = generate_covariance_matrix(N, sigma_w, rho);
            
            % Generate correlated noise
            w = generate_correlated_noise(Cww, L);
            
            % Estimate sample covariance
            Chat = estimate_sample_covariance(w);
            
            % Compute MSE
            MSE_results(i, j) = compute_mse(Chat, Cww);
            
            % CRB for covariance estimation (theoretical)
            CRB_samples(i, j) = (sigma_w^4/N) * (1 + rho^2)/(1 - rho^2)^2;
        end
    end
    
    % Plot results
    figure('Name', 'Noise Generation: MSE vs Sample Size', 'Position', [100, 100, 800, 600]);
    for j = 1:length(rho_values)
        semilogy(N_values, MSE_results(:, j), 'o-', 'LineWidth', 2, 'DisplayName', ['\rho = ' num2str(rho_values(j))]);
        hold on;
        semilogy(N_values, CRB_samples(:, j), '--', 'LineWidth', 2, 'DisplayName', ['CRB \rho = ' num2str(rho_values(j))]);
    end
    hold off;
    
    title('1.1: MSE of Covariance Matrix Estimation');
    xlabel('Sample Size (N)');
    ylabel('MSE');
    legend('Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    
    % Save results
    results.MSE = MSE_results;
    results.CRB = CRB_samples;
end

%% %%%%%%%%%%%%%%%%%%  ---------- 1.2 Frequency Estimation -------- %%%%%%%%
function results = run_frequency_estimation(a0, N_values, omega_values, SNR_dB, ...
                                           rho, h_case, filter_coeffs, ...
                                           num_simulations, Oversampling)
    fprintf('\nRunning Frequency Estimation: ');
    fprintf('h_case=%s, rho=%.2f\n', h_case, rho);
    
    % Initialize results structure
    results = struct();
    results.MSE = zeros(length(N_values), length(omega_values), length(SNR_dB));
    results.CRB = zeros(length(N_values), length(SNR_DB));
    results.SNR_dB = SNR_dB;
    results.N_values = N_values;
    results.omega_values = omega_values;
    results.rho = rho;
    results.h_case = h_case;
    
    % Precompute CRB for all N and SNR
    for i = 1:length(N_values)
        N = N_values(i);
        for j = 1:length(SNR_dB)
            snr_linear = 10^(SNR_dB(j)/10);
            results.CRB(i, j) = compute_crb(N, snr_linear);
        end
    end
    
    % Process each omega value
    for w_idx = 1:length(omega_values)
        omega = omega_values(w_idx);
        fprintf('  Processing ω=%.4f...\n', omega);
        
        % Initialize MSE storage for this omega
        MSE_omega = zeros(length(N_values), length(SNR_dB));
        
        % Process each sample size
        for n_idx = 1:length(N_values)
            N = N_values(n_idx);
            fprintf('    N=%d: ', N);
            
            % Process each SNR
            for s_idx = 1:length(SNR_dB)
                SNR_val = SNR_dB(s_idx);
                snr_linear = 10^(SNR_val/10);
                sigma_w2 = (a0^2/2) / snr_linear;  % Correct noise variance
                
                % Initialize error storage
                errors = zeros(num_simulations, 1);
                
                % Monte Carlo simulations
                for mc = 1:num_simulations
                    % Generate signal with random phase
                    phi0 = 2*pi*rand();
                    n = (0:N-1)';
                    s = a0 * cos(omega*n + phi0);
                    
                    % Generate correlated noise
                    Cww = generate_covariance_matrix(N, sqrt(sigma_w2), rho);
                    w = generate_correlated_noise(Cww, 1);
                    
                    % Create observed signal
                    x = s + w;
                    
                    % Apply filter if needed
                    if strcmp(h_case, "filter")
                        x = filter(filter_coeffs.b, filter_coeffs.a, x);
                    end
                    
                    % Estimate frequency using parabolic regression
                    omega_est = estimate_frequency_parabolic(x, N, Oversampling);
                    
                    % Store error (normalized angular frequency)
                    errors(mc) = (omega_est - omega)^2;
                    
                    % Progress indicator
                    if mod(mc, num_simulations/10) == 0
                        fprintf('.');
                    end
                end
                
                % Compute MSE for this SNR
                MSE_omega(n_idx, s_idx) = mean(errors);
                fprintf('|');
            end
            fprintf('\n');
        end
        
        % Save MSE for this omega
        results.MSE(:, w_idx, :) = MSE_omega;
        
        % Plot results for this omega
        plot_frequency_results(results, w_idx);
    end
end

%% Core Estimation Function with Parabolic Regression
function omega_estimate = estimate_frequency_parabolic(x, N, Oversampling)
    % Compute FFT size with oversampling
    M = round(Oversampling * N);
    
    % Compute FFT (with zero-padding)
    X = fft(x, M);
    
    % Compute power spectrum (positive frequencies only)
    P = abs(X(1:floor(M/2))).^2 / N;
    
    % Find peak index (avoid DC component)
    [~, k_max] = max(P(2:end));
    k_max = k_max + 1;  % Compensate for skipping DC
    
    % Handle boundary cases
    if k_max == 1
        k_est = 1;
    elseif k_max == length(P)
        k_est = length(P);
    else
        % Parabolic interpolation
        alpha = P(k_max-1);
        beta = P(k_max);
        gamma = P(k_max+1);
        
        % Parabolic interpolation formula
        delta = 0.5 * (alpha - gamma) / (alpha + gamma - 2*beta);
        k_est = k_max + delta;
    end
    
    % Convert to normalized frequency [0, 0.5]
    f_est = (k_est - 1) / M;
    
    % Convert to angular frequency [0, π]
    omega_estimate = 2 * pi * f_est;
end

%% CRB Calculation
function crb = compute_crb(N, snr_linear)
    % Cramér-Rao Bound for frequency estimation
    % Valid for single sinusoid in white Gaussian noise
    crb = 6 / (snr_linear * N * (N^2 - 1));
end

%% Noise Generation Functions
function Cww = generate_covariance_matrix(N, sigma_w, rho)
    % Generate covariance matrix for AR(1) process
    k = abs((1:N)' - (1:N));
    Cww = (sigma_w^2) * rho.^k;
end

function w = generate_correlated_noise(Cww, L)
    % Generate L realizations of correlated noise
    N = size(Cww, 1);
    
    % Eigen decomposition for matrix square root
    [V, D] = eig(Cww);
    A = V * sqrt(D);
    
    % Generate noise
    w = A * randn(N, L);
end

function Chat = estimate_sample_covariance(w)
    % Compute sample covariance matrix
    [N, L] = size(w);
    mu = mean(w, 2);
    w_centered = w - mu;
    Chat = (w_centered * w_centered') / (L-1);
end

function mse = compute_mse(Chat, Cww)
    % Compute MSE between estimated and true covariance
    mse = mean((Chat(:) - Cww(:)).^2);
end

%% Plotting Function
function plot_frequency_results(results, omega_idx)
    % Create figure
    fig = figure('Name', sprintf('Frequency Estimation: ω=%.4f, ρ=%.2f, %s', ...
                 results.omega_values(omega_idx), results.rho, results.h_case), ...
                 'Position', [100, 100, 1000, 700]);
    
    % Get MSE for this omega
    MSE_omega = squeeze(results.MSE(:, omega_idx, :));
    
    % Plot for each N
    colors = lines(length(results.N_values));
    line_styles = {'-', '--', ':', '-.'};
    
    hold all;
    grid on;
    
    % Plot MSE curves
    for n_idx = 1:length(results.N_values)
        N = results.N_values(n_idx);
        
        % MSE curve
        plot(results.SNR_dB, 10*log10(MSE_omega(n_idx, :)), ...
             'LineWidth', 2, ...
             'LineStyle', line_styles{mod(n_idx-1, length(line_styles))+1}, ...
             'Color', colors(n_idx, :), ...
             'DisplayName', sprintf('MSE (N=%d)', N));
        
        % CRB curve
        plot(results.SNR_dB, 10*log10(results.CRB(n_idx, :)), ...
             '--', ...
             'LineWidth', 2, ...
             'Color', colors(n_idx, :), ...
             'DisplayName', sprintf('CRB (N=%d)', N));
    end
    
    % Add theoretical thresholds
    threshold = 10*log10(1/(12*(results.N_values(1)*Oversampling^2)));
    plot(results.SNR_dB, threshold*ones(size(results.SNR_dB)), 'k:', ...
         'LineWidth', 1.5, 'DisplayName', 'Quantization Threshold');
    
    % Finalize plot
    hold off;
    xlabel('SNR (dB)');
    ylabel('MSE (dB)');
    title(sprintf('Frequency Estimation: \\omega=%.4f, \\rho=%.2f, h=%s', ...
          results.omega_values(omega_idx), results.rho, results.h_case));
    legend('Location', 'best');
    set(gca, 'FontSize', 12);
    grid minor;
    
    % Add annotation
    annotation('textbox', [0.15, 0.8, 0.2, 0.1], 'String', ...
        sprintf('Oversampling: %dx\nSimulations: %d', Oversampling, num_simulations), ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white');
end