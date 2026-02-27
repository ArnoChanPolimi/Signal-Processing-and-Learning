
% SPL_Hw1_1_2_3_Optimized.m
% Optimized code for estimating frequency bias, MSE, and CRB

clear; clc;

% Parameters
N_list = [100, 500, 1024];
omega0_list = [0.79, 1.57, 2.22, 2.54];
SNR_dB = -30:5:40;
MC = 100; % Monte Carlo trials

% Preallocate results
Bias = zeros(length(SNR_dB), length(omega0_list), length(N_list));
MSE = zeros(length(SNR_dB), length(omega0_list), length(N_list));
CRB = zeros(length(SNR_dB), length(N_list));

% Main loop
for idxN = 1:length(N_list)
    N = N_list(idxN);
    n = 0:N-1;
    
    for idxOmega = 1:length(omega0_list)
        omega0 = omega0_list(idxOmega);
        
        for idxSNR = 1:length(SNR_dB)
            snr_db = SNR_dB(idxSNR);
            snr = 10^(snr_db/10);
            sigma2 = 1 / snr;
            
            mse_temp = zeros(1, MC);
            bias_temp = zeros(1, MC);
            
            for mc = 1:MC
                noise = sqrt(sigma2/2)*(randn(1,N) + 1j*randn(1,N));
                x = exp(1j * omega0 * n) + noise;
                
                % FFT-based frequency estimation
                L = 4*N;
                X = abs(fft(x, L));
                [~, idx] = max(X);
                omega_hat = 2*pi*(idx-1)/L;
                
                % Phase-wrapped error
                err = angle(exp(1j*(omega_hat - omega0)));
                mse_temp(mc) = err^2;
                bias_temp(mc) = err;
            end
            
            MSE(idxSNR, idxOmega, idxN) = mean(mse_temp);
            Bias(idxSNR, idxOmega, idxN) = mean(bias_temp);
            
            % CRB: Cramer-Rao Bound (approximation for frequency)
            CRB(idxSNR, idxN) = 6 / (snr * N * (N^2 - 1));
        end
    end
end

% Plot example for omega0 = 0.79 (1st)
figure;
hold on;
for idxN = 1:length(N_list)
    semilogy(SNR_dB, squeeze(MSE(:,1,idxN)), 'LineWidth', 1.5);
    semilogy(SNR_dB, CRB(:,idxN), '--', 'LineWidth', 1.5);
end
xlabel('SNR (dB)'); ylabel('MSE / CRB'); grid on;
legend('MSE N=100','CRB N=100','MSE N=500','CRB N=500','MSE N=1024','CRB N=1024');
title('MSE vs CRB at \omega_0 = 0.79');
