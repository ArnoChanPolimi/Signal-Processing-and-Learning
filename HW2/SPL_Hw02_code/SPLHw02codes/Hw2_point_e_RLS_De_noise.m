%%%%%%%%%%%  Hw2 - point e: RLS denoise %%%%%%%%
clear all;
close all;
clc;

Sin = load('Hw2e.mat'); 
load('Sn_ref_d1.mat');

fs = Sin.fs;
sin_frame_e = Sin.Sin_e;     
sn_ref_e = sn_ref_matirx;  
sin_e = sin_frame_e(:, 1);         
sn_ref_e1 = sn_ref_e(:, 1);   
sn_ref_e2 = sn_ref_e(:, 2);   

% RLS
K_order = 10;     

num_samples = length(sin_e);  

lambda = 0.999; % forgetting factor
delta = 1e-6;   % initialization parameter           
W = zeros(K_order, 1); % filter weights
R = delta * eye(K_order); % initial correlation matrix

s_est_e = zeros(num_samples, 1);

for k = 1:num_samples
    sn_ref_frame = sn_ref_e(:, k);   

    % gain vector
    K = (R * sn_ref_frame) / (lambda + sn_ref_frame' * R * sn_ref_frame);
    % error signal
    ERROR = sin_e(k) - W' * sn_ref_frame;

    W = W + K * ERROR; % update weights
    R = (1 / lambda) * (R - K * sn_ref_frame' * R); % update correlation matrix

    % estimated signal
    s_est_e(k) = ERROR;  
end

N = 800; % window size
L = N / 2;  
window = hamming(N); 

figure;
[S1, F1, T1] = spectrogram(sin_e, window, L, N, fs);
imagesc(T1, F1, 20 * log10(abs(S1)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Point e: Time-Frequency Analysis of Signals mixed with noise');
colorbar;

figure;
[S2, F2, T2] = spectrogram(s_est_e, window, L, N, fs);
imagesc(T2, F2, 20 * log10(abs(S2)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Point e: Time-Frequency Analysis using RLS');
colorbar;

pause(0.2);  % ensure plot is fully rendered
sound(s_est_e, fs);
