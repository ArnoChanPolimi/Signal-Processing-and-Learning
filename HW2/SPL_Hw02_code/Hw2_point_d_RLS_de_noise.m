%%%%%%%%%%%  Hw2 - point d: RLS denoise %%%%%%%%
clear all;
close all;
clc;

Sin = load('Hw2d.mat'); 
load('Sn_ref_d1.mat');

fs = Sin.fs;
sin_frame_d = Sin.Sin_d;     
sn_ref_d = sn_ref_matirx;  
sin_d = sin_frame_d(:, 1);         
sn_ref_d1 = sn_ref_d(:, 1);   
sn_ref_d2 = sn_ref_d(:, 2);   

% RLS parameters
K_o = 10;         % filter order
num_samples = length(sin_d);  
lambda = 0.999;   % forgetting factor
delta = 1e-6;     % initialization value for correlation matrix
W = zeros(K_o, 1);          % filter weights
R = delta * eye(K_o);       % initial correlation matrix

s_est_d = zeros(num_samples, 1);  % estimated clean signal

for k = 1:num_samples
    sn_ref_frame = sn_ref_d(:, k);  

    % gain vector
    K = (R * sn_ref_frame) / (lambda + sn_ref_frame' * R * sn_ref_frame);
    
    % error signal (desired - estimated)
    ERROR = sin_d(k) - W' * sn_ref_frame;
    
    % update filter weights
    W = W + K * ERROR;

    % update correlation matrix
    R = (1 / lambda) * (R - K * sn_ref_frame' * R);
    
    % store estimated signal
    s_est_d(k) = ERROR;  
end

% Spectrogram settings
N = 800;               % FFT window size
L = N / 2;             % overlap
window = hamming(N);   % window function

% Original signal spectrogram
figure('Name','Original Signal Spectrogram','NumberTitle','off');
[S1, F1, T1] = spectrogram(sin_d, window, L, N, fs);
imagesc(T1, F1, 20 * log10(abs(S1)));
axis xy;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Point d: Time-Frequency Analysis of Signals mixed with noise');
colorbar;

% RLS-filtered signal spectrogram
figure('Name','RLS Processed Spectrogram','NumberTitle','off');
[S2, F2, T2] = spectrogram(s_est_d, window, L, N, fs);
imagesc(T2, F2, 20 * log10(abs(S2)));
axis xy;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Point d: Time-Frequency Analysis using RLS');
colorbar;

% Display basic statistics of the output
disp([min(s_est_d), max(s_est_d), mean(s_est_d)]);

% Play denoised signal
sound(s_est_d, fs);
% pause(length(s_est_d)/fs + 0.5); % Optional: wait for audio to finish
