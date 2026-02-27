%% %%%%%% ----- SP&L Homework2 -------- %%%%%%%%%%%%%%%
%% %%%% ----------- RLS -----------------%%%%%%%
clear all;
% 输入：fs（采样率）, Sin_a（含噪立体声信号）, Sn_ref_a（参考噪声信号）
% 输出：S_est（去噪信号）, 对比波形图, 播放去噪音频

%% 1. 加载数据并预处理
% Load data from .mat file (假设数据文件为Hw2a.mat)
load('Hw2a.mat');          % 替换为实际文件名
% load('Hw2b.mat');        % 其他数据集取消注释
% load('Hw2c.mat');



% 参数设置
fs = double(fs);           % 确保采样率为double类型
N = 64;                    % RLS滤波器长度（建议32-128）
lambda = 0.998;            % 遗忘因子（0.95-0.999）
delta = 0.01;              % 正则化参数（确保矩阵可逆）

% 初始化信号存储
signal_len = length(Sin_a);
S_est = zeros(signal_len, 2); % 去噪后的立体声信号

%% 2. RLS滤波器初始化（双通道独立滤波）
% 左声道初始化
w_left = zeros(N, 1);      % 初始权重向量
P_left = (1/delta) * eye(N); % 初始逆相关矩阵

% 右声道初始化（与左声道独立）
w_right = zeros(N, 1);
P_right = (1/delta) * eye(N);

%% 3. 执行RLS算法
for k = 1:signal_len
    % 获取当前参考信号向量（滑动窗口）
    ref_idx = max(1, k-N+1):k;
    if length(ref_idx) < N
        % 起始阶段填充零
        s_ref_vec = [zeros(N-length(ref_idx), 1); Sn_ref_a(ref_idx)]';
    else
        s_ref_vec = Sn_ref_a(ref_idx)';
    end
    
    % --- 左声道处理 ---
    % 计算先验误差
    e_left = Sin_a(k,1) - w_left' * s_ref_vec;
    
    % 计算卡尔曼增益
    K_left = (P_left * s_ref_vec) / (lambda + s_ref_vec' * P_left * s_ref_vec);
    
    % 更新权重
    w_left = w_left + K_left * e_left;
    
    % 更新逆相关矩阵
    P_left = (1/lambda) * (P_left - K_left * s_ref_vec' * P_left);
    
    % 去噪信号估计
    S_est(k,1) = Sin_a(k,1) - w_left' * s_ref_vec;
    
    % --- 右声道处理（同理）---
    e_right = Sin_a(k,2) - w_right' * s_ref_vec;
    K_right = (P_right * s_ref_vec) / (lambda + s_ref_vec' * P_right * s_ref_vec);
    w_right = w_right + K_right * e_right;
    P_right = (1/lambda) * (P_right - K_right * s_ref_vec' * P_right);
    S_est(k,2) = Sin_a(k,2) - w_right' * s_ref_vec;
end

%% 4. 播放音频对比
disp('播放原始含噪音频...');
sound(Sin_a, fs); pause(5);
disp('播放去噪后音频...');
sound(S_est, fs);

%% 5. 绘制时域对比图
t = (0:signal_len-1)/fs;
figure('Position', [100, 100, 1200, 600]);

% 左声道对比
subplot(2,1,1);
plot(t, Sin_a(:,1), 'Color', [0.7 0.7 0.7], 'LineWidth', 1); hold on;
plot(t, S_est(:,1), 'b', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('幅度');
title('左声道对比');
legend('含噪信号', '去噪信号', 'Location', 'northeast');
xlim([0, t(end)]);

% 右声道对比
subplot(2,1,2);
plot(t, Sin_a(:,2), 'Color', [0.7 0.7 0.7], 'LineWidth', 1); hold on;
plot(t, S_est(:,2), 'r', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('幅度');
title('右声道对比');
legend('含噪信号', '去噪信号', 'Location', 'northeast');
xlim([0, t(end)]);

% 保存图像
saveas(gcf, 'Noise_Cancellation_Comparison.png');

%% 6. 可选：保存去噪音频（需Audio Toolbox）
% audiowrite('Denoised_Audio.wav', S_est, fs);