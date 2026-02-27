%% %%%%%% ----- SP&L Homework2 -------- %%%%%%%%%%%%%%%
%% %%%% ----------- RLS -----------------%%%%%%%
clear all;
load("Hw2a.mat");

% 数据设置
Sin_x = Sin_a;                % 原始信号
Sn_ref_x = Sn_ref_a;          % 噪声参考信号
[N, M] = size(Sin_x);         % N为样本数量，M为通道数量
% 参数设置
filter_order = 20;
segment_length = filter_order; %500;          % 每段长度
overlap_length = segment_length/2;          % 每段重叠长度（增大重叠比例）
lambda = 0.9; %遗忘因子（0≤<λ≤1），用于控制历史数据的加权，较小的 λ 表示更多地依赖最新数据
mu_0 = 0.01;                   % 初始步长
alpha = 10;                    % 动态步长调整参数

% 初始化
segments = floor((N - overlap_length) / (segment_length - overlap_length)); % 分段数
Sin_est = zeros(N, M);         % 去噪后的信号初始化
Adapter = zeros(filter_order, M); % 滤波器权重初始化

% 窗函数（使用Tukey窗进行平滑处理）
window = tukeywin(segment_length, 0.5);  % Tukey窗
overlap_window = window(1:overlap_length);  % 重叠区域窗函数

P = repmat(eye(filter_order), [1, 1, M]);  % 协方差矩阵初始化，假设为单位矩阵
epsilon = 1e-6;  % 为避免除零的一个小常数

% 分段处理
for seg = 1:segments
    % 当前段的起始与结束索引
    start_idx = (seg - 1) * (segment_length - overlap_length) + 1;
    end_idx = start_idx + segment_length - 1;

    % 获取当前分段数据
    Sin_segment = Sin_x(start_idx:end_idx, :);  % 当前分段的输入信号
    Sn_ref_segment = Sn_ref_x(start_idx:end_idx, :);  % 当前分段的参考信号

    % 对于每个声道处理
    for m = 1:M
        % 初始化当前声道的增益向量和误差
        K = zeros(filter_order, 1);  % 当前声道增益向量
        e = zeros(segment_length, 1);  % 当前声道的估计误差

        % 当前声道的滤波器权重
        a_m = Adapter(:, m);

        % 当前声道的协方差矩阵
        P_m = P(:, :, m);


            % 计算增益向量 K(n)
            K = P_m * Sn_ref_segment(:,m) / (lambda + Sn_ref_segment(:, m)' * P_m * Sn_ref_segment(:, m) + epsilon);

            % 计算滤波器输出
            output_pred = conv(a_m, Sn_ref_segment(:, m));

            % 计算估计误差 e(n)
            e(:, m) = Sin_segment(:, m) - output_pred(1:segment_length);

            % 更新权重 a_m
            a_m = a_m + K .* e(:,m);

            % 更新协方差矩阵 P_m
            P_m = (1 / lambda) * (P_m - K * Sn_ref_segment(:, m)' * P_m);

            % 将更新后的权重和协方差存入相应的矩阵
            Adapter(:, m) = a_m;
            P(:, :, m) = P_m;
        end
        
        Sin_segment_est = conv(Adapter(:,m), Sn_ref_segment(:, m));

        % 融合重叠区域
        if seg > 1
            overlap_start = start_idx;
            overlap_end = start_idx + overlap_length - 1;

            % 平滑连接
            Sin_est(overlap_start:overlap_end, m) = ...
                (Sin_est(overlap_start:overlap_end, m) .* (1 - overlap_window) + ...
                Sin_segment_est(1:overlap_length) .* overlap_window)/2;
        end

        % 保存非重叠区域信号
        Sin_est(start_idx + overlap_length:end_idx, m) = ...
            Sin_segment_est(overlap_length+1:segment_length);

end

% 播放去噪后的信号
sound(Sin_est, fs);

% 绘图对比
figure;
subplot(2, 1, 1);
plot(Sin_x(:, 1), 'b'); hold on;
plot(Sin_est(:, 1), 'r');
title('通道 1: 原始信号与去噪信号对比');
xlabel('样本点'); ylabel('信号幅值');
legend('原始信号', '去噪信号');

subplot(2, 1, 2);
plot(Sin_x(:, 2), 'b'); hold on;
plot(Sin_est(:, 2), 'r');
title('通道 2: 原始信号与去噪信号对比');
xlabel('样本点'); ylabel('信号幅值');
legend('原始信号', '去噪信号');
