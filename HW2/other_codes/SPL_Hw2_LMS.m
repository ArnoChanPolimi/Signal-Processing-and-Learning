%% %%%%%% ----- SP&L Homework2 -------- %%%%%%%%%%%%%%%
%% %%%% ----------- LMS -----------------%%%%%%%
clear all;
load("Hw2a.mat");

% 数据设置
Sin_x = Sin_a;                % 原始信号
Sn_ref_x = Sn_ref_a;          % 噪声参考信号
[N, M] = size(Sin_x);         % N为样本数量，M为通道数量
% 参数设置
filter_order = 20;
segment_length = 500;          % 每段长度
overlap_length = 250;          % 每段重叠长度（增大重叠比例）
mu_0 = 0.01;                   % 初始步长
alpha = 10;                    % 动态步长调整参数

% 初始化
segments = floor((N - overlap_length) / (segment_length - overlap_length)); % 分段数
Sin_est = zeros(N, M);         % 去噪后的信号初始化
Adapter = zeros(filter_order, M); % 滤波器权重初始化

% 窗函数（使用Tukey窗进行平滑处理）
window = tukeywin(segment_length, 0.5);  % Tukey窗
overlap_window = window(1:overlap_length);  % 重叠区域窗函数

% 分段处理
for seg = 1:segments
    % 当前段的起始与结束索引
    start_idx = (seg - 1) * (segment_length - overlap_length) + 1;
    end_idx = start_idx + segment_length - 1;

    % 当前段信号
    Sin_segment = Sin_x(start_idx:end_idx, :);
    Sn_ref_segment = Sn_ref_x(start_idx:end_idx, :);

    % 批量计算误差和更新权重
    for col = 1:M
        signal_col = Sin_segment(:, col);         % 当前通道的信号
        ref_col = Sn_ref_segment(:, col);         % 当前通道的参考信号

        % 预测输出 (加权和)
        output_pred = zeros(segment_length, 1);  % 初始化预测输出
        mid1 = conv(Adapter(:, col), ref_col(:));
        output_pred = mid1(1:segment_length);

        % 误差向量
        error = signal_col - output_pred;

        % 动态步长（基于误差平方）
        step_size = mu_0 ./ (1 + alpha * error.^2);

        % 更新权重
        mid2 = step_size .* error .* ref_col(:);
        Adapter(:, col) = Adapter(:, col) + 2*mid2(1:filter_order);

        % 去噪信号
        Sin_segment_est = signal_col - output_pred;

        % 融合重叠区域
        if seg > 1
            overlap_start = start_idx;
            overlap_end = start_idx + overlap_length - 1;

            % 平滑连接
            Sin_est(overlap_start:overlap_end, col) = ...
                (Sin_est(overlap_start:overlap_end, col) .* (1 - overlap_window) + ...
                Sin_segment_est(1:overlap_length) .* overlap_window)/2;

            % 低通滤波进一步平滑
            Sin_est(overlap_start:overlap_end, col) = ...
                lowpass(Sin_est(overlap_start:overlap_end, col), 0.1, fs);  % 低通滤波
        end

        % 保存非重叠区域信号
        Sin_est(start_idx + overlap_length:end_idx, col) = ...
            Sin_segment_est(overlap_length+1:end);
    end
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
