% %% %%%%%% ----- SP&L Homework2 -------- %%%%%%%%%%%%%%%
%% %%%%%%% 卡尔曼滤波 %%
clear all;
load("Hw2a.mat");


% 数据设置
Sin_x = Sin_a;                % 原始信号
Sn_ref_x = Sn_ref_a;          % 噪声参考信号

[N, M] = size(Sin_x);         % N为样本数量，M为通道数量


p = 100;
a = zeros(p, M);  % 滤波器权重初始化
P = eye(p) * 1e-2;  % 初始协方差矩阵
Q = eye(p) * 1e-8;  % 过程噪声协方差
R = 1e-3;  % 观测噪声协方差

% 去噪过程
Sin_est = zeros(N, M);  % 去噪后的信号
for n = 1:N
    for col = 1:M
        % 参考信号窗口
        if n >= p
            s_ref_window = Sn_ref_x(n-p+1:n, col);
        else
            s_ref_window = [zeros(p-n, 1); Sn_ref_x(1:n, col)];
        end

        % 预测阶段
        a_pred = a(:, col);  % 权重预测
        P_pred = P + Q;  % 协方差预测

        % 卡尔曼增益
        K = (P_pred * s_ref_window) / (s_ref_window' * P_pred * s_ref_window + R);

        % 更新权重
        a(:, col) = a_pred + K * (Sin_x(n, col) - s_ref_window' * a_pred);

        % 更新协方差
        P = (eye(p) - K * s_ref_window') * P_pred;

        % 估计噪声
        w_hat = a(:, col)' * s_ref_window;

        % 去噪
        Sin_est(n, col) = Sin_x(n, col) - w_hat;
    end
end

sound(Sin_est);



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
