%% %%%%%% ----- SP&L Homework2 -------- %%%%%%%%%%%%%%%
%% %%%% -----------  -----------------%%%%%%%
clear all;
load("Hw2a.mat");

% 数据设置
Sin_x = Sin_a;                % 原始信号
Sn_ref_x = Sn_ref_a;          % 噪声参考信号
[N, M] = size(Sin_x);         % N为样本数量，M为通道数量

% 分块大小
block_size = 1000;  % 可以根据内存情况选择适当的块大小

% % 初始化互相关矩阵
% P_1_1 = zeros(N, N);

% 分块计算互相关矩阵
for i = 1:block_size:N
    for j = 1:block_size:N
        % 计算当前块的互相关矩阵部分
        P_1_1(i:i+block_size-1, j:j+block_size-1) = ...
            (Sin_x(i:i+block_size-1,1) * Sn_ref_x(j:j+block_size-1,1)') / N;
    end
end
% 计算互相关矩阵
P_1_1 = (Sin_x(:,1) * Sn_ref_x(:,1)') / N;  % 样本互相关矩阵，外积并除以N进行标准化
% P_1_2 = (Sin_x(:,1) * Sn_ref_x(:,2)') / N;
% P_2_1 = (Sin_x(:,2) * Sn_ref_x(:,1)') / N;
% P_2_2 = (Sin_x(:,2) * Sn_ref_x(:,2)') / N;
plot(P_1_1);
