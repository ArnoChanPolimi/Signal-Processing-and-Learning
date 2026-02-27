% 加载信号数据
load('Hw2d.mat'); % 或者 load('Hw2e.mat')
% 假设信号变量是 Sin_d 或 Sin_e
Sin_x = Sin_d;  % 或者 Sin_e，根据加载的文件名选择
N = length(Sin_x);  % 样本数量

% 定义参数
segment_length = 500; % 每段的长度
window = hamming(segment_length); % 设定窗口函数
Fs = 44100; % 采样频率，假设为44.1kHz
noverlap = 250; % 重叠部分的长度，通常为 segment_length / 2

% 初始化去噪后的信号
Sin_est = zeros(N, 1);

% 初始化周期图矩阵 P
num_segments = floor((N - noverlap) / (segment_length - noverlap));
P = zeros(num_segments, segment_length);

% 进行信号去噪和频谱分析
for k = 1:num_segments
    % 获取每段的起始和结束索引
    start_idx = (k - 1) * (segment_length - noverlap) + 1;
    end_idx = start_idx + segment_length - 1;
    
    if end_idx > N
        break;
    end
    
    % 提取当前段数据
    segment = Sin_x(start_idx:end_idx);
    
    % 对信号加窗
    segment_windowed = segment .* window;
    
    % 计算该段的周期图
    segment_periodogram = abs(fft(segment_windowed)).^2;  % 计算功率谱
    
    % 将该段的周期图添加到P矩阵中
    P(k, :) = segment_periodogram;
    
    % 估计该段的频谱去噪（这里使用简单的阈值法来去除噪声频率）
    threshold = median(segment_periodogram) * 1.5; % 设置阈值
    segment_denoised = segment_windowed;
    segment_denoised(abs(segment_periodogram) < threshold) = 0; % 去噪处理
    
    % 将去噪后的信号添加到 Sin_est 中
    Sin_est(start_idx:end_idx) = segment_denoised;
end

% 画出时频图（期望图像）
figure;
imagesc(P');
title('Time-Frequency Representation (Periodogram)');
xlabel('Time (segments)');
ylabel('Frequency bins');
colorbar;

% 播放去噪后的音频
sound(Sin_est, Fs);

% 保存去噪后的信号
audiowrite('denoised_signal.wav', Sin_est, Fs);
