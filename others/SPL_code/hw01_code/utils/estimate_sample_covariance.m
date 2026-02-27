% w 是 N x L 的噪声矩阵，返回样本协方差矩阵
function Chat = estimate_sample_covariance(w)
    [~, L] = size(w);
    w_mean = mean(w);              % 按列计算样本均值
    w_centered = w - w_mean;          % 每列减去均值
    Chat = (w_centered * w_centered') / L;  % 样本协方差矩阵
end