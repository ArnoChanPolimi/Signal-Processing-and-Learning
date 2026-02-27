% 计算样本协方差矩阵 Chat 和理论协方差矩阵 Cww 之间的 MSE
function mse = compute_mse(Chat, Cww)
    mse = mean((Chat(:) - Cww(:)).^2);
end