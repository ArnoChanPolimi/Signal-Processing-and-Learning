%%%%  使用协方差矩阵 Cww 和样本数 L 生成相关噪声  %%%% 
function w = generate_correlated_noise_Cww(Cww, L)
  
    N = size(Cww, 1);
    z = randn(N, L);  % 生成标准正态分布噪声
    % A = myCholesky(Cww);  % Cholesky 分解
    A = fun_A(Cww);
    w = A * z;  % 生成相关噪声
end