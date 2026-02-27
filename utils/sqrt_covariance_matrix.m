%%%% 计算协方差矩阵的平方根，使得 Cww = A * A^T %%%%
function A = sqrt_covariance_matrix(Cww)
    % 进行特征值分解
    [Q, Lambda] = fun_eig(Cww); % Q是特征向量矩阵，Lambda是特征值矩阵
    A = Q * sqrt(Lambda); % Cww=A * A^T 
end