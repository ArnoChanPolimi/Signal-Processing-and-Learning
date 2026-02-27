% 求Cww的特征向量矩阵和特征值矩阵
function [Q, Lambda] = fun_eig(Cww)
    [Q, Lambda] = eig(Cww); % Q是特征向量矩阵，Lambda是特征值矩阵
end