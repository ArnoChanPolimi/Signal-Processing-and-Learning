%%%%  生成 N x N 的协方差矩阵  %%%%
function Cww = generate_covariance_matrix(N, sigma_w, rho)
    Cww = zeros(N, N);
    for i = 1:N
        for j = 1:N
            Cww(i, j) = sigma_w^2 * rho^abs(i - j);
        end
    end
end