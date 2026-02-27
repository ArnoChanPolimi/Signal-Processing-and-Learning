%% Hw3_Part2-1
clc; clear; close all;

%% 参数设置
N = 800;            % 总数据点数
T0 = 400;           % 真实突变点
rho = 0.8;          % AR(1) 系数
phi = pi/4;         % AR(2) 极点角度
sigma_alpha = 1;    % AR(1) 噪声标准差
sigma_beta = 1;     % AR(2) 噪声标准差
MC = 3000;          % Monte Carlo 实验次数

% 计算 AR(2) 系数
a1 = rho * cos(phi);    
a2 = -rho^2 / 4;        

%% Monte Carlo 仿真
mse_values = zeros(MC, 1); 

for mc = 1:MC
    % 生成 AR(1) 过程 (n ≤ T0)
    x = zeros(N, 1);
    x(1) = sqrt(sigma_alpha) * randn; 
    for n = 2:T0
        x(n) = rho * x(n-1) + sqrt(sigma_alpha) * randn;
    end
    
    % 生成 AR(2) 过程 (n > T0)
    x(T0+1) = sqrt(sigma_beta) * randn; 
    x(T0+2) = sqrt(sigma_beta) * randn; 
    for n = T0+3:N
        x(n) = a1 * x(n-1) + a2 * x(n-2) + sqrt(sigma_beta) * randn;
    end

    %% 计算后验概率 P(T_0 = T_hat | x)
    posterior_prob = zeros(N, 1);
    
    for T_hat = 2:N-2
        % 前段 AR(1) 误差
        if T_hat > 2
            e_AR1 = x(2:T_hat) - rho * x(1:T_hat-1);
            RSS1 = sum(e_AR1.^2);
        else
            RSS1 = inf; 
        end
        
        % 后段 AR(2) 误差
        if (N - (T_hat+2)) >= 1 
            e_AR2 = x(T_hat+3:N) - a1*x(T_hat+2:N-1) - a2*x(T_hat+1:N-2);
            RSS2 = sum(e_AR2.^2);
        else
            RSS2 = inf;
        end
        
        % 计算后验概率 (不考虑先验，直接使用似然)
        posterior_prob(T_hat) = exp(- (RSS1 + RSS2) / 2);
    end
    
    % 归一化后验概率
    posterior_prob = posterior_prob / sum(posterior_prob, 'omitnan');

    % 选择最大后验概率的 T_hat
    [~, T_hat_opt] = max(posterior_prob);
    
    % 记录 MSE
    mse_values(mc) = (T_hat_opt - T0)^2;
end

%% 计算均方误差 (MSE)
MSE = mean(mse_values);
fprintf('Monte Carlo 计算得到的 MSE: %.4f\n', MSE);

%% 绘制最后一次实验的后验概率曲线
figure(1);
plot(2:N-2, posterior_prob(2:N-2), 'b', 'LineWidth', 2); hold on;
xline(T0, '-.r', 'LineWidth', 2, 'Label', '真实 T_0');
xline(T_hat_opt, '--g', 'LineWidth', 2, 'Label', '预测 T_0');
title('后验概率 vs. 可能的突变点 T''');
xlabel('T'' (假设突变点)');
ylabel('P(T_0 = T'' | x)');
legend('后验概率', '真实 T_0', '预测 T_0', 'Location', 'best');
grid on;
