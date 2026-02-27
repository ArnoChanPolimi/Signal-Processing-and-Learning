%% %%%%% ------ SP&L Homework3 -------- %%%%%%%%%%
clear all;
load('GoogleDataset.mat'); % 

%% %%%%%% 数据归一化处理 %%%%%%%
days = length(Cost(:,1)); % 数据一共多少天
T20 = 397; % 第20家公司投入广告的日期
x_Cost = zeros(days, 20);
x_Click = zeros(days, 20);
x_Conversion = zeros(days, 20);
for k = 1 : 20
    x_cost_min = min(Cost(:,k));
    x_cost_max = max(Cost(:,k));
    x_Cost(:,k) = (Cost(:,k) - x_cost_min)/(x_cost_max - x_cost_min); % 归一化之后的KPI_Cost
    x_click_min = min(Click(:,k));
    x_click_max = max(Click(:,k));
    x_Click(:,k) = (Click(:,k) - x_click_min)/(x_click_max - x_click_min); % 归一化之后的KPI_Click
    x_conversion_min = min(Conversion(:,k));
    x_conversion_max = max(Conversion(:,k));
    x_Conversion(:,k) = (Conversion(:,k) - x_conversion_min)/(x_conversion_max - x_conversion_min); % 归一化之后的KPI_Conversion
end


% KPI数据矩阵包含三个维度：Cost, Click, Conversion
% 控制 KPI (列 1–19)，目标 KPI (列 20)
%% %%%%%%%%%%%% --- figure ---- %%%%%%%%%%%
figure(1); % 画出第20家公司的KPI

plot(x_Cost(:,20), 'color', 'r', 'LineWidth', 2, 'DisplayName', 'Cost'); % 红色曲线
hold on;
plot(x_Click(:,20), 'color', 'b', 'LineWidth', 1.5, 'DisplayName', 'Click'); % 蓝色曲线
hold on;
plot(x_Conversion(:,20), 'color', 'g', 'LineWidth', 1.5, 'DisplayName', 'Conversion'); % 绿色曲线
legend; % 添加图例
title('KPI Plot K = 20'); % 添加标题
xlabel('Index'); % 添加横坐标标签
ylabel('Value'); % 添加纵坐标标签

%% %%%%% RLS 用前19家公司预测 第20 家公司的 KPI
% 初始化
lambda = 0.98; % 遗忘因子
P = eye(19) * 1e-3; % 初始协方差矩阵（值较大，用于初始学习）
A = zeros(19, 3); % 用于预测第20家公司KPI的权重，前19家公司+3种KPI
X_20_p = zeros(days, 3); % 根据前19家公司预测第20家的数据
alpha = zeros(3, 1); % gk(t) = α*t


error_cost = zeros(days, 1);
error_click = zeros(days, 1);
error_conversion = zeros(days, 1);

%% %%% --- 判断同类公司 ----- %%%% 
rho_cost = abs(calculateCrossCorrelation(x_Cost(1:T20, :)));
rho_click = abs(calculateCrossCorrelation(x_Click(1:T20, :)));
rho_conversion = abs(calculateCrossCorrelation(x_Conversion(1:T20, :)));

figure;
stem(rho_cost, 'MarkerFaceColor', 'b', 'DisplayName', 'ρ_cost');
hold on;
stem(rho_click, 'MarkerFaceColor', 'r', 'DisplayName', 'ρ_click');
hold on;
stem(rho_conversion, 'MarkerFaceColor', 'g', 'DisplayName', 'ρ_conversion');
legend;
title('Correlation with k=20');
xlabel('公司 k');
ylabel('相关系数ρ');
grid on;


%% %%%%% ------ function: 判断前19家公司哪些和第20家公司是同类
function corr_values = calculateCrossCorrelation(X)
    % X is a T x 20 matrix where each column represents a company's KPI time series
    % The 20th column represents the KPI time series of the target company
    
    % days =  length(X(:,1)); % 数据一共多少天;  % Number of time steps
    corr_values = zeros(19, 1);  % Array to store correlation values for the first 19 companies
    
    % Extract the KPI series for the 20th company (track of interest)
    x_target = X(:, 20);
    
    for k = 1:19  % Loop through the first 19 companies
        % Extract the KPI series for the k-th company
        x_k = X(:, k);
        
        % Calculate the mean of both series
        x_k_mean = mean(x_k);
        x_target_mean = mean(x_target);
        
        % Compute the numerator (sum of products of deviations)
        numerator = sum((x_k - x_k_mean) .* (x_target - x_target_mean));
        
        % Compute the denominator (product of standard deviations)
        denominator = sqrt(sum((x_k - x_k_mean).^2) * sum((x_target - x_target_mean).^2));
        
        % Calculate the correlation coefficient
        corr_values(k) = numerator / denominator;
    end
end


%% 对Cost的预测 cost是KPI 1 , i是天数

for i = 1 : T20
    x = x_Cost(i, 1:19)'; % 转为列向量
    X_20_p(i, 1) = dot(A(:, 1), x);
    error_cost(i) = x_Cost(i, 20) - X_20_p(i, 1); % 第20家公司的预测误差

    % 计算增益向量 K
    K = (P * x) / (lambda + x' * P * x);
    
    % 更新权重
    A(:, 1) = A(:, 1) + K * error_cost(i);
    
    % 更新协方差矩阵
    P = (P - K * x' * P) / lambda;
end 
% T20 之后的值
for i = T20 + 1 : days
    x = x_Cost(i, 1:19)'; % 转为列向量
    X_20_p(i, 1) = dot(A(:, 1), x);
    error_cost(i) = x_Cost(i, 20) - X_20_p(i, 1);   
end 
figure;
gk = zeros(days - T20, 3);
gk(:, 1) = error_cost(T20 + 1 : days);
plot(gk(:, 1), 'b', 'DisplayName', 'gk');
legend;
xlabel('days');
ylabel('KPI-Cost');
title('第20家公司 gk');
grid on;
% figure;
% histogram(gk(:, 1)); % 让 MATLAB 自动选择分箱数

%% 对click的预测 click是KPI 2 , i是天数
% 初始化
num_click = 0;
threshold_click = 0.2;
x_click_after = [];
for k = 1 : 19
    if rho_click(k) >= threshold_click
        num_click = num_click + 1;
        x_click_after(:, num_click) = x_Click(:,k);        
    end
end
lambda_click = 0.98; % 遗忘因子
P_click = eye(num_click) * 0.00002; % 初始协方差矩阵（值较大，用于初始学习）
A_click = zeros(num_click, 1); % 用于预测第20家公司KPI的权重，前19家公司+3种KPI

for d = 1 : T20
    x = x_click_after(d, :)'; % 转为列向量
    X_20_p(d, 2) = dot(A_click(:), x);
    error_click(i) = x_Click(i, 20) - X_20_p(i, 2);

    % 计算增益向量 K_click
    K_click = (P_click * x) / (lambda_click + x' * P_click * x);
    
    % 更新权重
    A_click(:) = A_click(:) + K_click * error_click(i);
    
    % 更新协方差矩阵
    P_click = (P_click - K_click * x' * P_click) / lambda_click;
end 
% T20 之后的值
for dd = T20 + 1 : days
    x = x_click_after(dd, :)'; % 转为列向量
    X_20_p(dd, 2) = dot(A_click(:), x);
    error_click(dd) = x_Click(dd, 20) - X_20_p(dd, 2);   
end
figure;
gk(:, 2) = error_click(T20 + 1 : days);
plot(gk(:, 2), 'b', 'DisplayName', 'gk-click');
legend;
xlabel('days');
ylabel('KPI-Click');
title('第20家公司 gk-Click');
grid on;

%% 对conversion的预测 conversion是KPI 3 , i是天数
for i = 1 : T20
    x = x_Conversion(i, 1:19)'; % 转为列向量
    X_20_p(i, 3) = dot(A(:, 3), x);
    error_conversion(i) = x_Conversion(i, 20) - X_20_p(i, 3);

    % 计算增益向量 K
    K = (P * x) / (lambda + x' * P * x);
    
    % 更新权重
    A(:, 3) = A(:, 3) + K * error_conversion(i);
    
    % 更新协方差矩阵
    P = (P - K * x' * P) / lambda;
end 
% T20 之后的值
for i = T20 + 1 : days
    x = x_Conversion(i, 1:19)'; % 转为列向量
    X_20_p(i, 3) = dot(A(:, 3), x);
    error_conversion(i) = x_Conversion(i, 20) - X_20_p(i, 3);   
end


%% %%%%% ------ 对第20家公司KPI gk 的预测 在T20之后
figure;
gk(:, 3) = error_conversion(T20 + 1 : days);
plot(gk(:, 3), 'b', 'DisplayName', 'gk-Conversion');
legend;
xlabel('days');
ylabel('KPI-Conversion');
title('第20家公司 gk-Conversion');
grid on;

gk(:, 2) = error_click(T20 + 1 : days);
figure;
gk(:, 2) = error_conversion(T20 + 1 : days);
plot(gk(:, 2), 'b', 'DisplayName', 'gk-Conversion');
legend;
xlabel('days');
ylabel('KPI-Click');
title('第20家公司 gk-Click');
grid on;
% gk(:, 3) = error_conversion(T20 + 1 : days);
% figure(3);
% plot(gk(:, 1), 'b', 'DisplayName', 'gk');


% 对Click的预测 click是KPI 2 , i是天数


% 对Conversion的预测 conversion是KPI 3 , i是天数



%% %%%% ------ 对第20家公司 KPI 的预测
figure;
plot(x_Cost(:, 20), 'b', 'DisplayName', '归一化 Cost-真实值');
hold on;
plot(X_20_p(:,1), 'r--', 'DisplayName', '归一化 Cost-预测值');
legend;
xlabel('days');
ylabel('KPI-Cost');
title('RLS 对第20家公司 KPI-Cost 的预测');
grid on;

figure;
plot(x_Click(:, 20), 'b', 'DisplayName', 'Click-真实值');
hold on;
plot(X_20_p(:,2), 'r--', 'DisplayName', 'Click-预测值');
legend;
xlabel('days');
ylabel('KPI-Click');
title('RLS 对第20家公司 KPI-Click 的预测');
grid on;

figure;
plot(x_Conversion(:, 20), 'b', 'DisplayName', 'Conversion-真实值');
hold on;
plot(X_20_p(:,3), 'r--', 'DisplayName', 'Conversion-预测值');
legend;
xlabel('days');
ylabel('KPI-Conversion');
title('RLS 对第20家公司 KPI-Conversion 的预测');
grid on;