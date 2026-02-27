%% %%%%% ------ SP&L Homework3 -------- %%%%%%%%%%
clear all;
load('GoogleDataset.mat'); %

%% %%%%%%%%%%%% --- figure ---- %%%%%%%%%%%
figure(1); % 画出第20家公司的KPI

plot(Cost(:,20), 'color', 'r', 'LineWidth', 2, 'DisplayName', 'Cost'); % 红色曲线
hold on;
plot(Click(:,20), 'color', 'b', 'LineWidth', 1.5, 'DisplayName', 'Click'); % 蓝色曲线
hold on;
plot(Conversion(:,20), 'color', 'g', 'LineWidth', 1.5, 'DisplayName', 'Conversion'); % 绿色曲线
legend; % 添加图例
title('KPI Plot K = 20'); % 添加标题
xlabel('Index'); %
ylabel('Value'); %


%% %%% --- 判断同类公司 ----- %%%% 
rho_cost = abs(calculateCrossCorrelation(Cost(1:T20, :)));
rho_click = abs(calculateCrossCorrelation(Click(1:T20, :)));
rho_conversion = abs(calculateCrossCorrelation(Conversion(1:T20, :)));

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