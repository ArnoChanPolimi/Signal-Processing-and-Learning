%% %%%%% ------ SP&L Homework3 -------- %%%%%%%%%%
clc; clear; close all;
load('GoogleDataset.mat');  % 加载数据
X = Click(:,20);  % 选择目标 KPI 时间序列
T = length(X);   % 数据总长度524
T20 = 397;       % 广告干预时间点


% P 是estimator的长度, x[n-1],x[n-2]...x[n-p]
% h 是滞后的长度，x[n+h]

%% 2. 线性预测函数
function x_pred = linear_predict(X, P, h)
    T = length(X);
    x_pred = nan(T,1);  % 初始化预测结果
    for t = P:(T-h)
        X_train = X(t-P+1:t);
        Y_train = X(t+h);
        coef =  [X_train', 1] \ Y_train; 
        X_test = [X(t-P+2:t+1)', 1];
        x_pred(t+h) = X_test * coef;
    end
end

% 固定 h=12, x[n+12]
% 变化 P: x[n-1],x[n-2]...x[n-p]
% 得出的结论，h固定时，P=8的时候MSE最小,P: x[n-1],x[n-2]...x[n-8]
h_fixed = 10;
P_values = [3,4,5,6,7,8,9,10,11,12,13,15,17,19,20];  
MSE_P = zeros(length(P_values),1);
RMSE_P = zeros(length(P_values),1);
MAPE_P = zeros(length(P_values),1);
R2_P = zeros(length(P_values),1);

for i = 1:length(P_values)
    P = P_values(i);
    x_pred = linear_predict(X, P, h_fixed);

    valid_idx = ~isnan(x_pred);
    MSE_P(i) = mean((X(valid_idx) - x_pred(valid_idx)).^2);
    RMSE_P(i) = sqrt(MSE_P(i));
    MAPE_P(i) = mean(abs((X(valid_idx) - x_pred(valid_idx)) ./ X(valid_idx))) * 100;
    R2_P(i) = 1 - sum((X(valid_idx) - x_pred(valid_idx)).^2) / sum((X(valid_idx) - mean(X(valid_idx))).^2);
end


% 固定 P=8, x[n-1],x[n-2]...x[n-8]
% 变化 h:   x[n+h]
% 得出的结论，P固定时，h=2/h=12的时候MSE最小
P_fixed = 8;
h_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
MSE_h = zeros(length(h_values),1);
RMSE_h = zeros(length(h_values),1);
MAPE_h = zeros(length(h_values),1);
R2_h = zeros(length(h_values),1);
for j = 1:length(h_values)
    h = h_values(j);
    x_pred = linear_predict(X, P_fixed, h);

    valid_idx = ~isnan(x_pred);
    MSE_h(j) = mean((X(valid_idx) - x_pred(valid_idx)).^2);
    RMSE_h(j) = sqrt(MSE_h(j));
    MAPE_h(j) = mean(abs((X(valid_idx) - x_pred(valid_idx)) ./ X(valid_idx))) * 100;
    R2_h(j) = 1 - sum((X(valid_idx) - x_pred(valid_idx)).^2) / sum((X(valid_idx) - mean(X(valid_idx))).^2);
end






% 第1张图: 固定 h=12, 变化 P
figure;
subplot(2,2,1);
plot(P_values, MSE_P, 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('P'); ylabel('MSE');
title('MSE vs P (h=10)');
grid on;
subplot(2,2,2);
plot(P_values, RMSE_P, 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('P'); ylabel('RMSE');
title('RMSE vs P (h=10)');
grid on;
subplot(2,2,3);
plot(P_values, MAPE_P, 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('P'); ylabel('MAPE (%)');
title('MAPE vs P (h=10)');
grid on;
subplot(2,2,4);
plot(P_values, R2_P, 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('P'); ylabel('R²');
title('R² vs P (h=10)');
grid on;
sgtitle('误差指标随 P 变化 (h=12)');

% 第2张图: 固定 P=8, 变化 h
figure;
subplot(2,2,1);
plot(h_values, MSE_h, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('h'); ylabel('MSE');
title('MSE vs h (P=8)');
grid on;
subplot(2,2,2);
plot(h_values, RMSE_h, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('h'); ylabel('RMSE');
title('RMSE vs h (P=8)');
grid on;
subplot(2,2,3);
plot(h_values, MAPE_h, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('h'); ylabel('MAPE (%)');
title('MAPE vs h (P=8)');
grid on;
subplot(2,2,4);
plot(h_values, R2_h, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('h'); ylabel('R²');
title('R² vs h (P=8)');
grid on;
sgtitle('误差指标随 h 变化 (P=8)');




%得出结论，P=8, x[n-1],x[n-2]...x[n-8]
%得出结论，h=1, x[n+1] 或者 h=12, x[n+12]




%% **3. 计算最佳参数的预测结果**
P_best = 8;
h_values_best = [2, 12];  % 选择最佳 h
pred_results = cell(length(h_values_best), 1);  % 用于存储预测值

for i = 1:length(h_values_best)
    h_best = h_values_best(i);
    x_pred = linear_predict(X, P_best, h_best);
    pred_results{i} = x_pred;  % 存储预测结果
end

%% **4. 绘制最佳参数的预测曲线**
figure;
% 第1张图：h = 2
subplot(2,1,1);
hold on;
plot(X, 'k', 'LineWidth', 1.5, 'DisplayName', '真实值');
plot(pred_results{1}, 'r', 'LineWidth', 1.5, 'DisplayName', '预测值 (P=8, h=2)');
xline(T20, '--g', 'T20 (广告投放)', 'LineWidth', 2);
legend;
title('预测结果 (P=8, h=2)');
xlabel('时间 (天)');
ylabel('KPI');
grid on;
hold off;

% 第2张图：h = 12
subplot(2,1,2);
hold on;
plot(X, 'k', 'LineWidth', 1.5, 'DisplayName', '真实值');
plot(pred_results{2}, 'b', 'LineWidth', 1.5, 'DisplayName', '预测值 (P=8, h=12)');
xline(T20, '--g', 'T20 (广告投放)', 'LineWidth', 2);
legend;
title('预测结果 (P=8, h=12)');
xlabel('时间 (天)');
ylabel('KPI');
grid on;
hold off;

sgtitle('P=8, 不同 h 值的预测结果');