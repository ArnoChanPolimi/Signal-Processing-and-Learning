%% %%%%%% Hw3-1.2b %%%%%%%
clear; clc; close all;

% 加载数据
load('GoogleDataset.mat');  % 确保数据文件存在

% 选取 Click 指标的控制 KPI（1:19列）
click_KPI = Click(:,1:20);  
num_tracks = size(click_KPI, 2); % 轨迹数量（19个企业）

% **Step 1: 计算 Pearson 相关性矩阵**
corr_matrix = corrcoef(click_KPI);  


% **Step 2: 画相关性热力图**
figure;
imagesc(abs(corr_matrix));  % 画相关性矩阵
colorbar;                  % 添加颜色条
% 自定义颜色映射：从绿色到深红色
cmap = [linspace(0, 1, 64)', linspace(1, 0, 64)', zeros(64, 1)];  % 从绿色到红色
colormap(cmap);            % 设置颜色映射
clim([0 1]);               % 颜色范围 (0 到 1)
title('Click KPI Correlation Matrix');
xlabel('Track Index');
ylabel('Track Index');

% **Step 3: 添加数值标注**
[num_rows, num_cols] = size(corr_matrix);
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, sprintf('%.2f', abs(corr_matrix(i, j))), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'k');
    end
end



% **Step 4: 基于相关性进行聚类**
Z = linkage(1 - corr_matrix, 'average');  % 相关性高=距离近
num_clusters = 2; % 设定聚类数量（可以调整）
clusters = cluster(Z, 'Maxclust', num_clusters);  

% **Step 3: 绘制 Subplot**
figure;
hold on;
for c = 1:num_clusters
    cluster_idx = find(clusters == c);  % 获取当前类别的轨迹索引
    num_in_cluster = length(cluster_idx); % 该类中的轨迹数
    
    for i = 1:num_in_cluster
        subplot(num_clusters, num_in_cluster, (c-1)*num_in_cluster + i);
        plot(click_KPI(:, cluster_idx(i)), 'b'); % 画 Click 轨迹
        title(['Cluster ', num2str(c), ' - Track ', num2str(cluster_idx(i))]);
        xlabel('Time (Days)');
        ylabel('Clicks');
    end
end
hold off;

% **Step 4: 在每个 Cluster 内进行预测**
figure;
hold on;

for c = 1:num_clusters
    cluster_idx = find(clusters == c);  % 获取当前类别的轨迹索引
    num_in_cluster = length(cluster_idx); % 该类中的轨迹数
    
    % 仅处理包含多个 track 的 Cluster
    if num_in_cluster < 2
        continue;
    end
    
    num_rows = ceil(num_in_cluster / 3); % 计算需要的 subplot 行数（每行最多 3 个）

    for i = 1:num_in_cluster
        target_idx = cluster_idx(i);  % 选取当前要预测的 Track
        predictor_idx = cluster_idx;  % 其他 Track 作为预测变量
        predictor_idx(predictor_idx == target_idx) = []; % 去掉目标 Track 自身
        
        % 构建数据集
        X_train = click_KPI(:, predictor_idx);  % 预测变量（同类中的其他 Track）
        y_train = click_KPI(:, target_idx);  % 目标 Track
        
        % 线性回归估计参数
        beta = (X_train' * X_train) \ (X_train' * y_train);  
        y_pred = X_train * beta; % 预测结果
        
        % 计算 MSE
        MSE = mean((y_train - y_pred).^2);
        MSE_values(target_idx) = MSE;  % 记录 MSE
        
        % **Step 5: 画预测结果**
        subplot(num_rows, 3, i);  % 每行最多 3 个子图
        hold on;
        plot(y_train, 'b', 'LineWidth', 1.5);  % 真实值
        plot(y_pred, 'r--', 'LineWidth', 1.5); % 预测值
        title(['Cluster ', num2str(c), ' - Track ', num2str(target_idx), ...
               ' (MSE=', num2str(MSE, '%.2f'), ')']);
        xlabel('Time (Days)');
        ylabel('Clicks');
        legend('Actual', 'Predicted');
        hold off;
    end
end
hold off;