%%%%%%%%%%% Hw3 Part1 3)_repeat 2b - Using All KPIs for Similarity %%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
load('GoogleDataset.mat');   % Load dataset

% All KPIs
[N, K] = size(Click);  % N: number of days, K: number of companies

% ----- 拼接所有 KPI 并构造公司特征向量 -----
all_vec = zeros(3 * N, K);  % 每家公司是一个3N维向量
for i = 1:K
    V_i = [Click(:, i); Cost(:, i); Conversion(:, i)];  % 3N x 1
    all_vec(:, i) = zscore(V_i);  % 标准化
end

% ----- 计算公司之间的相关性 -----
corr_mat = abs(corr(all_vec));  % K x K 相似度矩阵（取绝对值）
threshold = 0.29;               % 相似性阈值

% ----- 构造每家公司对应的相似公司索引 -----
similar_index_list = cell(1, K);
for i = 1:K
    sim_idx = find(corr_mat(i, :) >= threshold & (1:K) ~= i);
    similar_index_list{i} = sim_idx;
end

% ----- 可视化热图 -----
figure;
imagesc(corr_mat);
colormap(jet);
colorbar;
caxis([0 1]);
title('Heatmap of Company Similarities using All KPIs', 'FontSize', 14);
xlabel('Company Index'); ylabel('Company Index');
axis square; xticks(1:K); yticks(1:K);

for i = 1:K
    for j = 1:K
        text(j, i, sprintf('%.2f', corr_mat(i, j)), ...
            'FontSize', 8, 'HorizontalAlignment', 'center', ...
            'Color', 'w', 'FontWeight', 'bold');
    end
end

% ----- 使用相似公司进行点击量预测 -----
X_click = Click;
mse_list = zeros(1, K);
all_pred = nan(N, K);

figure('Name', '2d - All KPI Similarity Prediction on Click');

for i = 1:K
    sim_idx = similar_index_list{i};
    if isempty(sim_idx)
        fprintf('公司 %d 无相似公司，跳过。\n', i);
        continue;
    end

    Y = X_click(:, i);          % 当前目标公司点击量
    X_sim = X_click(:, sim_idx); % 相似公司的点击量

    alpha = (X_sim' * X_sim) \ (X_sim' * Y);  % 最小二乘系数
    Y_hat = X_sim * alpha;                    % 预测值
    all_pred(:, i) = Y_hat;
    mse_list(i) = mean((Y_hat - Y).^2);

    subplot(4, 5, i);
    plot(Y, 'b', 'LineWidth', 1.2); hold on;
    plot(Y_hat, 'r--', 'LineWidth', 1.2);
    title(sprintf('Company %d', i), 'FontSize', 9);
    xlabel('Days'); ylabel('KPI');
    legend('True', 'Predicted');
    axis tight; grid on;
    set(gca, 'xtick', [], 'ytick', []);
end

% ----- 可视化MSE -----
figure;
bar(mse_list);
xlabel('Company Index'); ylabel('MSE');
title('Prediction Error (Click) using All KPI Similarity');
grid on;
