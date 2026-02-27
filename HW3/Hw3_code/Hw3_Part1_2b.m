%%%%%%%%%%% Hw3 Part1 2b)  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  GooooD   %%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
load('GoogleDataset.mat');   % Load dataset

X = Click;  % Choose KPI type (Cost / Click / Conversion)
% X = Conversion;  % Choose KPI type (Cost / Click / Conversion)

threshold = 0.29;  % Similarity threshold

similar_index_list = get_similar_companies_and_plot(X, threshold);

% Display similar company indices for selected companies
disp(similar_index_list{1});    % Similar companies for company 1
disp(similar_index_list{20});   % Similar companies for company 20

[mse_list, all_pred] = inst_predict_all_companies(X, threshold, true);

% Visualize MSE bar chart
figure;
bar(mse_list);
title('Prediction Error (MSE) for Each Company');
xlabel('Company Index'); ylabel('MSE');


%%%%%%%%%%% Hw3 Part1 2(c) – Estimate Ad TK using Error Jump %%%%%%%%%%%%%%%%%%%%%

fprintf('\n========= Part 1 - 2(c): Estimate TK using error jumps =========\n');

% -------- 使用 all_pred（2b已生成）与真实数据 X --------
[N, K] = size(X);
estimated_TK = nan(1, K);        % 存储每家公司的估计广告时间
true_TK_20 = 397;                % 公司20的真实广告时间，仅用于验证

figure('Name', 'Part 2(c) – Replot with Estimated TK');  % 新图，重新绘制全部曲线 + TK线

for i = 1:K
    y_true = X(:, i);
    y_pred = all_pred(:, i);     % 注意这里直接复用了 2b 的预测输出

    % 如果无预测结果（2b已跳过该公司），就跳过
    if all(isnan(y_pred))
        fprintf('公司 %d 无预测数据，跳过。\n', i);
        continue;
    end

    % --- 计算误差并检测误差突变点（广告估计时间） ---
    error_seq = y_true - y_pred;
    try
        change_pt = findchangepts(error_seq, 'Statistic', 'mean', 'MaxNumChanges', 1);
        estimated_TK(i) = change_pt;
    catch
        fprintf('❌ 公司 %2d 找不到误差跳变点。\n', i);
        continue;
    end

    % --- 新建图像重新绘制 prediction 曲线，并叠加 TK 估计 ---
    subplot(4, 5, i);
    plot(y_true, 'b', 'LineWidth', 1.2); hold on;
    plot(y_pred, 'r--', 'LineWidth', 1.2);
    xline(change_pt, 'k--', 'LineWidth', 1.4);  % 黑色虚线标出广告时间点
    title(sprintf('公司 %d', i), 'FontSize', 9);
    axis tight; grid on;
    set(gca, 'xtick', [], 'ytick', []);
end

% -------- 输出所有估计 TK --------
fprintf('\n📋 估计的广告启动时间 T_K（按公司）：\n');
for i = 1:K
    if ~isnan(estimated_TK(i))
        fprintf('公司 %2d → 估计 T_K = 第 %d 天\n', i, estimated_TK(i));
    end
end

fprintf('\n✅ 公司 20: 真实广告时间 = %d, 估计广告时间 = %d, 误差 = %d 天\n', ...
    true_TK_20, estimated_TK(20), abs(true_TK_20 - estimated_TK(20)));



%%%%%%%%%%%%%%%%%%%%%%%%%   Function  %%%%%%%%%%%%%%%%%%%%%
function similar_index_list = get_similar_companies_and_plot(X, threshold)
% Function: Compute and visualize the absolute correlation matrix between companies
%           Also returns the index list of similar companies for each one
% Inputs:
%   X         : N x K KPI matrix (rows: days, columns: companies)
%   threshold : Similarity threshold (if |correlation| ≥ threshold, consider similar)
%
% Output:
%   similar_index_list : 1 x K cell array, each cell contains indices of similar companies (excluding itself)

% --- Step 1: Z-score normalization ---
X_z = zscore(X);              % N x K, normalize each column
corr_matrix = corr(X_z);      % K x K correlation matrix

K = size(X, 2);
abs_corr_matrix = abs(corr_matrix);  % Absolute values for similarity
similar_index_list = cell(1, K);     % Container for results

% --- Step 2: Identify similar companies for each target ---
for i = 1:K
    sim_idx = find(abs_corr_matrix(i, :) >= threshold & (1:K) ~= i);
    similar_index_list{i} = sim_idx;
end

% --- Step 3: Plot heatmap of correlations with annotations ---
figure;
imagesc(abs_corr_matrix);
colormap(jet);
colorbar;
caxis([0 1]);
title('Heatmap of Company Similarities (|Correlation Coefficient|)', 'FontSize', 14);
xlabel('Company Index'); ylabel('Company Index');
axis square;
xticks(1:K); yticks(1:K);

% Annotate heatmap with values
for i = 1:K
    for j = 1:K
        val = abs_corr_matrix(i, j);
        text(j, i, sprintf('%.2f', val), ...
            'FontSize', 9, 'HorizontalAlignment', 'center', ...
            'Color', 'w', 'FontWeight', 'bold');
    end
end
end

%%%%%%%%%%% Function for Prediction in Part 2b %%%%%
function [mse_list, all_pred] = inst_predict_all_companies(X, threshold, do_plot)
% Function: For each company, predict KPI using similar companies' instantaneous data
% Inputs:
%   X         : N x K KPI matrix
%   threshold : Similarity threshold (based on absolute correlation)
%   do_plot   : Whether to plot individual predictions (true/false)
%
% Outputs:
%   mse_list  : 1 x K vector of prediction errors for each company
%   all_pred  : N x K matrix, each column is predicted KPI for a company

if nargin < 3
    do_plot = false;
end

[N, K] = size(X);
X_z = zscore(X);               % Normalize columns
corr_matrix = corr(X_z);
abs_corr = abs(corr_matrix);

mse_list = zeros(1, K);
all_pred = nan(N, K);

% Loop over companies
for target_col = 1:K
    % 1. Find similar companies (exclude self)
    sim_idx = find(abs_corr(target_col, :) >= threshold & (1:K) ~= target_col);
    
    if isempty(sim_idx)
        fprintf('Company %d has no similar companies. Skipping.\n', target_col);
        continue;
    end
    
    % 2. Prepare training data
    Y = X(:, target_col);     % Target KPI
    X_sim = X(:, sim_idx);    % Input from similar companies
    
    % 3. Linear least squares fit
    alpha = (X_sim' * X_sim) \ (X_sim' * Y);   % Coefficient vector
    
    % 4. Predict and compute error
    Y_hat = X_sim * alpha;
    all_pred(:, target_col) = Y_hat;
    mse_list(target_col) = mean((Y_hat - Y).^2);
    
    % 5. Optional visualization
    if do_plot && target_col == 1
        figure('Name', 'All Company Prediction Results');
    end

    if do_plot
        subplot(4, 5, target_col);  % 4 rows × 5 columns for up to 20 companies
        plot(Y, 'b', 'LineWidth', 1.2); hold on;
        plot(Y_hat, 'r--', 'LineWidth', 1.2);
        title(sprintf('Company %d', target_col), 'FontSize', 9);
        axis tight;
        grid on;
        set(gca, 'xtick', [], 'ytick', []);
    end
end
end
