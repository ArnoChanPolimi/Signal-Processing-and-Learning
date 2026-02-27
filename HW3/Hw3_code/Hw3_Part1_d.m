%%%%%%%%%%% Hw3 Part1 3) – Use All KPI from Other Companies %%%%%%%%%%%%%%%%

clc; clear; close all;
load('GoogleDataset.mat');  % Load Click, Cost, Conversion
% Dimensions: N x K
[N, K] = size(Click);

% Standardize all three KPI datasets (to avoid scale inconsistencies)
Click_z = zscore(Click);
Cost_z = zscore(Cost);
Conv_z = zscore(Conversion);

Y_true = Click_z;  % Target: standardized Click
% Y_true = Conv_z; 
Y_pred = zeros(N, K);
mse_list = zeros(1, K);
Beta_all = cell(1, K);  % Store regression coefficients for each company

for i = 1:K
    % Construct input: all other companies' Click/Cost/Conversion (3*(K-1) features)
    other_idx = setdiff(1:K, i);

    X_input = [Click_z(:, other_idx), Cost_z(:, other_idx), Conv_z(:, other_idx)];  % N x (3*(K-1))
    Y_target = Click_z(:, i);  % Target is Click of company i
    
    % Least squares solution
    beta_i = (X_input' * X_input) \ (X_input' * Y_target);
    Y_pred(:, i) = X_input * beta_i;
    mse_list(i) = mean((Y_pred(:, i) - Y_target).^2);
    Beta_all{i} = beta_i;
end

% ------------------- Visualization ---------------------
figure('Name', 'Part 1 – 2(d) Prediction Using All KPIs');
for i = 1:K
    subplot(4, 5, i);
    plot(Y_true(:, i), 'b'); hold on;
    plot(Y_pred(:, i), 'r--');
    title(sprintf('Company %d', i), 'FontSize', 9);
    % xlabel(''); ylabel('');
    xlabel('Days'); ylabel('KPI');
    legend('True', 'Predicted');
    set(gca, 'xtick', [], 'ytick', []);
    grid on;
end
sgtitle('Use All KPI (Click, Cost, Conv) to Predict Click');

% MSE bar chart
figure;
bar(mse_list);
xlabel('Company Index'); ylabel('MSE');
title('Part 2(d): MSE of Click Prediction Using All KPIs');
grid on;
