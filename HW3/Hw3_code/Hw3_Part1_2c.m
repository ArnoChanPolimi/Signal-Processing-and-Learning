% %%%%%%%%%%%%%%% Part1 - 2(c) Advertising Effect Analysis %%%%%%%%%%%%%%%
% clc; clear; close all;
% load('GoogleDataset.mat');
% X = Click;           % Select KPI type (here using click volume)
% target_col = 20;     % Company 20 is the advertising subject
% threshold = 0.3;     % Similarity threshold for similar companies
% T_cut = 397;         % Ad launch time (cut-off day)
% 
% % ---------- Step 1: Identify similar companies ----------
% X_z = zscore(X);
% corr_mat = corr(X_z);
% abs_corr = abs(corr_mat);
% sim_idx = find(abs_corr(target_col, :) >= threshold & (1:20) ~= target_col);
% 
% % ---------- Step 2: Fit a model using similar companies (pre-ad period) ----------
% X_sim = X(1:T_cut, sim_idx);            % N x P matrix of similar companies
% Y = X(1:T_cut, target_col);             % N x 1 actual clicks for target company
% 
% alpha = (X_sim' * X_sim) \ (X_sim' * Y);  % Least squares estimate of coefficients
% 
% % ---------- Step 3: Predict the full period using similar companies ----------
% X_sim_all = X(:, sim_idx);                % Full-period input
% Y_hat = X_sim_all * alpha;                % Predicted clicks
% Y_hat = max(0, Y_hat);                    % Enforce non-negativity
% 
% % ---------- Step 4: Visualize error curve ----------
% error_val = X(:, target_col) - Y_hat;
% figure;
% plot(error_val, 'm', 'LineWidth', 1.3);
% xline(T_cut, 'k--', 'Ad Launch', 'LabelVerticalAlignment','bottom');
% title('Click Volume Error Trend (True - Predicted)');
% xlabel('Time (days)'); ylabel('Prediction Error');
% 
% % ---------- Step 5: Visualize actual vs predicted click volume ----------
% figure;
% plot(X(:, target_col), 'b', 'LineWidth', 1.5); hold on;
% plot(Y_hat, 'r--', 'LineWidth', 1.5);
% xline(T_cut, 'k--', 'Ad Launch', 'LabelVerticalAlignment','bottom');
% legend('Actual Clicks', 'Predicted Clicks');
% title('Company 20: Actual vs Predicted Click Volume (Ad Impact Analysis)');
% xlabel('Time (days)'); ylabel('Click Volume');
% ylim([0, max(X(:, target_col)) + 1000]);
% grid on;

