%%%%%%%%%%% Hw3 Part1 2a)  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  GooooD   %%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
load('GoogleDataset.mat');   % Load dataset

X = Click;  % Choose KPI type (Cost / Click / Conversion)
[Y_true, Y_pred, mse_list, Beta_all] = instantaneous_linear_predictor(X);

function [Y_true, Y_pred, mse_list, Beta_all] = instantaneous_linear_predictor(X)
% Function: Use instantaneous linear predictor to forecast KPI for all companies
% Inputs:
%   X        : N x K matrix, each column is a company's KPI over N days
%
% Outputs:
%   Y_true   : N x K matrix of ground truth values
%   Y_pred   : N x K matrix of predicted values
%   mse_list : 1 x K vector of MSE for each company
%   Beta_all : K x (K-1) matrix, each row is the weight vector for a company

[N, K] = size(X);
Y_true = X;              % Ground truth values
Y_pred = zeros(N, K);    % Predicted values
Beta_all = zeros(K, K-1);% Store linear coefficients
mse_list = zeros(1, K);  % Store MSE for each company

for i = 1:K
    % Current target company
    X_target = X(:, i);               
    
    % Use other K-1 companies as input
    idx = setdiff(1:K, i);           
    X_input = X(:, idx);             
    
    % Linear least squares solution: beta_i = (X^T X)^(-1) X^T y
    beta_i = (X_input' * X_input) \ (X_input' * X_target); 
    
    % Store coefficients
    Beta_all(i, :) = beta_i';
    
    % Predict values
    Y_pred(:, i) = X_input * beta_i;
    
    % Compute MSE
    mse_list(i) = mean((Y_pred(:, i) - X_target).^2);
end

% Visualize prediction results
figure;
for i = 1:K
    subplot(4, 5, i);
    plot(Y_true(:, i), 'b'); hold on;
    plot(Y_pred(:, i), 'r--');
    title(sprintf('Company %d', i));
    xlabel('Days'); ylabel('KPI');
    legend('True', 'Predicted');
end
sgtitle('Part 1 - Question 2a: Instantaneous Linear Predictor');

% Visualize MSE
figure;
bar(mse_list);
xlabel('Company Index'); ylabel('MSE');
title('Prediction Error (MSE) per Company');
grid on;

end
