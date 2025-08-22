function [X_smote, y_smote] = applySMOTE(X, y, k)
% Simple SMOTE implementation for binary classes
% Inputs:
%   X = [n_samples × n_features] matrix
%   y = [n_samples × 1] labels (must be binary: 0 and 1)
%   k = number of neighbors (default = 4)
% Output:
%   X_smote, y_smote = balanced dataset

if nargin < 3
    k = 4;
end
y = double(y); 
classes = unique(y);
if length(classes) ~= 2
    error('SMOTE requires binary classification labels.');
end

% Ensure y is a column vector and numeric
y = double(y(:));

% Determine class counts
class_counts = arrayfun(@(c) sum(y == c), classes);
[~, minIdx] = min(class_counts);
minority_class = classes(minIdx);

% Split data by class
X_min = X(y == minority_class, :);
X_maj = X(y ~= minority_class, :);
N_needed = size(X_maj, 1) - size(X_min, 1);

% Find k-nearest neighbors
n_min = size(X_min, 1);  % number of minority-class samples
k_safe = min(k, n_min - 1);  % ensure we don’t ask for more neighbors than exist

if k_safe < 1
    warning('Not enough minority samples to apply SMOTE safely. Skipping SMOTE.');
    X_smote = X;
    y_smote = y;
    return;
end

% Calculate safe k based on sample count
n_min = size(X_min, 1);
k_safe = min(k, n_min - 1);  % Must have at least k+1 for knnsearch

if k_safe < 1
    warning('SMOTE skipped: not enough minority class samples (%d)', n_min);
    X_smote = X;
    y_smote = y;
    return;
end

% Find neighbors
idx = knnsearch(X_min, X_min, 'K', k_safe + 1);
idx = idx(:, 2:end);  % remove self-match

% Generate synthetic points
X_synthetic = zeros(N_needed, size(X,2));
for i = 1:N_needed
    sample_idx = randi(n_min);
    neighbor_idx = idx(sample_idx, randi(k_safe));  % <-- uses safe k
    diff = X_min(neighbor_idx,:) - X_min(sample_idx,:);
    gap = rand(1, size(X,2));
    X_synthetic(i,:) = X_min(sample_idx,:) + gap .* diff;
end

% Generate synthetic samples
X_synthetic = zeros(N_needed, size(X,2));
rng(1); % for reproducibility
for i = 1:N_needed
    sample_idx = randi(n_min);
    neighbor_idx = idx(sample_idx, randi(k_safe));  
    diff = X_min(neighbor_idx,:) - X_min(sample_idx,:);
    gap = rand(1, size(X,2));
    X_synthetic(i,:) = X_min(sample_idx,:) + gap .* diff;
end

X_smote = [X; X_synthetic];
y_smote = [y; repmat(minority_class, N_needed, 1)];
end