function AUC = run_svm_auc(X, y, label, participant)
% 10-fold CV by default (5 for FA/CR if needed — handled upstream)
if strcmp(label, 'FAvsCR')
    k = 5;
else
    k = 10;
end

% --- drop participants that make the data single-class ---
if nargin >= 4 && ~isempty(participant)
    pid  = participant(:);
    uids = unique(pid);
    keep_uid = false(numel(uids),1);
    for ii = 1:numel(uids)
        yi = y(pid == uids(ii));
        keep_uid(ii) = numel(unique(yi)) >= 2;  % must have both classes
    end
    keep_mask = ismember(pid, uids(keep_uid));
    X = X(keep_mask, :);
    y = y(keep_mask);

    if numel(unique(y)) < 2
        warning('All trials reduce to one class after participant filtering for %s.', label);
        AUC = NaN; return;
    end
end


% Use non-stratified CV if classes are severely imbalanced
classes = unique(y);
classCounts = histcounts(y, [classes; max(classes)+1]);
if min(classCounts) < k
    cv = cvpartition(length(y), 'KFold', k);  % Non-stratified
else
    cv = cvpartition(y, 'KFold', k);          % Stratified
end

AUCs = zeros(k, 1);
validFolds = 0;

for i = 1:k
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);
    
    % Check if BOTH train and test folds have both classes
    if length(unique(y(trainIdx))) < 2 || length(unique(y(testIdx))) < 2
        fprintf('Fold %d: Missing classes in train/test, skipping\n', i);
        continue;
    end
    
    % Apply SMOTE to balance training data
    X_train = X(trainIdx,:);
    y_train = y(trainIdx);
    try
        [X_train, y_train] = applySMOTE(X_train, y_train, 4);
    catch
        % If SMOTE fails, use original data
    end
    
    model = fitcsvm(X_train, y_train, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', 0.5, ...
        'Standardize', true);
    
    [~, scores] = predict(model, X(testIdx,:));
    posClass = 1; % assume 1 = positive
    
    [~, ~, ~, auc] = perfcurve(y(testIdx), scores(:,2), posClass);
    validFolds = validFolds + 1;
    AUCs(validFolds) = auc;
end

if validFolds == 0
    warning('No valid folds for %s', label);
    AUC = NaN;
else
    AUC = mean(AUCs(1:validFolds));
    fprintf('%s AUC: %.3f (based on %d folds)\n', label, AUC, validFolds);
end
end
