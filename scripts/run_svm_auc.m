function AUC = run_svm_auc(X, y, label)

% 10-fold CV by default (5 for FA/CR if needed — handled upstream)
if strcmp(label, 'FAvsCR')
    k = 5;
else
    k = 10;
end

cv = cvpartition(y, 'KFold', k);
AUCs = zeros(k, 1);

for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    model = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', 0.5, ...
        'Standardize', true);

    [~, scores] = predict(model, X(testIdx,:));
    posClass = 1;  % assume 1 = positive

    [~, ~, ~, AUCs(i)] = perfcurve(y(testIdx), scores(:,2), posClass);
end

AUC = mean(AUCs);
fprintf('%s AUC: %.3f\n', label, AUC);

end