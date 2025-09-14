function auc = run_lda_auc(X, y, taskName)
    % Match Sucheta's reproducibility setup
    rng(0, 'twister');  % Mersenne Twister with fixed seed

    % Choose fold count
    switch taskName
        case 'FAvsCR'
            nFolds = 5;
        otherwise
            nFolds = 10;
    end

    % Stratified cross-validation
    cv = cvpartition(y, 'KFold', nFolds);
    auc_folds = nan(nFolds, 1);   % <-- NaN so we can skip folds

    for k = 1:nFolds
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test  = X(testIdx, :);
        y_test  = y(testIdx);

        % Skip (don't drop participant) if a fold is single-class
        if numel(unique(y_test)) < 2 || numel(unique(y_train)) < 2
            warning('Fold %d skipped: missing class in train/test.', k);
            continue
        end

        % Apply SMOTE only to training set (unchanged)
        [X_train_bal, y_train_bal] = applySMOTE(X_train, y_train);

        % Train LDA with regularization (unchanged)
        model = fitcdiscr(X_train_bal, y_train_bal, ...
            'ClassNames', [0; 1], 'Gamma', 0.5, ...
            'FillCoeffs', 'off', 'SaveMemory', 'on');

        % Predict on test set (unchanged)
        [~, scores] = predict(model, X_test);
        [~, ~, ~, auc_folds(k)] = perfcurve(y_test, scores(:,2), 1);
    end

    % Final AUC = mean across valid folds only
    if all(isnan(auc_folds))
        warning('All folds invalid. Returning NaN.');
        auc = NaN;
    else
        auc = mean(auc_folds(~isnan(auc_folds)));
    end
end
