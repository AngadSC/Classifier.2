function auc = run_lda_auc(X, y, taskName)
    % Match Sucheta’s reproducibility setup
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
    auc_folds = zeros(nFolds, 1);

    for k = 1:nFolds
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test  = X(testIdx, :);
        y_test  = y(testIdx);

        % Ensure test set contains both classes
        if numel(unique(y_test)) < 2
            auc = NaN;
            warning('Fold %d is missing a class. Returning NaN.', k);
            return;
        end

        % Apply SMOTE only to training set
        [X_train_bal, y_train_bal] = applySMOTE(X_train, y_train);

        % Train LDA with regularization
        model = fitcdiscr(X_train_bal, y_train_bal, ...
            'ClassNames', [0; 1], 'Gamma', 0.5, ...
            'FillCoeffs', 'off', 'SaveMemory', 'on');

        % Predict on test set
        [~, scores] = predict(model, X_test);
        [~, ~, ~, auc_folds(k)] = perfcurve(y_test, scores(:,2), 1);
    end

    % Final AUC = mean across folds (not repeats)
    auc = mean(auc_folds);
end