function auc = run_lda_auc(X, y, taskName, participant)
    % Match Sucheta's reproducibility setup
    rng(0, 'twister');  % Mersenne Twister with fixed seed

    %  drop participants who don't have both classes ----
    if nargin >= 4 && ~isempty(participant)
        pid  = participant(:);
        uids = unique(pid);
        keep_uid = false(numel(uids),1);
        for ii = 1:numel(uids)
            yi = y(pid == uids(ii));
            keep_uid(ii) = numel(unique(yi)) >= 2;  % must contain both classes
        end
        keep_mask = ismember(pid, uids(keep_uid));
        X = X(keep_mask, :);
        y = y(keep_mask);

        if numel(unique(y)) < 2
            warning('All trials reduce to one class after participant filtering for %s.', taskName);
            auc = NaN; 
            return;
        end
    end


    % Choose fold count
    switch taskName
        case 'FAvsCR'
            nFolds = 5;
        otherwise
            nFolds = 10;
    end

    % Stratified cross-validation
    cv = cvpartition(y, 'KFold', nFolds);
    auc_folds = nan(nFolds, 1);   % NaN so we can skip folds

    for k = 1:nFolds
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test  = X(testIdx, :);
        y_test  = y(testIdx);

        % Guard: skip if a fold is single-class (should be rare after filtering)
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
