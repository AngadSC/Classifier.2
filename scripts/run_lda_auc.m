function auc = run_lda_auc(X, y, taskName, participant)
    % Match Sucheta's reproducibility setup
    rng(0, 'twister'); % Mersenne Twister with fixed seed
    y = y(:);

    % -------------------------------
    % 1) Drop participants w/ 1 class
    % -------------------------------
    if nargin >= 4 && ~isempty(participant)
        pid  = participant(:);
        uids = unique(pid);
        keep_uid = false(numel(uids),1);
        for ii = 1:numel(uids)
            yi = y(pid == uids(ii));
            keep_uid(ii) = numel(unique(yi)) >= 2; % must contain both classes
        end
        keep_mask = ismember(pid, uids(keep_uid));
        X = X(keep_mask, :);
        y = y(keep_mask);
        pid = pid(keep_mask); % keep filtered participant IDs

        if numel(unique(y)) < 2
            warning('All trials reduce to one class after participant filtering for %s.', taskName);
            auc = NaN; 
            return;
        end
    else
        pid = []; % not provided
    end

    % -----------------------
    % 2) Choose fold count
    % -----------------------
    switch taskName
        case 'FAvsCR'
            nFolds = 5;
        otherwise
            nFolds = 10;
    end

    % ---------------------------------------------------------
    % 3) (Participant path) Remove participants causing bad CV
    % ---------------------------------------------------------
    if ~isempty(pid)
        % Clamp folds to smallest class count to allow stratification
        classes = unique(y);
        classCounts = arrayfun(@(c) sum(y==c), classes);
        nFolds = min(nFolds, max(2, min(classCounts)));

        % Initial stratified CV to detect problematic participants
        cv_check = cvpartition(y, 'KFold', nFolds);
        problematic_participants = [];

        uy = unique(y);
        for k = 1:nFolds
            trainIdx = training(cv_check, k);
            testIdx  = test(cv_check, k);

            ytr = y(trainIdx);
            yte = y(testIdx);

            if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2
                % Score participants in this fold and mark the worst offenders
                for uid = unique(pid)'
                    % Remove this participant from train/test in this fold and see if it fixes it
                    tr_mask = trainIdx & (pid == uid);
                    te_mask = testIdx  & (pid == uid);

                    if any(tr_mask) || any(te_mask)
                        ytr_tmp = y(trainIdx & (pid ~= uid));
                        yte_tmp = y(testIdx  & (pid ~= uid));
                        if numel(unique(ytr_tmp)) >= 2 && numel(unique(yte_tmp)) >= 2
                            problematic_participants(end+1,1) = uid; %#ok<AGROW>
                        end
                    end
                end
            end
        end

        if ~isempty(problematic_participants)
            problematic_participants = unique(problematic_participants);
            keep_mask = ~ismember(pid, problematic_participants);
            X = X(keep_mask, :);
            y = y(keep_mask);
            pid = pid(keep_mask);  % <<< IMPORTANT: keep pid in sync

            fprintf('Removed %d participant(s) that caused single-class folds in %s.\n', ...
                    numel(problematic_participants), taskName);

            if numel(unique(y)) < 2
                warning('All trials reduce to one class after removing problematic participants for %s.', taskName);
                auc = NaN; 
                return;
            end
        end
    end

    % -----------------------------------------------
    % 4) Final stratified CV on cleaned dataset
    %    (clamp nFolds again after any pruning)
    % -----------------------------------------------
    classes = unique(y);
    classCounts = arrayfun(@(c) sum(y==c), classes);
    nFolds = min(nFolds, max(2, min(classCounts)));

    cv = cvpartition(y, 'KFold', nFolds);
    auc_folds = nan(nFolds, 1);

    for k = 1:nFolds
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test  = X(testIdx, :);
        y_test  = y(testIdx);

        % Safety: should not happen after participant pruning + clamping
        if numel(unique(y_train)) < 2 || numel(unique(y_test)) < 2
            warning('Unexpected single-class fold after participant filtering in fold %d. Returning NaN.', k);
            auc = NaN; 
            return;
        end

        % SMOTE on training set (unchanged)
        [X_train_bal, y_train_bal] = applySMOTE(X_train, y_train);

        % Train LDA (unchanged)
        model = fitcdiscr(X_train_bal, y_train_bal, ...
            'ClassNames', [0; 1], 'Gamma', 0.5, ...
            'FillCoeffs', 'off', 'SaveMemory', 'on');

        % Predict & AUC
        [~, scores] = predict(model, X_test);
        [~, ~, ~, auc_folds(k)] = perfcurve(y_test, scores(:,2), 1);
    end

    % Mean across folds
    auc = mean(auc_folds);
end
