function AUC = run_svm_auc(X, y, label, participant)
% run_svm_auc  —  SVM AUC with participant dropping (no fold dropping)
% Drops participants that would cause single-class issues; if still
% infeasible (e.g., within-subject FA/CR too lopsided), returns NaN for
% that participant instead of erroring.

    % ------------------ Fold count (default) ------------------
    if strcmp(label, 'FAvsCR')
        k = 5;
    else
        k = 10;
    end

    % ------------------ Participant pruning ------------------
    if nargin >= 4 && ~isempty(participant)
        pid  = participant(:);
        uids = unique(pid);

        % Keep only participants that have BOTH classes
        keep_uid = false(numel(uids),1);
        for ii = 1:numel(uids)
            yi = y(pid == uids(ii));
            keep_uid(ii) = numel(unique(yi)) >= 2;
        end
        keep_mask = ismember(pid, uids(keep_uid));
        X = X(keep_mask, :);
        y = y(keep_mask);
        pid = pid(keep_mask);

        if numel(unique(y)) < 2
            warning('All trials reduce to one class after participant filtering for %s. Returning NaN.', label);
            AUC = NaN; 
            return;
        end

        % Pre-screen participants that cause single-class folds
        classes = unique(y);
        classCounts = histcounts(y, [classes; max(classes)+1]);
        if min(classCounts) < k
            cv_check = cvpartition(length(y), 'KFold', k);  % non-stratified
        else
            cv_check = cvpartition(y, 'KFold', k);          % stratified
        end

        problematic_participants = [];

        for i = 1:k
            trainIdx = training(cv_check, i);
            testIdx  = test(cv_check, i);

            if length(unique(y(trainIdx))) < 2 || length(unique(y(testIdx))) < 2
                for uid = unique(pid)'  % candidate offenders in this fold
                    in_train = trainIdx & (pid == uid);
                    in_test  = testIdx  & (pid == uid);
                    if any(in_train) || any(in_test)
                        ytr_tmp = y(trainIdx & (pid ~= uid));
                        yte_tmp = y(testIdx  & (pid ~= uid));
                        if length(unique(ytr_tmp)) >= 2 && length(unique(yte_tmp)) >= 2
                            problematic_participants = [problematic_participants, uid]; %#ok<AGROW>
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
            pid = pid(keep_mask);  % keep pid aligned

            fprintf('Removed %d participants that caused single-class folds in %s.\n', ...
                    numel(problematic_participants), label);

            if numel(unique(y)) < 2
                warning('All trials reduce to one class after removing problematic participants for %s. Returning NaN.', label);
                AUC = NaN; 
                return;
            end
        end
    else
        pid = []; % no participant info provided
    end

    % ------------------ Final CV setup ------------------
    classes = unique(y);
    classCounts = histcounts(y, [classes; max(classes)+1]);

    % If any class has < 2 trials, we cannot form valid folds -> drop participant
    if min(classCounts) < 2
        warning('Insufficient per-class trials (%d) for %s; returning NaN.', min(classCounts), label);
        AUC = NaN; 
        return;
    end

    % Clamp k to smallest class count (and at least 2) to avoid impossible splits
    k = min(k, max(2, min(classCounts)));

    % Stratified if feasible, else non-stratified
    if min(classCounts) < k
        cv = cvpartition(length(y), 'KFold', k);  % non-stratified
    else
        cv = cvpartition(y, 'KFold', k);          % stratified
    end

    % ------------------ Cross-validation ------------------
    AUCs = nan(k, 1);

    for i = 1:k
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        % If a fold still becomes single-class, drop this participant gracefully
        if length(unique(y(trainIdx))) < 2 || length(unique(y(testIdx))) < 2
            warning('Single-class fold after participant filtering in fold %d for %s. Returning NaN.', i, label);
            AUC = NaN;
            return;
        end

        % Train set (with SMOTE)
        X_train = X(trainIdx,:);
        y_train = y(trainIdx);
        try
            [X_train, y_train] = applySMOTE(X_train, y_train, 4);
        catch
            % If SMOTE fails, proceed with original training data
        end

        % Train linear SVM
        model = fitcsvm(X_train, y_train, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', 0.5, ...
            'Standardize', true);

        % Predict on test set and compute AUC
        [~, scores] = predict(model, X(testIdx,:));
        [~, ~, ~, auc] = perfcurve(y(testIdx), scores(:,2), 1);

        AUCs(i) = auc;
    end

    % ------------------ Aggregate ------------------
    AUC = mean(AUCs, 'omitnan');
    fprintf('%s AUC: %.3f (based on all %d folds)\n', label, AUC, k);
end
