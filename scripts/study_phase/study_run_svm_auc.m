function AUC = study_run_svm_auc(X, y, participant, label)
% run_svm_auc_study  —  SVM AUC for STUDY phase (remembered vs forgotten)
%
% Usage:
%   AUC = run_svm_auc_study(X, y, participantIDs);
%   AUC = run_svm_auc_study(X, y, participantIDs, 'StudyMem');
%
% X : trials x features (STUDY features)
% y : trials x 1, 0 = forgotten, 1 = remembered
% participant : trials x 1 participant IDs (numeric or categorical)
% label : optional name used in printed output (default 'StudyMem')

    % ------------------ Defaults ------------------
    if nargin < 4 || isempty(label)
        label = 'StudyMem';
    end

    % Base number of folds (will be clamped later)
    k = 10;

    % ------------------ Basic quality check ------------------
    y = y(:);
    if numel(unique(y)) < 2
        warning('run_svm_auc_study:OnlyOneClass', ...
            'Only one class present for %s (no remembered/forgotten separation). Returning NaN.', label);
        AUC = NaN;
        return;
    end

    % Minimum trials per class (tunable; here = 10)
    MIN_TRIALS_PER_CLASS = 10;
    classes      = unique(y);
    classCounts  = histcounts(y, [classes; max(classes)+1]);
    minClassN    = min(classCounts);

    if minClassN < MIN_TRIALS_PER_CLASS
        warning('run_svm_auc_study:TooFewTrials', ...
            'Insufficient trials per class (%d < %d) for %s. Returning NaN.', ...
            minClassN, MIN_TRIALS_PER_CLASS, label);
        AUC = NaN;
        return;
    end

    % ------------------ Participant pruning ------------------
    if nargin >= 3 && ~isempty(participant)
        pid  = participant(:);
        uids = unique(pid);

        % Drop participants that have only one class
        keep_uid = false(numel(uids),1);
        for ii = 1:numel(uids)
            yi = y(pid == uids(ii));
            keep_uid(ii) = numel(unique(yi)) >= 2;
        end
        keep_mask = ismember(pid, uids(keep_uid));
        X   = X(keep_mask, :);
        y   = y(keep_mask);
        pid = pid(keep_mask);

        if numel(unique(y)) < 2
            warning('run_svm_auc_study:OnlyOneClassAfterPrune', ...
                'All trials reduce to one class after participant filtering for %s. Returning NaN.', label);
            AUC = NaN;
            return;
        end

        % Pre-screen participants that cause single-class folds
        classes     = unique(y);
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
            X   = X(keep_mask, :);
            y   = y(keep_mask);
            pid = pid(keep_mask);  % keep pid aligned

            fprintf('Removed %d participants that caused single-class folds in %s.\n', ...
                    numel(problematic_participants), label);

            if numel(unique(y)) < 2
                warning('run_svm_auc_study:OnlyOneClassAfterProblematicRemoved', ...
                    'All trials reduce to one class after removing problematic participants for %s. Returning NaN.', label);
                AUC = NaN;
                return;
            end
        end
    else
        pid = []; %#ok<NASGU>
    end

    % ------------------ Final CV setup ------------------
    classes     = unique(y);
    classCounts = histcounts(y, [classes; max(classes)+1]);

    % If any class has < 2 trials, we cannot form valid folds
    if min(classCounts) < 2
        warning('run_svm_auc_study:TooFewPerClassForCV', ...
            'Insufficient per-class trials (%d) for %s; returning NaN.', min(classCounts), label);
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

        % If a fold still becomes single-class, bail out gracefully
        if length(unique(y(trainIdx))) < 2 || length(unique(y(testIdx))) < 2
            warning('run_svm_auc_study:SingleClassFold', ...
                'Single-class fold in fold %d for %s. Returning NaN.', i, label);
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
    fprintf('%s (study) AUC: %.3f (based on %d folds)\n', label, AUC, k);
end
