function AUC = study_run_rvm_auc(X, y, participant, label, kernelType, gamma)
% run_rvm_auc_study — RVM AUC for STUDY phase (remembered vs forgotten)
%
% X : trials x features (STUDY features)
% y : trials x 1, 0 = forgotten, 1 = remembered
% participant : trials x 1 participant IDs (optional, [] if not used)
% label : optional string for printing (default 'StudyMem')
% kernelType : 'gaussian' (default), 'linear', 'polynomial', 'sigmoid', 'laplacian'
% gamma : kernel width/scale (default 0.5)

    % ---------------- Basic setup ----------------
    y = double(y(:));     % RVM library wants doubles
    X = zscore(X);        % feature z-scoring

    if nargin < 4 || isempty(label)
        label = 'StudyMem';
    end
    if nargin < 5 || isempty(kernelType)
        kernelType = 'gaussian';
    end
    if nargin < 6 || isempty(gamma)
        gamma = 0.5;
    end

    % ---------------- Basic label sanity ----------------
    if numel(unique(y)) < 2
        warning('run_rvm_auc_study:OnlyOneClass', ...
            'Only one class present for %s. Returning NaN.', label);
        AUC = NaN;
        return;
    end

    MIN_TRIALS_PER_CLASS = 10;  % tunable
    classes     = unique(y);
    classCounts = histcounts(y, [classes; max(classes)+1]);
    minClassN   = min(classCounts);
    if minClassN < MIN_TRIALS_PER_CLASS
        warning('run_rvm_auc_study:TooFewTrials', ...
            'Insufficient trials per class (%d < %d) for %s. Returning NaN.', ...
            minClassN, MIN_TRIALS_PER_CLASS, label);
        AUC = NaN;
        return;
    end

    % ---------------- Fold count ----------------
    k = 10;  % base folds, will clamp later

    % ---------------- Participant pruning ----------------
    if nargin >= 3 && ~isempty(participant)
        pid  = participant(:);
        uids = unique(pid);

        % Keep only participants that have BOTH classes
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
            warning('run_rvm_auc_study:OnlyOneClassAfterPrune', ...
                'All trials reduce to one class after participant filtering for %s. Returning NaN.', label);
            AUC = NaN;
            return;
        end

        % Pre-screen participants that cause single-class folds
        classes     = unique(y);
        classCounts = histcounts(y, [classes; max(classes)+1]);
        k = min(k, max(2, min(classCounts)));

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
                warning('run_rvm_auc_study:OnlyOneClassAfterProblematicRemoved', ...
                    'All trials reduce to one class after removing problematic participants for %s. Returning NaN.', label);
                AUC = NaN;
                return;
            end
        end
    else
        pid = []; %#ok<NASGU>
    end

    % ---------------- Final CV setup ----------------
    classes     = unique(y);
    classCounts = histcounts(y, [classes; max(classes)+1]);

    if min(classCounts) < 2
        warning('run_rvm_auc_study:TooFewPerClassForCV', ...
            'Insufficient per-class trials (%d) for %s; returning NaN.', min(classCounts), label);
        AUC = NaN;
        return;
    end

    k = min(k, max(2, min(classCounts)));

    if min(classCounts) < k
        cv = cvpartition(length(y), 'KFold', k);  % non-stratified
    else
        cv = cvpartition(y, 'KFold', k);          % stratified
    end

    % ---------------- Cross-validation ----------------
    AUCs = nan(k, 1);

    for i = 1:k
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        if length(unique(y(trainIdx))) < 2 || length(unique(y(testIdx))) < 2
            warning('run_rvm_auc_study:SingleClassFold', ...
                'Single-class fold after participant filtering in fold %d for %s. Returning NaN.', i, label);
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
        y_train = double(y_train(:));

        % ---------- Kernel selection ----------
        switch kernelType
            case 'gaussian'
                kernel = Kernel('type', 'gaussian', 'gamma', gamma);
            case 'linear'
                kernel = Kernel('type', 'linear');
            case 'polynomial'
                kernel = Kernel('type', 'polynomial', 'gamma', gamma, 'offset', 0, 'degree', 2);
            case 'sigmoid'
                kernel = Kernel('type', 'sigmoid', 'gamma', gamma, 'offset', 0);
            case 'laplacian'
                kernel = Kernel('type', 'laplacian', 'gamma', gamma);
            otherwise
                warning('Unknown kernel type: %s. Using gaussian.', kernelType);
                kernel = Kernel('type', 'gaussian', 'gamma', gamma);
        end

        % RVM parameters
        parameter = struct('display', 'off', ...  % quiet for CV
                           'type', 'RVC', ...    % classification
                           'kernelFunc', kernel, ...
                           'freeBasis', 'on');   % include bias term

        rvm = BaseRVM(parameter);

        try
            % Train
            rvm.train(X_train, y_train);

            % Test
            X_test = X(testIdx,:);
            y_test = double(y(testIdx));

            % Kernel matrix on test set vs relevance vectors
            K_test = kernel.computeMatrix(X_test, rvm.relevanceVectors);

            if strcmp(rvm.freeBasis, 'on')
                BASIS_test = [K_test, ones(sum(testIdx), 1)];
            else
                BASIS_test = K_test;
            end

            % Decision values and sigmoid probabilities
            decision_values = BASIS_test * rvm.weight;
            probabilities   = SB2_Sigmoid_local(decision_values);

            % AUC (assume class "1" is remembered)
            [~, ~, ~, auc] = perfcurve(y_test, probabilities, 1);
            AUCs(i) = auc;

        catch ME
            warning('RVM training failed for fold %d in %s: %s', i, label, ME.message);
            continue;
        end
    end

    % ---------------- Aggregate ----------------
    AUC = mean(AUCs, 'omitnan');
    fprintf('%s (study) RVM AUC: %.3f (based on %d valid folds out of %d)\n', ...
            label, AUC, sum(~isnan(AUCs)), k);
end

% Local sigmoid helper (to avoid dependency issues)
function y = SB2_Sigmoid_local(x)
    y = 1 ./ (1 + exp(-x));
end
