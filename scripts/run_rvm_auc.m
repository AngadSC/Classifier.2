function AUC = run_rvm_auc(X,y, label, participant, kernelType, gamma)
%tun_rvm_auc = RVM AUC with pariticpant dropping 
%
% Inputs:
%   X - feature matrix
%   y - labels
%   label - string describing comparison (e.g., 'FAvsCR')
%   participant - participant IDs (optional)
%   kernelType - kernel type for RVM (default: 'gaussian')
%   gamma - gamma parameter for kernel (default: 0.5)


% need to do a normalziation to the y for it to work witht the rvm
% libraryto work 
y = double(y(:));
X=zscore(X);

%----------------default params----------------------------
if nargin < 5 || isempty(kernelType)
    kernelType = 'gaussian'; %default kernel 
end 

if nargin < 6 || isempty(gamma)
    gamma = 0.5; 
end




%-------------------Fold count ------------------------

if strcmp(label, 'FAvsCR')
        k = 5;
    else
        k = 10;
end



 % ------------------ Participant pruning -----------
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



 % ------------------ Final CV setup ----------------
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



 % ------------------ Cross-validation --------------

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
        %also needed to make it work with the library
        y_train = double(y_train(:));

        % ------------------ Train RVM model ------------------
        % Set up kernel function based on input parameters
        switch kernelType
            case 'gaussian'
                kernel = Kernel('type', 'gaussian', 'gamma', gamma);
            case 'linear'
                kernel = Kernel('type', 'linear');
            case 'polynomial'
                % we can add mmore params as we see fit 
                kernel = Kernel('type', 'polynomial', 'gamma', gamma, 'offset', 0, 'degree', 2);
            case 'sigmoid'
                kernel = Kernel('type', 'sigmoid', 'gamma', gamma, 'offset', 0);
            case 'laplacian'
                kernel = Kernel('type', 'laplacian', 'gamma', gamma);
            otherwise
                warning('Unknown kernel type: %s. Using gaussian.', kernelType);
                kernel = Kernel('type', 'gaussian', 'gamma', gamma);
        end
        
        % RVM parameter setup
        parameter = struct( 'display', 'off',...  % Turn off display for CV
                           'type', 'RVC',...       % Classification
                           'kernelFunc', kernel,...
                           'freeBasis', 'on');     % Include bias term
        
        % Create and train RVM model
        rvm = BaseRVM(parameter);
        
        try
            % Train RVM (it automatically handles binary classification conversion)
            rvm.train(X_train, y_train);

            

             
            
            X_test = X(testIdx,:);
            y_test = double(y(testIdx));
            results = rvm.test(X_test, y_test);
            
            % Get probability scores for AUC calculation
            % The RVM returns binary predictions, but we need probabilities
            % We'll compute the sigmoid of the decision values
            K_test = kernel.computeMatrix(X(testIdx,:), rvm.relevanceVectors);
            
            if strcmp(rvm.freeBasis, 'on')
                BASIS_test = [K_test, ones(sum(testIdx), 1)];
            else
                BASIS_test = K_test;
            end
            
            % Get the decision values (before sigmoid)
            decision_values = BASIS_test * rvm.weight;
            
            % Apply sigmoid to get probabilities
            probabilities = SB2_Sigmoid(decision_values);
            
            % Compute AUC
            %  RVM internally converts labels to 0/1, where the second class is 1
            [~, ~, ~, auc] = perfcurve(y_test, probabilities, 1);
            
            AUCs(i) = auc;
            
        catch ME
            % If RVM training fails for this fold, skip it
            warning('RVM training failed for fold %d in %s: %s', i, label, ME.message);
            continue;
        end
    end

    % ------------------ Aggregate ------------------
    AUC = mean(AUCs, 'omitnan');
    fprintf('%s RVM AUC: %.3f (based on %d valid folds out of %d)\n', ...
            label, AUC, sum(~isnan(AUCs)), k);
end

% Helper function - SB2_Sigmoid (redundant we can remove its in another
% file but its only 2 lines so i put it in 

function y = SB2_Sigmoid(x)
    y = 1./(1+exp(-x));
end