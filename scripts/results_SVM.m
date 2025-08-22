% ====================================================================
% Summary Stats for Multivariate SVM Classification (Like Table 4)
% ====================================================================

% Load SVM results
load('/Users/faisalanqouor/Desktop/Research/Multivariate_pipeline/results_moving_SVM/SVM_results_raw.mat');

% Extract participant IDs
subjects = fieldnames(AUC_all);

% Classification tasks
tasks = {'OldNew', 'HitMiss', 'FAvsCR'};
task_labels = {'Old/New', 'Hit/Miss', 'FA/CR'};

fprintf('SVM Classification Results for raw\n');
fprintf('--------------------------------------------\n');

for i = 1:length(tasks)
    task = tasks{i};
    aucs = [];

    for s = 1:length(subjects)
        subj = subjects{s};
        val = AUC_all.(subj).(task);
        if ~isnan(val)
            aucs(end+1) = val;
        end
    end

    % Compute mean and 95% confidence interval
    M = mean(aucs);
    CI = [M - 1.96*std(aucs)/sqrt(length(aucs)), ...
          M + 1.96*std(aucs)/sqrt(length(aucs))];

    % Perform one-sample t-test against 0.5 (chance)
    [~, pval, ~, stats] = ttest(aucs, 0.5);
    
    % Report
    fprintf('%-12s AUC = %.3f [%.3f, %.3f], t(%d) = %.2f, p = %.4f\n', ...
        task_labels{i}, M, CI(1), CI(2), stats.df, stats.tstat, pval);
end