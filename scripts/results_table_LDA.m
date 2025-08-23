% ====================================================================
% Summary Stats for Multivariate LDA Classification (Sucheta-style CI and t-test)
% ====================================================================
transformation_type = 'raw';  % Change as needed

% Load LDA results
load(sprintf('C:/Users/Angad/OneDrive/Desktop/Comp Memory Lab/Classifier.2/outputs/LDA_results_%s.mat', transformation_type));

% Extract participant IDs
subjects = fieldnames(AUC_all);

% Classification tasks and labels
tasks = {'OldNew', 'HitMiss', 'FAvsCR'};
task_labels = {'Old/New', 'Hit/Miss', 'FA/CR'};

% Initialize output structure
AUC_stats = struct();

% Loop through each classification task
for i = 1:length(tasks)
    task = tasks{i};
    aucs = [];

    for s = 1:length(subjects)
        subjID = subjects{s};
        if isfield(AUC_all.(subjID), task)
            val = AUC_all.(subjID).(task);
            if ~isnan(val)
                aucs(end+1) = val;
            end
        end
    end

    % Skip if not enough subjects
    if length(aucs) < 3
        fprintf('%s: Not enough data.\n', task_labels{i});
        continue;
    end

    % Group-level stats
    mean_auc = mean(aucs);
    std_auc = std(aucs);
    n = length(aucs);
    SEM = std_auc / sqrt(n);

    % Use t-distribution instead of Z
    t_val = tinv(0.975, n - 1);
    CI_lower = mean_auc - t_val * SEM;
    CI_upper = mean_auc + t_val * SEM;

    % One-sample t-test against chance
    [~, p, ci, stats] = ttest(aucs, 0.5);

    % Store everything
    AUC_stats.(transformation_type).(task) = struct(...
        'Mean', mean_auc, ...
        'CI', [CI_lower, CI_upper], ...
        'N', n, ...
        'T', stats.tstat, ...
        'DF', stats.df, ...
        'P', p ...
    );
end

% ====================================================================
% Display Summary Table (LDA Classification Performance)
% ====================================================================

fprintf('\nMultivariate LDA Classification Summary (Transformation = %s)\n', upper(transformation_type));
fprintf('%-12s %-25s %-25s %-25s\n', 'Metric', 'Old/New', 'Hit/Miss', 'FA/CR');
fprintf('%s\n', repmat('-', 1, 90));

metrics = {'Mean AUC', '95%% CI', 't(df), p', 'N'};

for m = 1:length(metrics)
    fprintf('%-12s', metrics{m});
    for i = 1:length(tasks)
        task = tasks{i};

        if isfield(AUC_stats.(transformation_type), task)
            stats = AUC_stats.(transformation_type).(task);
            switch metrics{m}
                case 'Mean AUC'
                    entry = sprintf('%.3f', stats.Mean);
                case '95%% CI'
                    entry = sprintf('[%.3f, %.3f]', stats.CI(1), stats.CI(2));
                case 't(df), p'
                    entry = sprintf('t(%d)=%.2f, p=%.4f', stats.DF, stats.T, stats.P);
                case 'N'
                    entry = sprintf('%d', stats.N);
            end
        else
            entry = 'N/A';
        end
        fprintf('%-25s', entry);
    end
    fprintf('\n');
end

disp('Summary complete.');

%testing 