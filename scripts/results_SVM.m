% ====================================================================
% Summary Stats for Multivariate SVM Classification (Like Table 4)
% ====================================================================

% Load SVM results
load("C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\SVM_results_raw.mat")

% Extract participant IDs
subjects = fieldnames(AUC_all);
tasks = {'OldNew','HitMiss','FAvsCR'};

for i = 1:numel(tasks)
    task = tasks{i};
    vals = nan(numel(subjects),1);
    for s = 1:numel(subjects)
        subj = subjects{s};
        if isfield(AUC_all.(subj), task)
            v = AUC_all.(subj).(task);
            if ~isempty(v) && isscalar(v)
                vals(s) = v;
            elseif ~isempty(v)           % if stored per-fold, collapse
                vals(s) = mean(v(:));
            end
        end
    end
    bad = subjects(~isfinite(vals));
    fprintf('%s: valid=%d / total=%d\n', task, sum(isfinite(vals)), numel(vals));
    if ~isempty(bad)
        fprintf('  Missing/NaN: %s\n', strjoin(bad', ', ')); 
    end
end


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
% ---------------------------
% Plot SVM AUCs by task (raw)
% ---------------------------
means   = nan(1, numel(tasks));
CIs     = nan(numel(tasks), 2);
pvals   = nan(1, numel(tasks));
aucCell = cell(1, numel(tasks));

for i = 1:numel(tasks)
    task = tasks{i};
    aucs = [];

    for s = 1:numel(subjects)
        subj = subjects{s};
        val  = AUC_all.(subj).(task);
        if ~isnan(val)
            aucs(end+1) = val; %#ok<SAGROW>
        end
    end

    aucCell{i} = aucs;
    M  = mean(aucs);
    SE = std(aucs) / sqrt(numel(aucs));
    means(i) = M;
    CIs(i,:) = [M - 1.96*SE, M + 1.96*SE];

    [~, p] = ttest(aucs, 0.5); % against chance
    pvals(i) = p;
end

fig = figure('Color','w','Name','SVM AUC by Task (raw)'); hold on;

% Bar + 95% CI error bars
b = bar(1:numel(tasks), means, 'FaceColor',[0.7 0.78 0.92], 'EdgeColor','none'); %#ok<NASGU>
errLow = means - CIs(:,1)';
errUp  = CIs(:,2)' - means;
errorbar(1:numel(tasks), means, errLow, errUp, 'k', 'LineStyle','none', 'LineWidth', 1.4);

% Overlay individual subject AUCs (jittered)
for i = 1:numel(tasks)
    x = i + 0.07*randn(1, numel(aucCell{i}));
    scatter(x, aucCell{i}, 28, 'filled', 'MarkerFaceAlpha', 0.6);
end

% Chance line
yline(0.5, '--', 'Chance', 'LabelHorizontalAlignment','left', 'HandleVisibility','off');

% Ticks/labels
set(gca, 'XTick', 1:numel(tasks), 'XTickLabel', task_labels, 'FontSize', 11);
ylabel('AUC');
title('SVM Classification (raw)');
ylim([0.4 1]); grid on; box on;

% Significance stars
for i = 1:numel(tasks)
    stars = '';
    if pvals(i) < 1e-3
        stars = '***';
    elseif pvals(i) < 1e-2
        stars = '**';
    elseif pvals(i) < 5e-2
        stars = '*';
    end
    if ~isempty(stars)
        text(i, CIs(i,2) + 0.03, stars, 'HorizontalAlignment','center', 'FontSize', 12);
    end
end

hold off;

% Optional: save the figure
% outdir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs";
% saveas(fig, fullfile(outdir, 'SVM_AUC_plot_raw.png'));
