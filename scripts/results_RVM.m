% ====================================================================
% Summary Stats for Multivariate RVM Classification
% ====================================================================

% ------------------ Path Selection Flag ------------------
% true  -> use RAW results      (expects: RVM_results_raw.mat)
% false -> use FILTERED results (expects: RVM_results_filtered.mat)
useRawResults = false;   % <-- toggle here

% ------------------ Resolve results file + label ------------------
if useRawResults
    transformation_type = 'raw';
    resultsFile = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\RVM\RVM_results_raw.mat";
    fprintf('Using RAW RVM results\n');
else
    transformation_type = 'filtered';
    resultsFile = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\RVM\RVM_results_filtered.mat";
    fprintf('Using FILTERED RVM results\n');
end

% ------------------ Load RVM results ------------------
load(resultsFile, 'AUC_all', 'kernelType', 'gamma');

% Display RVM configuration
fprintf('\nRVM Configuration:\n');
fprintf('  Kernel Type: %s\n', kernelType);
if exist('gamma', 'var')
    fprintf('  Gamma: %.3f\n', gamma);
end
fprintf('\n');

% ------------------ Subjects & tasks ------------------
subjects = fieldnames(AUC_all);
tasks = {'OldNew','HitMiss','FAvsCR'};
task_labels = {'Old/New','Hit/Miss','FA/CR'};

% ------------------ Quick validity counts (unchanged logic) ------------------
fprintf('Data Validity Check:\n');
fprintf('--------------------\n');
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

% =========================
% RVM Summary (LDA-style)
% =========================
% Build a stats struct (no logic changes — same CI and t-tests)
AUC_stats = struct();

for i = 0 + (1:length(tasks))
    task = tasks{i};
    aucs = [];

    for s = 1:length(subjects)
        subj = subjects{s};
        if isfield(AUC_all.(subj), task)
            val = AUC_all.(subj).(task);
            if ~isempty(val) && isscalar(val)
                v = val;                 % scalar AUC
            elseif ~isempty(val)
                v = mean(val(:));        % collapse per-fold if needed
            else
                v = NaN;
            end
            if ~isnan(v), aucs(end+1) = v; end %#ok<SAGROW>
        end
    end

    if numel(aucs) >= 3
        M  = mean(aucs);
        SE = std(aucs)/sqrt(numel(aucs));
        % Keep original Z-based 95% CI
        CI = [M - 1.96*SE, M + 1.96*SE];
        [~, pval, ~, stats] = ttest(aucs, 0.5);

        AUC_stats.(transformation_type).(task) = struct( ...
            'Mean', M, ...
            'CI', CI, ...
            'N', numel(aucs), ...
            'T', stats.tstat, ...
            'DF', stats.df, ...
            'P', pval );
    end
end

% ------------------ LDA-style table printout ------------------
fprintf('\n==============================================================================\n');
fprintf('Multivariate RVM Classification Summary (Transformation = %s)\n', upper(transformation_type));
if exist('kernelType', 'var')
    fprintf('Kernel: %s', kernelType);
    if exist('gamma', 'var') && ~isempty(gamma)
        fprintf(', Gamma: %.3f', gamma);
    end
    fprintf('\n');
end
fprintf('==============================================================================\n');
fprintf('%-12s %-25s %-25s %-25s\n', 'Metric', 'Old/New', 'Hit/Miss', 'FA/CR');
fprintf('%s\n', repmat('-', 1, 90));

metrics = {'Mean AUC','95%% CI','t(df), p','N'};
for m = 1:length(metrics)
    fprintf('%-12s', metrics{m});
    for i = 1:length(tasks)
        task = tasks{i};
        if isfield(AUC_stats.(transformation_type), task)
            st = AUC_stats.(transformation_type).(task);
            switch metrics{m}
                case 'Mean AUC'
                    entry = sprintf('%.3f', st.Mean);
                case '95%% CI'
                    entry = sprintf('[%.3f, %.3f]', st.CI(1), st.CI(2));
                case 't(df), p'
                    entry = sprintf('t(%d)=%.2f, p=%.4f', st.DF, st.T, st.P);
                case 'N'
                    entry = sprintf('%d', st.N);
            end
        else
            entry = 'N/A';
        end
        fprintf('%-25s', entry);
    end
    fprintf('\n');
end
fprintf('==============================================================================\n');
fprintf('Summary complete.\n');

% ---------------------------
% Plot RVM AUCs by task (title reflects raw/filtered and kernel info)
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
        if isfield(AUC_all.(subj), task)
            val  = AUC_all.(subj).(task);
            if ~isempty(val) && isscalar(val)
                a = val;
            elseif ~isempty(val)
                a = mean(val(:));
            else
                a = NaN;
            end
            if ~isnan(a), aucs(end+1) = a; end %#ok<SAGROW>
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

% Create figure with kernel information in title
if exist('kernelType', 'var') && exist('gamma', 'var')
    titleStr = sprintf('RVM Classification (%s, %s kernel, γ=%.3f)', transformation_type, kernelType, gamma);
elseif exist('kernelType', 'var')
    titleStr = sprintf('RVM Classification (%s, %s kernel)', transformation_type, kernelType);
else
    titleStr = sprintf('RVM Classification (%s)', transformation_type);
end

fig = figure('Color','w','Name',titleStr); hold on;

% Bar + 95% CI error bars (using a different color scheme for RVM)
b = bar(1:numel(tasks), means, 'FaceColor',[0.85 0.65 0.65], 'EdgeColor','none'); %#ok<NASGU>
errLow = means - CIs(:,1)';
errUp  = CIs(:,2)' - means;
errorbar(1:numel(tasks), means, errLow, errUp, 'k', 'LineStyle','none', 'LineWidth', 1.4);

% Overlay individual subject AUCs (jittered)
colors = [0.2 0.4 0.8; 0.8 0.4 0.2; 0.4 0.8 0.2];  % Different colors for each task
for i = 1:numel(tasks)
    x = i + 0.07*randn(1, numel(aucCell{i}));
    scatter(x, aucCell{i}, 28, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.6);
end

% Chance line
yline(0.5, '--', 'Chance', 'LabelHorizontalAlignment','left', 'HandleVisibility','off');

% Ticks/labels
set(gca, 'XTick', 1:numel(tasks), 'XTickLabel', task_labels, 'FontSize', 11);
ylabel('AUC');
title(titleStr);
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
%outdir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\RVM";
%saveas(fig, fullfile(outdir, sprintf('RVM_AUC_plot_%s_%s.png', transformation_type, kernelType)));
%fprintf('\nFigure saved to: %s\n', fullfile(outdir, sprintf('RVM_AUC_plot_%s_%s.png', transformation_type, kernelType)));