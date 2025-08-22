% ====================================================================
% Multivariate SVM Classifier: Old/New, Hit/Miss, FA/CR
% ====================================================================

% ------------------ Setup ------------------
inputDir = '/Users/faisalanqouor/Desktop/Research/Multivariate_pipeline/feature_label_raw';
outputDir = '/Users/faisalanqouor/Desktop/Research/Multivariate_pipeline/results_SVM';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%featureFiles = dir(fullfile(inputDir, 'features_labels_square_*.mat'));
%featureFiles = dir(fullfile(inputDir, 'features_labels_abs_*.mat'));
%featureFiles = dir(fullfile(inputDir, 'features_labels_cube_*.mat'));
featureFiles = dir(fullfile(inputDir, 'features_labels_raw_*.mat'));
AUC_all = struct();

% ------------------ Loop Over Subjects ------------------
for i = 1:length(featureFiles)

    % Load subject data
    file = fullfile(inputDir, featureFiles(i).name);
    data = load(file);  % loads X and y
    X = data.X;
    y = data.y;

    % Extract participant ID
    participantID = regexp(featureFiles(i).name, '\d+', 'match', 'once');
    subjField = ['P' participantID];

    fprintf('\nProcessing Participant %s\n', participantID);

    % ------------------ Task 1: Old vs New ------------------
    % Classify [Hit, Miss] (1,2) vs [CR, FA] (3,4)
    mask = ismember(y, [1, 2, 3, 4]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = ismember(y_task, [1, 2]);  % Old = 1, New = 0

    AUC_all.(subjField).OldNew = run_svm_auc(X_task, y_bin, 'OLDvsNew');

    % ------------------ Task 2: Hit vs Miss ------------------
    mask = ismember(y, [1, 2]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 1;  % Hit = 1, Miss = 0

    AUC_all.(subjField).HitMiss = run_svm_auc(X_task, y_bin, 'HITvsMiss');

    % ------------------ Task 3: FA vs CR ------------------
    mask = ismember(y, [3, 4]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 4;  % FA = 1, CR = 0

    if numel(unique(y_bin)) < 2
        warning('Skipping participant %s for FA vs CR: one class missing.', participantID);
        AUC_all.(subjField).FAvsCR = NaN;
    else
        AUC_all.(subjField).FAvsCR = run_svm_auc(X_task, y_bin, 'FAvsCR');
    end

    fprintf('Completed: %s\n', subjField);
end

% ------------------ Save Results ------------------
%save(fullfile(outputDir, 'SVM_results_square.mat'), 'AUC_all');
%save(fullfile(outputDir, 'SVM_results_abs.mat'), 'AUC_all');
%save(fullfile(outputDir, 'SVM_results_cube.mat'), 'AUC_all');
save(fullfile(outputDir, 'SVM_results_raw.mat'), 'AUC_all');
disp('SVM classification tasks completed and saved.');