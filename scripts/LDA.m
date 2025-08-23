% ====================================================================
% Multivariate LDA Classifier: Old/New, Hit/Miss, FA/CR
% ====================================================================

% ------------------ Setup ------------------
inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving-bin-raw";
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs";


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

    AUC_all.(subjField).OldNew = run_lda_auc(X_task, y_bin,'OLDvsNew');

    % ------------------ Task 2: Hit vs Miss ------------------
    mask = ismember(y, [1, 2]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 1;  % Hit = 1, Miss = 0

    AUC_all.(subjField).HitMiss = run_lda_auc(X_task, y_bin,'HITvsMiss');

    % ------------------ Task 3: FA vs CR ------------------
    skip_FA_CR = {'P65', 'P58', 'P56', 'P09'};  
if ismember(subjField, skip_FA_CR)
    fprintf('Skipping participant %s for FA vs CR: manually excluded.\n', participantID);
    AUC_all.(subjField).FAvsCR = NaN;
else
    mask = ismember(y, [3, 4]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 4;  % FA = 1, CR = 0

    % Optional: skip if only one class
    if numel(unique(y_bin)) < 2
        warning('Skipping participant %s for FA vs CR: one class missing.', participantID);
        AUC_all.(subjField).FAvsCR = NaN;
    else
        AUC_all.(subjField).FAvsCR = run_lda_auc(X_task, y_bin, 'FAvsCR');
    end
end
end
% ------------------ Save Results ------------------
%save(fullfile(outputDir, 'LDA_results_square.mat'), 'AUC_all');
%save(fullfile(outputDir, 'LDA_results_abs.mat'), 'AUC_all');
%save(fullfile(outputDir, 'LDA_results_cube.mat'), 'AUC_all');
save(fullfile(outputDir, 'LDA_results_raw.mat'), 'AUC_all');
disp('LDA classification tasks completed and saved.');