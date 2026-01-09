% ====================================================================
% Multivariate LDA Classifier: STUDY phase (Remembered vs Forgotten)
% ====================================================================

% ------------------ Path Selection Flag ------------------
% Set this to true for raw data, false for filtered data
useRawPath = true;  % CHANGE THIS FLAG TO SWITCH BETWEEN RAW AND FILTERED

% ------------------ Setup ------------------
% Select input directory based on flag
if useRawPath
    % STUDY RAW features (from feature_label_moving_bin_study_v3, 'raw')
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw_study";
    featurePattern = 'features_labels_study_raw_*.mat';
    outputFileName = 'LDA_study_results_raw.mat';
    fprintf('Using STUDY RAW data path\n');
else
    % STUDY FILTERED features (update this to your filtered-study folder + pattern)
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_filtered_study";
    featurePattern = 'features_labels_study_filtered_*.mat';
    outputFileName = 'LDA_study_results_filtered.mat';
    fprintf('Using STUDY FILTERED data path\n');
end

% Output directory for study LDA results
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\LDA_study";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get feature files based on selected pattern
featureFiles = dir(fullfile(inputDir, featurePattern));

AUC_all = struct();

% ------------------ Loop Over Participants ------------------
for i = 1:length(featureFiles)
    % Load participant data
    file = fullfile(inputDir, featureFiles(i).name);
    data = load(file); % expect X and y (y: 0 = forgotten, 1 = remembered)

    if ~isfield(data, 'X') || ~isfield(data, 'y')
        warning('File %s missing X or y; skipping.', featureFiles(i).name);
        continue;
    end

    X = data.X;
    y = data.y(:);  % ensure column

    % Extract participant ID
    tokens = regexp(featureFiles(i).name, '(\d+)\.mat$', 'tokens', 'once');
    if isempty(tokens)
        warning('Could not parse participant ID from %s; skipping.', featureFiles(i).name);
        continue;
    end

    participantID = tokens{1};
    subjField = ['P' participantID];
    fprintf('\nProcessing Participant %s (STUDY)\n', participantID);

    % Sanity: need both remembered and forgotten trials
    if numel(unique(y)) < 2
        warning('Participant %s has only one class (all remembered or all forgotten). Skipping.', participantID);
        AUC_all.(subjField).StudyMem = NaN;
        continue;
    end

    % ------------------ Task: Remembered vs Forgotten ------------------
    % For study, y already encodes 0 = forgotten, 1 = remembered.
    % We let run_lda_auc_study handle folds, class counts, etc.
    AUC_all.(subjField).StudyMem = study_run_lda_auc(X, y, [], 'StudyMem');

    fprintf('Completed STUDY LDA for %s (AUC = %.3f)\n', ...
        subjField, AUC_all.(subjField).StudyMem);
end

% ------------------ Save Results ------------------
save(fullfile(outputDir, outputFileName), 'AUC_all');
fprintf('LDA STUDY-phase classification completed and saved as %s.\n', outputFileName);
