% ====================================================================
% Multivariate RVM Classifier: STUDY phase (Remembered vs Forgotten)
% ====================================================================

% ------------------ Path / Kernel Settings ------------------
% Set this to true for raw data, false for filtered data
useRawPath = true;   % CHANGE THIS FLAG TO SWITCH BETWEEN RAW AND FILTERED

kernelType = 'laplacian';  % already implemented in run_rvm_auc_study
gamma      = 0.5;

% ------------------ Setup ------------------
if useRawPath
    % STUDY RAW features (from feature_label_moving_bin_study_v3, 'raw')
    inputDir       = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw_study";
    featurePattern = 'features_labels_study_raw_*.mat';
    outputFileName = 'RVM_study_results_raw.mat';
    fprintf('Using STUDY RAW data path\n');
else
    % STUDY FILTERED features (update this path + pattern to match your pipeline)
    inputDir       = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_filtered_study";
    featurePattern = 'features_labels_study_filtered_*.mat';
    outputFileName = 'RVM_study_results_filtered.mat';
    fprintf('Using STUDY FILTERED data path\n');
end

% Output directory for study RVM
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\RVM_study";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get feature files based on selected pattern
featureFiles = dir(fullfile(inputDir, featurePattern));

fprintf('\n==================================================\n');
fprintf('STUDY RVM Classification with %s kernel (gamma=%.3f)\n', kernelType, gamma);
fprintf('==================================================\n');

AUC_all = struct();

% ------------------ Loop Over Participants ------------------
for i = 1:length(featureFiles)
    % Load participant data
    file = fullfile(inputDir, featureFiles(i).name);
    data = load(file); % expect X and y; y: 0 = forgotten, 1 = remembered

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
    fprintf('----------------------------------------\n');

    % Need both remembered and forgotten
    if numel(unique(y)) < 2
        warning('Participant %s has only one class (all remembered or all forgotten). Skipping.', participantID);
        AUC_all.(subjField).StudyMem = NaN;
        continue;
    end

    % ------------------ Task: Remembered vs Forgotten ------------------
    % y already encodes 0 = forgotten, 1 = remembered.
    fprintf('Task: Remembered vs Forgotten - ');
    AUC_all.(subjField).StudyMem = study_run_rvm_auc(X, y, [], 'StudyMem', kernelType, gamma);

    fprintf('Completed: %s (RVM AUC = %.3f)\n', ...
        subjField, AUC_all.(subjField).StudyMem);
end

% ------------------ Save Results ------------------
save(fullfile(outputDir, outputFileName), 'AUC_all', 'kernelType', 'gamma');

fprintf('\n==================================================\n');
fprintf('STUDY RVM classification completed and saved as:\n%s\n', ...
    fullfile(outputDir, outputFileName));
fprintf('Kernel: %s, Gamma: %.3f\n', kernelType, gamma);
fprintf('==================================================\n');
