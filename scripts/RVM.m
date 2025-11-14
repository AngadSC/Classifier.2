% ------------------ Path Selection Flag ------------------
% Set this to true for raw data, false for filtered data
useRawPath = false;  % CHANGE THIS FLAG TO SWITCH BETWEEN RAW AND FILTERED
kernelType = 'gaussian'; % we can change it i already coded it into the function 
gamma = 0.5;

% ------------------ Setup ------------------
% Select input directory based on flag
if useRawPath
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw";
    featurePattern = 'features_labels_raw_*.mat';
    outputFileName = 'RVM_results_raw.mat';
    fprintf('Using RAW data path\n');
else
    % ADD YOUR FILTERED PATH HERE
   % inputDir  = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage3";
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\d_prime";
    %featurePattern = 'stage3_*.mat';  % Update this pattern if different
    featurePattern = 'dprime_exclusion_*.mat';
    outputFileName = 'RVM_results_filtered.mat';
    fprintf('Using FILTERED data path\n');
end

% Output directory remains the same for both
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\RVM";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get feature files based on selected pattern
featureFiles = dir(fullfile(inputDir, featurePattern));

% Alternative patterns (uncomment if needed):
% For squared features:
% if useRawPath
%     featurePattern = 'features_labels_square_*.mat';
%     outputFileName = 'SVM_results_square.mat';
% end
% For abs features:
% if useRawPath
%     featurePattern = 'features_labels_abs_*.mat';
%     outputFileName = 'SVM_results_abs.mat';
% end
% For cube features:
% if useRawPath
%     featurePattern = 'features_labels_cube_*.mat';
%     outputFileName = 'SVM_results_cube.mat';
% end


fprintf('\n==================================================\n');
fprintf('RVM Classification with %s kernel (gamma=%.3f)\n', kernelType, gamma);
fprintf('==================================================\n');

AUC_all = struct();

% ------------------ Loop Over Subjects ------------------
for i = 1:length(featureFiles)
    % Load subject data
    file = fullfile(inputDir, featureFiles(i).name);
    data = load(file); % loads X and y
    X = data.X;
    y = data.y;
    
    % Extract participant ID
    tokens = regexp(featureFiles(i).name, '_(\d+)\.mat$', 'tokens', 'once');
    participantID = tokens{1};
    subjField = ['P' participantID];
    
    fprintf('\nProcessing Participant %s\n', participantID);
    fprintf('----------------------------------------\n');
    
    % ------------------ Task 1: Old vs New ------------------
    % Classify [Hit, Miss] (1,2) vs [CR, FA] (3,4)
    mask = ismember(y, [1, 2, 3, 4]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = ismember(y_task, [1, 2]); % Old = 1, New = 0
    
    fprintf('Task 1: Old vs New - ');
    AUC_all.(subjField).OldNew = run_rvm_auc(X_task, y_bin, 'OLDvsNew', [], kernelType, gamma);
    
    % ------------------ Task 2: Hit vs Miss ------------------
    mask = ismember(y, [1, 2]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 1; % Hit = 1, Miss = 0
    
    fprintf('Task 2: Hit vs Miss - ');
    AUC_all.(subjField).HitMiss = run_rvm_auc(X_task, y_bin, 'HITvsMiss', [], kernelType, gamma);
    
    % ------------------ Task 3: FA vs CR ------------------
    mask = ismember(y, [3, 4]);
    X_task = X(mask, :);
    y_task = y(mask);
    y_bin = y_task == 4; % FA = 1, CR = 0
    
    if numel(unique(y_bin)) < 2
        warning('Skipping participant %s for FA vs CR: one class missing.', participantID);
        AUC_all.(subjField).FAvsCR = NaN;
    else
        fprintf('Task 3: FA vs CR - ');
        AUC_all.(subjField).FAvsCR = run_rvm_auc(X_task, y_bin, 'FAvsCR', [], kernelType, gamma);
    end
    
    fprintf('Completed: %s\n', subjField);
end

% ------------------ Save Results ------------------
save(fullfile(outputDir, outputFileName), 'AUC_all', 'kernelType', 'gamma');
fprintf('\n==================================================\n');
fprintf('RVM classification tasks completed and saved as:\n%s\n', fullfile(outputDir, outputFileName));
fprintf('Kernel: %s, Gamma: %.3f\n', kernelType, gamma);
fprintf('==================================================\n');