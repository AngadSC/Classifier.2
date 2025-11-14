% ------------------ Path Selection Flag ------------------
% Set this to true for raw data, false for filtered data
useRawPath = false;  % CHANGE THIS FLAG TO SWITCH BETWEEN RAW AND FILTERED

% ------------------ Setup ------------------
% Select input directory based on flag
if useRawPath
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw";
    featurePattern = 'features_labels_raw_*.mat';
    outputFileName = 'SVM_results_raw.mat';
    fprintf('Using RAW data path\n');
else
    % ADD YOUR FILTERED PATH HERE
   % inputDir  = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage3";
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\d_prime";
    %featurePattern = 'stage3_*.mat';  % Update this pattern if different
    featurePattern = 'dprime_exclusion_*.mat';
    outputFileName = 'SVM_results_filtered.mat';
    fprintf('Using FILTERED data path\n');
end

% Output directory remains the same for both
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\SVM";
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

AUC_all = struct();

