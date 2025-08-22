% ====================================================================
% EEG Classification: Averaged ROC & AUC Computation
% ====================================================================
% Computes:
% - Averaged Receiver Operating Characteristic (ROC) Curves
% - Area Under Curve (AUC) for classification tasks
%
% ====================================================================
%{
% Define directories
featuresDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_four_Matt/cleaned';
outputDir = '/Users/faisalanqouor/Desktop/Research/classification_Matt_Results/cleaned';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get feature files
featureFiles = dir(fullfile(featuresDir, 'features_*.mat'));
%}
function [ROC_data, AUC_results] = Univariate_Classifier(FN400_values, LPP_values, OldNew_labels, HitsMisses_labels, CR_FA_labels, Perceived_labels)
% Define classification tasks
classification_tasks = {'OldNew', 'HitsMisses', 'CR_FA', 'Perceived'};

% Initialize storage for AUC values and ROC data
AUC_results = struct();
ROC_data = struct();

% Loop through each participant
%for i = 1:length(featureFiles)
    %{
    % Extract participant ID
    participantID = regexp(featureFiles(i).name, '\d+', 'match', 'once');
    
    % Load extracted EEG features
    featureFile = fullfile(featuresDir, sprintf('features_%s.mat', participantID));
    if exist(featureFile, 'file')
        load(featureFile, 'FN400_values', 'LPP_values');
    else
        warning('Feature file missing for Participant %s. Skipping...', participantID);
        continue;
    end

    % Load classification labels
    labelsFile = fullfile(featuresDir, sprintf('labels_%s.mat', participantID));
    if exist(labelsFile, 'file')
        load(labelsFile, 'OldNew_labels', 'HitsMisses_labels', 'CR_FA_labels', 'Perceived_labels');
    else
        warning('Labels file missing for Participant %s. Skipping...', participantID);
        continue;
    end
    %}
    % Convert labels to column vectors
    OldNew_labels = OldNew_labels(:,1); 
    HitsMisses_labels = HitsMisses_labels(:,1); 
    CR_FA_labels = CR_FA_labels(:,1); 
    Perceived_labels = Perceived_labels(:,1);
    %{
    % Ensure all labels have correct sizes
    if length(OldNew_labels) ~= 450 || length(Perceived_labels) ~= 450
        warning('Mismatch in OldNew/Perceived label count for Participant %s. Expected 450.', participantID);
        continue;
    end
    if length(HitsMisses_labels) ~= 225 || length(CR_FA_labels) ~= 225
        warning('Mismatch in Hits/Misses or CR/FA label count for Participant %s. Expected 225.', participantID);
        continue;
    end
    %}
    % Store label sets with proper alignment
    label_data = {OldNew_labels, HitsMisses_labels, CR_FA_labels, Perceived_labels};

    % Ensure feature values are column vectors
    FN400_values = FN400_values(:);
    LPP_values = LPP_values(:);

    % Loop through FN400 and LPP separately
    for feature_type = ["FN400", "LPP"]
        
        % Select the correct feature set
        feature_values = FN400_values;
        if feature_type == "LPP"
            feature_values = LPP_values;
        end
        
        % Loop through each classification task
        for j = 1:length(classification_tasks)
            class_name = classification_tasks{j};
            labels = label_data{j}; 

            % Adjust feature length for CR/FA and Hits/Misses (225 trials)
            if strcmp(class_name, 'CR_FA') || strcmp(class_name, 'HitsMisses')
                feature_values_adj = feature_values(1:225); % Use only first 225 trials
            else
                feature_values_adj = feature_values; % Use full 450 trials
            end
            %{
            % Ensure labels and features are properly aligned
            if length(labels) ~= length(feature_values_adj)
                warning('Mismatch in feature and label length for Participant %s, %s. Skipping...', participantID, class_name);
                continue;
            end

            % Debugging print statement
            fprintf('Participant %s: %s - Trials = %d, Labels = %d\n', ...
                    participantID, class_name, length(feature_values_adj), length(labels));

            % Ensure feature values do not contain NaNs
            if any(isnan(feature_values_adj))
                warning('Skipping %s for Participant %s: Found NaN values in features.', class_name, participantID);
                continue;
            end
            %}
            % Ensure labels contain exactly 2 unique classes
            unique_labels = unique(labels);
            if numel(unique_labels) ~= 2 || any(isnan(labels))
                warning('Skipping %s: Labels do not contain exactly 2 unique values or contain NaNs.', class_name);
                continue;
            end
            % Compute ROC curve and AUC
            [X, Y, ~, AUC] = perfcurve(labels, feature_values_adj, max(unique_labels));

            % Store AUC value
            %AUC_results.(sprintf('P%s', participantID)).(sprintf('%s_%s_AUC', feature_type, class_name)) = AUC;
            AUC_results.(sprintf('%s_%s_AUC', feature_type, class_name)) = AUC;
            
          %  1 value per trial per freq (up to 20 samples ) 
          %  20xlenxtrial 
          %  450x327

            % Ensure ROC_data structure is initialized correctly
            if ~isfield(ROC_data, class_name)
                ROC_data.(class_name) = struct('FN400', struct('X', [], 'Y', []), ...
                                               'LPP', struct('X', [], 'Y', []));
            end

            % Store ROC X, Y values for averaging
            ROC_data.(class_name).(feature_type).X = ...
                [ROC_data.(class_name).(feature_type).X; ...
                 interp1(linspace(0,1,length(X)), X, linspace(0,1,100))];
            ROC_data.(class_name).(feature_type).Y = ...
                [ROC_data.(class_name).(feature_type).Y; ...
                 interp1(linspace(0,1,length(Y)), Y, linspace(0,1,100))];
        end
    end

    %fprintf('✅ Computed ROC & AUC for Participant %s\n', participantID);
%end

% ====================================================================
% Compute and Save AUC Results
% ====================================================================

% Save AUC results for all participants
%save(fullfile(outputDir, 'AUC_results_.mat'), 'AUC_results');

disp('🎯 Averaged ROC and AUC computation complete.');

% just checking to see if my git repos are working 
% checking if it works from laptop to pc 