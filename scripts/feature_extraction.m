% ====================================================================
% EEG Feature Extraction: FN400 & LPP
% ====================================================================
% This script extracts trial-wise EEG features for classification.
% It computes the mean amplitude for two key event-related potentials (ERPs):
% - **FN400** (400–600 ms, at Fz)
% - **LPP** (600–900 ms, at P3)
%
% ====================================================================
%{
% Define directories
eegDir = '/Users/faisalanqouor/Desktop/Research/data_matt/data'; 
eventDir = '/Users/faisalanqouor/Desktop/Research/data_matt/behaviour';
outputDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_four_Matt/cleaned';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get EEG files
eegFiles = dir(fullfile(eegDir, 'test*.mat')); 
%}
function [FN400_values, LPP_values] = feature_extraction(data,test)
% Define electrode indices (update with actual indices)
Fz_index = 21; % FN400 at Fz
P3_index = 87; % LPP at P3

% Define time windows (in milliseconds)
FN400_window = [400, 600]; % FN400 effect
LPP_window = [600, 900];   % LPP effect

% Define sampling rate
sampling_rate = 250; 

% Convert time windows to sample indices
FN400_idx = round(FN400_window / 1000 * sampling_rate);
LPP_idx = round(LPP_window / 1000 * sampling_rate);

% Loop through each participant
%for i = 1:length(eegFiles)
    %{
    % Load EEG data
    eegFile = fullfile(eegFiles(i).folder, eegFiles(i).name);
    eegData = load(eegFile);
    
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once'); 
    participantKey = sprintf('P%s', participantID);
    
    % Load event data
    eventFile = fullfile(eventDir, sprintf('events_%s.mat', participantID));
    eventData = load(eventFile);
    %}
    % Extract EEG signal and event codes
    correctedEEG = data; % Ensure this matches the structure of your .mat files
    eventCodes = test(:,2); % Extract **only the trial outcome column**

    % Get number of trials
    numTrials = size(correctedEEG, 3);
    
    % Initialize feature storage
    FN400_values = zeros(numTrials, 1);
    LPP_values = zeros(numTrials, 1);
    
    % MAIN STEP: Extract mean amplitude per trial
    for t = 1:numTrials
        FN400_values(t) = mean(correctedEEG(Fz_index, FN400_idx(1):FN400_idx(2), t), 2);
        LPP_values(t) = mean(correctedEEG(P3_index, LPP_idx(1):LPP_idx(2), t), 2);
    end
    %{
    % Print debugging information
    fprintf('Participant %s: Extracted FN400 = %d trials, LPP = %d trials\n', participantID, length(FN400_values), length(LPP_values));

    % Print sample feature-label mapping
    disp('Sample Check (First 10 Trials):');
    disp(table(eventCodes(1:10), FN400_values(1:10), LPP_values(1:10)));

    % Save extracted features
    save(fullfile(outputDir, sprintf('features_%s.mat', participantID)), 'FN400_values', 'LPP_values');
    
    fprintf('Processed Participant %s: %d Trials\n', participantID, numTrials);
    %}
%end

disp('Feature extraction complete.');
