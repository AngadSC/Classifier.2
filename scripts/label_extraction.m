% ====================================================================
% EEG Classification Labels Extraction (Final Fix - Uncleaned Dataset)
% ====================================================================
% This script assigns classification labels to each EEG trial based on 
% event codes stored in `eventData.test(:,2)`. 
%
% This version uses the **uncleaned dataset**, meaning it retains all 
% participants and trials, without FA/CR-specific exclusions.
%
% ====================================================================
%{
% Define directories
featuresDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_four_Matt/cleaned';
eegDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_three_Matt'; 
eventDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_three_Matt';

% Get EEG files
eegFiles = dir(fullfile(eegDir, 'filtered_FA_CR_EEG_*.mat'));
%}
function [OldNew_labels, HitsMisses_labels, CR_FA_labels, Perceived_labels] = label_extraction(test)
eventCodes = test(:,2); % Extract **only the trial outcome column**
numTrials=450;
% Define event codes
HIT_CODE = 1;    
MISS_CODE = 2;  
CR_CODE = 3;     
FA_CODE = 4;    

% Loop through each participant
%for i = 1:length(eegFiles)
    %{
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once');
    
    % Load extracted EEG features
    featureFile = fullfile(featuresDir, sprintf('features_%s.mat', participantID));
    if ~exist(featureFile, 'file')
        warning('⚠️ Feature file missing for Participant %s. Skipping...', participantID);
        continue;
    end
    load(featureFile, 'FN400_values', 'LPP_values');
    
    % Load event data
    eventFile = fullfile(eventDir, sprintf('filtered_FA_CR_events_%s.mat', participantID));
    if ~exist(eventFile, 'file')
        warning('⚠️ Event file missing for Participant %s. Skipping...', participantID);
        continue;
    end
    eventData = load(eventFile);

    % Extract event codes (second column only)
    eventCodes = eventData.eventCodes;
    
    % Ensure label sizes match EEG data
    numTrials = length(eventCodes);
    if numTrials ~= length(FN400_values)
        warning('⚠️ Mismatch in trial counts for Participant %s. Skipping...', participantID);
        continue;
    end
    
    % Debugging: Print event code counts
    fprintf('🔍 Participant %s - Event Code Counts: Hits=%d, Misses=%d, CR=%d, FA=%d, Total=%d\n', ...
        participantID, sum(eventCodes == HIT_CODE), sum(eventCodes == MISS_CODE), ...
        sum(eventCodes == CR_CODE), sum(eventCodes == FA_CODE), length(eventCodes));

    % Debugging: Check if any event codes are missing
    fprintf('🔍 Participant %s - Unique Event Codes: %s\n', participantID, mat2str(unique(eventCodes)));
    %}
    % **Fix: Apply Labels Only to Relevant Trials**
    OldNew_labels = ismember(eventCodes, [HIT_CODE, FA_CODE]);  % Old (Hit + FA) = 1, New (Miss + CR) = 0
    
    % ✅ Ensure Hits/Misses Only Includes OLD Trials (Hits & Misses)
    HitsMisses_labels = nan(numTrials, 1);
    HitsMisses_labels(eventCodes == HIT_CODE) = 1;  % Hit = 1
    HitsMisses_labels(eventCodes == MISS_CODE) = 0; % Miss = 0

    % ✅ Ensure CR/FA Only Includes NEW Trials (CR & FA)
    CR_FA_labels = nan(numTrials, 1);
    CR_FA_labels(eventCodes == FA_CODE) = 1; % FA = 1
    CR_FA_labels(eventCodes == CR_CODE) = 0; % CR = 0

    % ✅ Perceived Old/New: Hit and FA = 1, Miss and CR = 0
    Perceived_labels = ismember(eventCodes, [HIT_CODE, FA_CODE]); 

    % **Remove NaNs correctly**
    HitsMisses_labels = HitsMisses_labels(~isnan(HitsMisses_labels)); % Keep valid trials only
    CR_FA_labels = CR_FA_labels(~isnan(CR_FA_labels)); % Keep valid trials only
    %{
    % Debugging: Print FA/CR and Hits/Misses actual sums
    fprintf('✅ Participant %s - FA/CR Check: CR = %d, FA = %d, Total FA/CR = %d (Expected: 225)\n', ...
        participantID, sum(eventCodes == CR_CODE), sum(eventCodes == FA_CODE), length(CR_FA_labels));

    fprintf('✅ Participant %s - Hits/Misses Check: Hits = %d, Misses = %d, Total Hits/Misses = %d (Expected: 225)\n', ...
        participantID, sum(eventCodes == HIT_CODE), sum(eventCodes == MISS_CODE), length(HitsMisses_labels));

    % Ensure label sums match expectations
    if length(CR_FA_labels) ~= 225
        warning('⚠️ Participant %s - FA/CR labels do not sum to 225!', participantID);
    end
    if length(HitsMisses_labels) ~= 225
        warning('⚠️ Participant %s - Hits/Misses labels do not sum to 225!', participantID);
    end
    if (sum(Perceived_labels == 0) + sum(Perceived_labels == 1)) ~= 450
        warning('⚠️ Participant %s - Perceived labels do not sum to 450!', participantID);
    end
    if (sum(OldNew_labels == 0) + sum(OldNew_labels == 1)) ~= 450
        warning('⚠️ Participant %s - Old/New labels do not sum to 450!', participantID);
    end

    % Save classification labels
    save(fullfile(featuresDir, sprintf('labels_%s.mat', participantID)), ...
        'OldNew_labels', 'HitsMisses_labels', 'CR_FA_labels', 'Perceived_labels');
    
    fprintf('✅ Labels assigned for Participant %s\n\n', participantID);
    %}
%end

disp('✅ Event labels extracted and saved.');
