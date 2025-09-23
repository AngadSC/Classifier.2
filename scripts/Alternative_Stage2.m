% ====================================================================
% EEG Data Preprocessing: Artifact Rejection (Stage 2 - No Individual Trial Removal)
% ====================================================================
% This script applies artifact rejection criteria **only for participant exclusion**.
% - **Voltage Threshold:** Trials exceeding ±50 µV are counted.
% - **Point-to-Point Difference:** Trials with >25 µV difference are counted.
% - **Baseline Correction:** 100 ms pre-stimulus baseline is subtracted.
% - **Exclusion Criteria:** Participants with **>15% rejected trials** are removed.
%
% -----------------------------
% Key Fixes:
% -----------------------------
% - **DOES NOT remove individual trials.**
% - **Only removes entire participants** who exceed rejection threshold.
% - **Ensures EEG and event codes remain perfectly aligned.**
%
% ====================================================================
% Output:
% - `finalcorrected_EEG_{ID}.mat` -> EEG data (all trials, only valid participants)
% - `finalcorrected_events_{ID}.mat` -> Event codes (aligned)
% - Summary of rejection statistics.
%
% ====================================================================

% Define directories
eegDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_one_Matt';
eventDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_one_Matt';
outputDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_two_Matt_no_trial_removal';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Artifact rejection parameters
Fsample = 250; % Sampling rate in Hz
baseline_ms = 100; % Baseline duration in ms (pre-stimulus)
baseline_len = round(baseline_ms / 1000 * Fsample); % Convert baseline to samples
voltageThreshold = 50; % Absolute voltage threshold in µV
pointToPointThreshold = 25; % Point-to-point voltage difference threshold in µV
rejectionThreshold = 0.15; % **15% rejection threshold**

% Get EEG files
eegFiles = dir(fullfile(eegDir, 'filtered_EEG_*.mat'));

excludedParticipants = {}; % Store excluded participant IDs
summaryResults = struct(); % Store rejection summary

% Loop through each participant
for i = 1:length(eegFiles)
   
    % Load EEG data
    eegFile = fullfile(eegFiles(i).folder, eegFiles(i).name);
    eegData = load(eegFile);
    
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once'); 
    participantKey = sprintf('P%s', participantID); % Ensure field name starts with a letter
   
    % Load event data
    eventFile = fullfile(eventDir, sprintf('filtered_events_%s.mat', participantID));
    
    if ~exist(eventFile, 'file')
        fprintf('Participant %s skipped: Missing event file.\n', participantID);
        continue;
    end
    
    eventData = load(eventFile);
    eegsignal = eegData.eegsignal; 
    eventCodes = eventData.eventCodes; % Ensure this has correct trials
    
    numTrials = size(eegsignal, 3);
    rejected_abs_voltage = 0;
    rejected_point_to_point = 0;

    % Iterate through each trial
    for trial = 1:numTrials
        
        % Compute baseline and subtract it
        baseline = mean(eegsignal(:, 1:baseline_len, trial), 2); 
        trialData = eegsignal(:, :, trial) - baseline;

        % **Voltage artifact rejection check**: Absolute threshold
        if any(abs(trialData(:)) > voltageThreshold)
            rejected_abs_voltage = rejected_abs_voltage + 1;
            continue;
        end
        
        % **Point-to-point difference threshold check**
        pointToPointDiff = abs(diff(trialData, 1, 2)); % Compute difference along time axis
        if any(pointToPointDiff(:) > pointToPointThreshold)
            rejected_point_to_point = rejected_point_to_point + 1;
            continue;
        end
    end
    
    % **Check if participant should be excluded**
    total_rejected = rejected_abs_voltage + rejected_point_to_point;
    rejectionRate = total_rejected / numTrials;
    
    if rejectionRate > rejectionThreshold
        fprintf('Participant %s excluded: %.2f%% of trials rejected (Threshold: 15%%).\n', participantID, rejectionRate * 100);
        excludedParticipants{end + 1} = participantID;
        continue; % Skip saving data for this participant
    end

    % **SAVE ALL TRIALS, Renaming for Consistency**
    correctedEEG = eegsignal; % Keeping all trials
    correctedEvents = eventCodes; % Keeping all event codes

    save(fullfile(outputDir, sprintf('finalcorrected_EEG_%s.mat', participantID)), 'correctedEEG');
    save(fullfile(outputDir, sprintf('finalcorrected_events_%s.mat', participantID)), 'correctedEvents');

    % Store rejection summary
    summaryResults.(participantKey) = struct(...
        'total_trials', numTrials, ...
        'valid_trials', numTrials, ... % Keeping all trials
        'rejected_abs_voltage', rejected_abs_voltage, ...
        'rejected_point_to_point', rejected_point_to_point, ...
        'rejection_rate', rejectionRate ...
    );

    fprintf('Participant %s retained: %d trials (no individual trial removal).\n', participantID, numTrials);
end

% **Print summary of trial rejections per participant**
fprintf('\n===== Trial Rejection Summary =====\n');
participants = fieldnames(summaryResults);

for p = 1:length(participants)
    pid = participants{p};
    data = summaryResults.(pid);
    
    fprintf('Participant %s:\n', pid(2:end));
    fprintf('  Total Trials: %d\n', data.total_trials);
    fprintf('  Rejected (Absolute Voltage > %d µV): %d\n', voltageThreshold, data.rejected_abs_voltage);
    fprintf('  Rejected (Point-to-Point > %d µV): %d\n', pointToPointThreshold, data.rejected_point_to_point);
    fprintf('  Rejection Rate: %.2f%%\n\n', data.rejection_rate * 100);
end

% **Print excluded participants**
if ~isempty(excludedParticipants)
    fprintf('\n===== Excluded Participants =====\n');
    for i = 1:length(excludedParticipants)
        fprintf('Participant %s: More than 15%% trials rejected.\n', excludedParticipants{i});
    end
else
    fprintf('\nNo participants were excluded based on the rejection rate threshold.\n');
end
