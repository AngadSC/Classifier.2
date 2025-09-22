% ====================================================================
% FA/CR Classification Participant Filtering (FA ≥ 10)
% ====================================================================
% This script filters participants for the FA/CR classification.
% It ensures that only participants with:
%   - At least 10 false alarms (FA ≥ 10)
%
% - Runs AFTER baseline correction & feature extraction.
% - Loads corrected EEG & event files, counts false alarms (FA = 4).
% - Saves a new FA/CR-specific event & EEG file for classification.
%
% ====================================================================

% Define directories
eventDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_two_Matt_no_trial_removal';
eegDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_two_Matt_no_trial_removal';
outputDir = '/Users/faisalanqouor/Desktop/Research/Data_filtering/stage_three_new_Matt_no_trial_removal';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get event files
eventFiles = dir(fullfile(eventDir, 'finalcorrected_events_*.mat'));

% Initialize FA count storage
filteredParticipants_FA_CR = {};
excluded_FA_CR = {};

% Loop through each participant
for i = 1:length(eventFiles)
    
    % Extract participant ID
    participantID = regexp(eventFiles(i).name, '\d+', 'match', 'once');
    
    % Load event data
    eventFile = fullfile(eventDir, sprintf('finalcorrected_events_%s.mat', participantID));
    eventData = load(eventFile);
    
    % Load EEG data
    eegFile = fullfile(eegDir, sprintf('finalcorrected_EEG_%s.mat', participantID));
    eegData = load(eegFile);

    % Extract event codes & EEG signal
    eventCodes = eventData.correctedEvents;  
    correctedEEG = eegData.correctedEEG;

    % Count False Alarms (event code = 4)
    falseAlarms = sum(eventCodes == 4);
    
    % Apply filtering criteria (Only FA ≥ 10)
    if falseAlarms >= 10
        filteredParticipants_FA_CR{end+1} = participantID;
        
        % Save both filtered event codes and EEG signals
        save(fullfile(outputDir, sprintf('filtered_FA_CR_events_%s.mat', participantID)), 'eventCodes');
        save(fullfile(outputDir, sprintf('filtered_FA_CR_EEG_%s.mat', participantID)), 'correctedEEG');

        fprintf('✅ Participant %s included in FA/CR classification (FA count = %d)\n', participantID, falseAlarms);
    else
        excluded_FA_CR{end+1} = participantID;
        fprintf('❌ Participant %s excluded (FA count = %d, below threshold of 10)\n', participantID, falseAlarms);
    end
end

% Save filtered participant lists
save(fullfile(outputDir, 'filtered_FA_CR_participants.mat'), 'filteredParticipants_FA_CR');
save(fullfile(outputDir, 'excluded_FA_CR_participants.mat'), 'excluded_FA_CR');

fprintf('\nFA/CR filtering complete: %d participants included, %d excluded.\n', length(filteredParticipants_FA_CR), length(excluded_FA_CR));
