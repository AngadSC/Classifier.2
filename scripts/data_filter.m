% ====================================================================
% EEG Data Preprocessing: Miss Count & Accuracy Calculation (Stage 1)
% ====================================================================
% This script processes event codes to determine:
% 1. The number of misses (i.e., trials where the participant 
%    failed to recognize an old word).
% 2. The participant's overall recognition accuracy.
% 3. Excludes participants based on:
%      - Condition 1: Misses < 11 (Fewer than 11 misses)
%      - Condition 2: Accuracy < 50% (Less than 50% correct recognition)
%
% -----------------------------
% Key Steps:
% -----------------------------
% - Extracts event codes for each participant.
% - Counts the number of misses (incorrect responses to old words).
% - Computes recognition accuracy using the formula:
%      Accuracy = (Hits + Correct Rejections) / (Hits + Misses + CR + FA)
%   where:
%      - Hits (H) = Correctly identifying an old word
%      - Correct Rejections (CR) = Correctly rejecting a new word
%      - Misses (M) = Failing to recognize an old word
%      - False Alarms (FA) = Incorrectly identifying a new word as old
%
% ====================================================================

%eegDir = '/Users/faisalanqouor/Desktop/Research/Data_MV';
%eventDir = '/Users/faisalanqouor/Desktop/Research/Data_MV';
%outputDir = '/Users/faisalanqouor/Desktop/Research/Data_MV/s1+s2'; 

eegDir = "\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
eventDir = "\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data"; 
outputDir = "\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage1";


if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

eegFiles = dir(fullfile(eegDir, 'test_*.mat')); 
eventFiles = dir(fullfile(eventDir, 'events_*.mat'));

% Initialize storage variables
filteredEEG = {}; 
filteredEvents = {}; 
excludedParticipants = [];
missCounts = []; 

HIT_CODE = 1;
MISS_CODE = 2;
CR_CODE = 3;
FA_CODE = 4;

for i = 1:length(eegFiles)
    
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once'); 
    
    % Load EEG data
    eegFile = fullfile(eegFiles(i).folder, eegFiles(i).name);
    eegData = load(eegFile);
    %eegsignal = eegData.test;
    eegsignal = eegData.data;


    % Fix the leading zero issue in event (behavior) filenames
    %eventFile = fullfile(eventDir, ['Label_', sprintf('%02d', str2double(participantID)), '.mat']); % Format ID with leading zero
    eventFile = fullfile(eventDir, sprintf('events_%s.mat', participantID));


    if ~isfile(eventFile)
        fprintf('Participant %s skipped: Missing event file.\n', participantID);
        continue;
    end

    % Load event data
    eventData = load(eventFile);
    %eventCodes = eventData.events();
    eventCodes = eventData.test(:,2);

    %  % Extract trial outcomes

    % Compute accuracy metrics
    hits = sum(eventCodes == HIT_CODE);
    correctRejections = sum(eventCodes == CR_CODE);
    totalTrials = length(eventCodes);
    accuracy = (hits + correctRejections) / totalTrials * 100;

    % Count misses
    oldTrials = sum(eventCodes == HIT_CODE | eventCodes == MISS_CODE);
    numMisses = sum(eventCodes == MISS_CODE);

    % Store participant ID and miss count
    missCounts = [missCounts; str2double(participantID), numMisses]; 

    % Exclusion based on criteria
    if numMisses < 11
        excludedParticipants = [excludedParticipants; str2double(participantID)];
        fprintf('Participant %s removed: Misses = %d (Condition 1)\n', participantID, numMisses);
        continue;
    elseif accuracy < 50
        excludedParticipants = [excludedParticipants; str2double(participantID)];
        fprintf('Participant %s removed: Accuracy = %.2f%% (Condition 2)\n', participantID, accuracy);
        continue;
    end

    % Store filtered EEG & events
    filteredEEG{end+1} = eegsignal;
    filteredEvents{end+1} = eventCodes;

    % Save processed data
    save(fullfile(outputDir, sprintf('filtered_EEG_%s.mat', participantID)), 'eegsignal');
    save(fullfile(outputDir, sprintf('filtered_events_%s.mat', participantID)), 'eventCodes');

    fprintf('Participant %s saved.\n', participantID);
end

% Print final summary
fprintf('\nSummary of Stage 1 Processing:\n');
fprintf('Total Participants Processed: %d\n', length(eegFiles));
fprintf('Participants Retained: %d\n', length(filteredEEG));
fprintf('Participants Excluded: %d\n', length(excludedParticipants));

% Save excluded participant list
save(fullfile(outputDir, 'excluded_participants.mat'), 'excludedParticipants');
fprintf('\nExcluded participant list saved to "excluded_participants.mat".\n');
