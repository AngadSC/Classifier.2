% ========================================================
% SWAPPABLE Moving Timebin Feature Extraction – STUDY PHASE
% Mirrors the TEST moving-bin script, but uses:
%   - EEG from:   studyData
%   - Labels from events_X.mat (variable "study", col 2)
%   - Output to:  outputs\study_moving_bin
% ========================================================

% ============== CONFIGURATION SWITCH ==============
% CHANGE THIS TO SWITCH BETWEEN RAW AND FILTERED STUDY DATA
USE_RAW_DATA = true;  % true = raw studyData, false = stage2 filtered
% ==================================================

if USE_RAW_DATA
    % ---------- RAW STUDY DATA CONFIGURATION ----------
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\studyData";
    inputDir_label = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
    outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\study_moving_bin";

    % Adjust these patterns if your filenames differ
    eegPattern   = 'OldNew_*_stims.mat';          % e.g., study_20.mat
    eventPattern = 'events_%s.mat';        % events_20.mat
    eegPrefix    = 'study_';

    % Variable names in RAW study files
    eegVarName      = 'stimdata';   % EEG variable inside study_XX.mat
    eventVarName    = 'study';  % matrix [nTrials x 2], use column 2
    useEventColumn  = true;
    eventColumn     = 2;

    fprintf('=== STUDY PHASE: USING RAW DATA ===\n');

else
    % ---------- STAGE2 FILTERED STUDY DATA CONFIG ----------
    % You can adjust these paths once you have stage2 study files
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage2_study";
    inputDir_label = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
    outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\study_moving_bin_after_filter";

    % Example pattern for filtered study EEG – update to your real names
    eegPattern   = 'stage2_filter_study_EEG_*.mat';
    eventPattern = 'events_%s.mat';          % still events_20.mat etc.
    eegPrefix    = 'stage2_filter_study_EEG_';

    % Variable names in stage2 files
    eegVarName      = 'data';
    eventVarName    = 'study';   % same event structure, column 2 codes
    useEventColumn  = true;
    eventColumn     = 2;

    fprintf('=== STUDY PHASE: USING STAGE2 FILTERED DATA ===\n');
end

% ---------- Common Parameters ----------
transformation = 'raw';  % 'raw' | 'abs' | 'square' | 'cube'
electrodes = [64 194 21 41 214 8 101 87 153 137];  % 10 electrodes
nElectrodes = numel(electrodes);

samplerate = 250;  % Hz

binSize     = round(0.10 * samplerate);  % 100 ms
binStep     = round(0.04 * samplerate);  % 40 ms
baselineEnd = round(0.10 * samplerate);  % first 100 ms
startAfterBaseline = baselineEnd + 1;

% ---------- Create output directory ----------
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% ---------- Get EEG files ----------
eegFiles = dir(fullfile(inputDir, eegPattern));

if isempty(eegFiles)
    error('No %s files found in: %s', eegPattern, inputDir);
end

fprintf('Found %d STUDY EEG files to process\n', numel(eegFiles));
fprintf('EEG directory:    %s\n', inputDir);
fprintf('Event directory:  %s\n', inputDir_label);
fprintf('Output directory: %s\n', outputDir);
fprintf('========================================\n');

% ---------- Process each participant ----------
successCount = 0;
failCount    = 0;

for i = 1:numel(eegFiles)
    participantID = 'UNKNOWN';
    try
        filename = eegFiles(i).name;

        % Extract participant ID from filename
        if USE_RAW_DATA
            % For study_XX.mat
            participantID = regexp(filename, '(?<=OldNew_)\d+(?=_stims\.mat)', 'match', 'once');
        else
            % For stage2_filter_study_EEG_XX.mat
            participantID = regexp(filename, '(?<=stage2_filter_study_EEG_)\d+(?=\.mat)', 'match', 'once');
        end

        if isempty(participantID)
            fprintf('Warning: Could not extract ID from %s, skipping...\n', filename);
            failCount = failCount + 1;
            continue;
        end

        fprintf('Processing STUDY participant %s...\n', participantID);

        % ---- Load EEG data ----
        eegFile   = fullfile(inputDir, filename);
        eegStruct = load(eegFile);

        if isfield(eegStruct, eegVarName)
            eeg = eegStruct.(eegVarName);
        elseif isfield(eegStruct, 'data')
            eeg = eegStruct.data;  % fallback
        else
            error('No EEG data found in %s (looked for "%s" or "data")', ...
                  eegFile, eegVarName);
        end

        % ---- Load event data (labels) ----
        eventFile = fullfile(inputDir_label, sprintf(eventPattern, participantID));

        if ~exist(eventFile, 'file')
            fprintf('Error: Event file not found: %s\n', eventFile);
            fprintf('Skipping participant %s...\n\n', participantID);
            failCount = failCount + 1;
            continue;
        end

        L = load(eventFile);

        % Get labels from "study" matrix or fallbacks
        if isfield(L, eventVarName)
            if useEventColumn && size(L.(eventVarName), 2) >= eventColumn
                labels = L.(eventVarName)(:, eventColumn);
            else
                labels = L.(eventVarName)(:);
            end
        else
            % flex fallbacks just in case
            if isfield(L, 'study') && size(L.study, 2) >= 2
                labels = L.study(:, 2);
            elseif isfield(L, 'label')
                labels = L.label(:);
            else
                error('No usable STUDY labels found in %s', eventFile);
            end
        end

        labels = labels(:);  % ensure column

        % ---- Trial count and bin setup ----
        [nCh, nTime, nTrials] = size(eeg);

        if numel(labels) ~= nTrials
            % For study phase, mismatch is an error (no artifact-rej version yet)
            error('Trial mismatch for %s: EEG=%d trials, labels=%d', ...
                  participantID, nTrials, numel(labels));
        end

        maxIdx    = nTime;
        startBins = startAfterBaseline:binStep:(maxIdx - binSize + 1);
        nMovingBins = numel(startBins);

        fprintf('  - EEG: [%d electrodes x %d timepoints x %d trials]\n', ...
                nCh, nTime, nTrials);
        fprintf('  - Features per trial: %d (%d bins x %d electrodes)\n', ...
                nElectrodes * nMovingBins, nMovingBins, nElectrodes);

        % ---- Preallocate ----
        X = zeros(nTrials, nElectrodes * nMovingBins);
        y = labels;

        % ---- Feature extraction ----
        for t = 1:nTrials
            featureVec = zeros(1, nElectrodes * nMovingBins);
            idx = 0;

            for e = 1:nElectrodes
                ch = electrodes(e);

                if ch > nCh
                    error('Electrode %d exceeds available channels (%d)', ch, nCh);
                end

                for b = 1:nMovingBins
                    startIdx = startBins(b);
                    endIdx   = startIdx + binSize - 1;

                    binData = eeg(ch, startIdx:endIdx, t);

                    % Apply transformation
                    switch transformation
                        case 'raw'
                            featureVal = mean(binData);
                        case 'abs'
                            featureVal = mean(abs(binData));
                        case 'square'
                            featureVal = mean(binData .^ 2);
                        case 'cube'
                            featureVal = mean(binData .^ 3);
                        otherwise
                            error('Unknown transformation: %s', transformation);
                    end

                    idx = idx + 1;
                    featureVec(idx) = featureVal;
                end
            end

            X(t, :) = featureVec;
        end

        % ---- Print label summary (generic) ----
        fprintf('  - Study label counts: ');
        uni = unique(y);
        for lbl = uni'
            fprintf('Code%d=%d ', lbl, sum(y == lbl));
        end
        fprintf('\n');

        % ---- Save results ----
        outFile = fullfile(outputDir, ...
            sprintf('features_labels_study_%s_%s.mat', transformation, participantID));

        save(outFile, 'X', 'y');
        fprintf('✓ Saved STUDY features for participant %s\n\n', participantID);
        successCount = successCount + 1;

    catch ME
        fprintf('✗ Error processing STUDY participant %s: %s\n\n', participantID, ME.message);
        failCount = failCount + 1;
    end
end

% ---------- Final Summary ----------
fprintf('========================================\n');
fprintf('STUDY FEATURE EXTRACTION COMPLETE\n');
fprintf('EEG source:   %s\n', inputDir);
fprintf('Event source: %s\n', inputDir_label);
fprintf('Results to:   %s\n', outputDir);
fprintf('Successfully processed: %d participant(s)\n', successCount);
if failCount > 0
    fprintf('Failed: %d participant(s)\n', failCount);
end
fprintf('========================================\n');
