% ========================================================
% SWAPPABLE Moving Timebin Feature Extraction
% Works with both RAW and STAGE2 FILTERED data
% Complete version with inputDir_label
% ========================================================

% ============== CONFIGURATION SWITCH ==============
% CHANGE THIS TO SWITCH BETWEEN RAW AND FILTERED DATA
USE_RAW_DATA = true;  % Set to true for raw data, false for stage2 filtered
% ==================================================

% --- Set paths based on data source ---
if USE_RAW_DATA
    % RAW DATA CONFIGURATION
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
    inputDir_label = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";  % Can be different if needed
    outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw";
    
    eegPattern = 'test_*.mat';
    eventPattern = 'events_%s.mat';
    eegPrefix = 'test_';
    
    % Variable names in raw files
    eegVarName = 'data';
    eventVarName = 'test';  % Will use column 2
    useEventColumn = true;
    eventColumn = 2;
    
    fprintf('=== USING RAW DATA ===\n');
    
else
    % STAGE2 FILTERED DATA CONFIGURATION
    inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage2";
    inputDir_label = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage2";  % Can be different if needed
    outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_after_filter";
    
    eegPattern = 'stage2_filter_EEG_*.mat';
    eventPattern = 'stage2_filter_events_%s.mat';
    eegPrefix = 'stage2_filter_EEG_';
    
    % Variable names in stage2 files
    eegVarName = 'data';
    eventVarName = 'label';  % Already just the codes
    useEventColumn = false;  % label is already 1D
    eventColumn = [];
    
    fprintf('=== USING STAGE2 FILTERED DATA ===\n');
end

% --- Common Parameters (same for both) ---
transformation = 'raw';  % Options: 'raw', 'abs', 'square', 'cube'
electrodes = [64 194 21 41 214 8 101 87 153 137];  % 10 electrodes
nElectrodes = length(electrodes);
samplerate = 250;  % Hz
binSize     = round(0.10*samplerate);  % 25 samples = 100 ms
binStep     = round(0.04*samplerate);  % 10 samples =  40 ms
baselineEnd = round(0.10*samplerate);  % 25 samples = 100 ms
startAfterBaseline = baselineEnd + 1;

% --- Create output directory ---
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% --- Get EEG files ---
eegFiles = dir(fullfile(inputDir, eegPattern));

if isempty(eegFiles)
    error('No %s files found in: %s', eegPattern, inputDir);
end

fprintf('Found %d EEG files to process\n', length(eegFiles));
fprintf('EEG directory: %s\n', inputDir);
fprintf('Event directory: %s\n', inputDir_label);
fprintf('Output directory: %s\n', outputDir);
fprintf('========================================\n');

% --- Process each participant ---
successCount = 0;
failCount = 0;

for i = 1:length(eegFiles)
    try
        % Extract participant ID based on file pattern
        filename = eegFiles(i).name;
        
        % Extract ID - works for both patterns
        if USE_RAW_DATA
            % For test_XX.mat
            participantID = regexp(filename, '(?<=test_)\d+(?=\.mat)', 'match', 'once');
        else
            % For stage2_filter_EEG_XX.mat
            participantID = regexp(filename, '(?<=stage2_filter_EEG_)\d+(?=\.mat)', 'match', 'once');
        end
        
        if isempty(participantID)
            fprintf('Warning: Could not extract ID from %s, skipping...\n', filename);
            continue;
        end
        
        fprintf('Processing participant %s...\n', participantID);
        
        % ---- Load EEG data ----
        eegFile = fullfile(inputDir, filename);
        eegStruct = load(eegFile);
        
        % Get EEG data based on expected variable name
        if isfield(eegStruct, eegVarName)
            eeg = eegStruct.(eegVarName);
        elseif isfield(eegStruct, 'data')
            eeg = eegStruct.data;  % Fallback
        else
            error('No EEG data found in %s (looked for "%s")', eegFile, eegVarName);
        end
        
        % ---- Load event data from inputDir_label ----
        eventFile = fullfile(inputDir_label, sprintf(eventPattern, participantID));
        
        if ~exist(eventFile, 'file')
            fprintf('Error: Event file not found: %s\n', eventFile);
            fprintf('Skipping participant %s...\n', participantID);
            failCount = failCount + 1;
            continue;
        end
        
        L = load(eventFile);
        
        % Get labels based on file structure
        if isfield(L, eventVarName)
            if useEventColumn && size(L.(eventVarName), 2) >= eventColumn
                labels = L.(eventVarName)(:, eventColumn);
            else
                labels = L.(eventVarName)(:);
            end
        else
            % Try fallbacks
            if isfield(L, 'test') && size(L.test, 2) >= 2
                labels = L.test(:, 2);
            elseif isfield(L, 'label')
                labels = L.label(:);
            elseif isfield(L, 'events') && size(L.events, 2) >= 2
                labels = L.events(:, 2);
            else
                error('No usable labels found in %s', eventFile);
            end
        end
        
        % Ensure labels is a column vector
        labels = labels(:);
        
        % ---- Trial count check ----
        nTrials = size(eeg, 3);
        if numel(labels) ~= nTrials
            if USE_RAW_DATA
                % For raw data, this is an error
                error('Trial mismatch for %s: EEG=%d trials, labels=%d', ...
                    participantID, nTrials, numel(labels));
            else
                % For filtered data, this is expected
                fprintf('  Note: %d trials after artifact rejection\n', nTrials);
            end
        end
        
        % ---- Calculate time bins ----
        maxIdx = size(eeg, 2);
        startBins = startAfterBaseline:binStep:(maxIdx - binSize + 1);
        nMovingBins = length(startBins);
        
        fprintf('  - EEG: [%d electrodes x %d timepoints x %d trials]\n', ...
            size(eeg, 1), size(eeg, 2), size(eeg, 3));
        fprintf('  - Features per trial: %d (%d bins x %d electrodes)\n', ...
            nElectrodes * nMovingBins, nMovingBins, nElectrodes);
        
        % ---- Preallocate ----
        X = zeros(nTrials, nElectrodes * nMovingBins);
        y = labels(:);
        
        % ---- Feature extraction ----
        for t = 1:nTrials
            featureVec = [];
            
            for e = 1:nElectrodes
                ch = electrodes(e);
                
                % Check electrode exists
                if ch > size(eeg, 1)
                    error('Electrode %d exceeds available channels (%d)', ch, size(eeg, 1));
                end
                
                for b = 1:nMovingBins
                    startIdx = startBins(b);
                    endIdx = startIdx + binSize - 1;
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
                    
                    featureVec(end+1) = featureVal;
                end
            end
            X(t, :) = featureVec;
        end
        
        % ---- Print label summary ----
        fprintf('  - Labels: ');
        unique_labels = unique(y);
        label_names = {'Hit', 'Miss', 'CR', 'FA'};
        for lbl = unique_labels'
            count = sum(y == lbl);
            if lbl >= 1 && lbl <= 4
                fprintf('%s=%d ', label_names{lbl}, count);
            else
                fprintf('Code%d=%d ', lbl, count);
            end
        end
        fprintf('\n');
        
        % ---- Save results ----
        outFile = fullfile(outputDir, sprintf('features_labels_%s_%s.mat', transformation, participantID));
        save(outFile, 'X', 'y');
        fprintf('✓ Saved features for participant %s\n\n', participantID);
        
        successCount = successCount + 1;
        
    catch ME
        fprintf('✗ Error processing participant %s: %s\n\n', participantID, ME.message);
        failCount = failCount + 1;
    end
end

% ---- Final Summary ----
fprintf('========================================\n');
fprintf('FEATURE EXTRACTION COMPLETE\n');
fprintf('EEG source: %s\n', inputDir);
fprintf('Event source: %s\n', inputDir_label);
fprintf('Results saved to: %s\n', outputDir);
fprintf('Successfully processed: %d participants\n', successCount);
if failCount > 0
    fprintf('Failed: %d participants\n', failCount);
end
fprintf('========================================\n');