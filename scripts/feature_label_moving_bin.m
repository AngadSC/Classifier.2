
%still use a 100 ms window; 25 samples
%slide it forward by smaller amounts; 10 samples = 40 ms step so that they
%overlap
%example: 
% bin 1 from 26 to 50 time : 100 to (50*4) 
% bin 2 from 36 to 60 : 
% bin 3 from 46 to 70 : 

% ========================================================
% Moving Timebin Feature & Label Extraction for Each Subject
% ========================================================

% --- Paths ---
%inputDir = '/"C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier\events_20.mat"';
inputDir = '\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier\events_20.mat';
outputDir = '/Users/faisalanqouor/Desktop/Research/Multivariate_pipeline/feature_label_moving_v3';
inputDir_label = '/Users/faisalanqouor/Desktop/Research/stage_two_Matt';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

transformation = 'raw';  % Options: 'raw', 'abs', 'square', 'cube'

% --- Parameters ---
electrodes = [64 194 21 41 214 8 101 87 153 137];  % 10 electrodes
nElectrodes = length(electrodes);
samplerate = 250;  % Hz
binSize = 20;      % 100 ms = 25 samples
binStep = 5;      % 40 ms step = 10 samples
baselineEnd = 25;  % Skip baseline (samples 1–25)
startAfterBaseline = baselineEnd + 1;

% --- Get EEG files ---
eegFiles = dir(fullfile(inputDir, 'finalcorrected_EEG_*.mat'));

for i = 1:length(eegFiles)
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once');

    % Load EEG
    eegFile = fullfile(inputDir, sprintf('finalcorrected_EEG_%s.mat', participantID));
    eegStruct = load(eegFile);
    eeg = eegStruct.correctedEEG;  % size: [257 x 325 x trials]
     


    % Load labels
    eventFile = fullfile(inputDir_label, sprintf('finalcorrected_events_%s.mat', participantID));
    eventStruct = load(eventFile);
    labels = eventStruct.correctedEvents();  % [n_trials x 1]

    % Trial count check
    nTrials = size(eeg, 3);
    if length(labels) ~= nTrials
        warning('Mismatch between EEG and label count for %s', participantID);
        continue;
    end

    maxIdx = size(eeg, 2); % number of time samples
    startBins = startAfterBaseline:binStep:(maxIdx - binSize + 1);
    nMovingBins = length(startBins);

    % Preallocate
    X = zeros(nTrials, nElectrodes * nMovingBins);
    y = labels(:);

    % Feature extraction
    for t = 1:nTrials
        featureVec = [];

        for e = 1:nElectrodes
            ch = electrodes(e);

            for b = 1:nMovingBins
                startIdx = startBins(b);
                endIdx = startIdx + binSize - 1;
                binData = eeg(ch, startIdx:endIdx, t);

                % Transformation
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

    % Save result
    outFile = fullfile(outputDir, sprintf('features_labels_%s_%s.mat', transformation, participantID));
    save(outFile, 'X', 'y');
    fprintf('Saved %s features for participant %s (%d features per trial)\n', transformation, participantID, size(X,2));
end

fprintf('\nFeature extraction complete for all participants.\n');
