% ========================================================
% Multivariate Feature & Label Extraction for Each Subject
% ========================================================

% --- Paths ---
inputDir = '/Users/faisalanqouor/Desktop/Research/stage_two_Matt';
outputDir = '/Users/faisalanqouor/Desktop/Research/Multivariate_pipeline/feature_label_final';
inputDir_label = '/Users/faisalanqouor/Desktop/Research/stage_two_Matt';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
transformation = 'cube';  % Options: 'raw', 'abs', 'square', 'cube'
% --- Electrode indices
electrodes =  [64 194 21 41 214 8 101 87 153 137]; % 10 electrodes to match suchetas approach 
nElectrodes = length(electrodes); %10 in the case of the paper
samplerate = 250; %as mentioned by sucheta in the paper.(first 25 samples are the baseline)
binSize = 25;%25* 250 * (1000/250sample rate= 4) = 100ms per bin, and we have 13000 ms so 13 bins, first being the baseline
nBins = 13;%Sucheta says 12, but we have baseline which im assuming is skipped over, i think its clearer to have 13 and skip the first bin


% --- Get EEG files 
eegFiles = dir(fullfile(inputDir, 'finalcorrected_EEG_*.mat'));

for i = 1:length(eegFiles)
    % Extract participant ID
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once');

    % Load EEG data
    eegFile = fullfile(inputDir, sprintf('finalcorrected_EEG_%s.mat', participantID));
    eegStruct = load(eegFile);
    eeg = eegStruct.correctedEEG;  % size: [257 x 325 x trials]
     

    % Load event codes
    eventFile = fullfile(inputDir_label, sprintf('finalcorrected_events_%s.mat', participantID));
    eventStruct = load(eventFile);
    labels = eventStruct.correctedEvents();  % [n_trials x 1]

    % Sanity check lol
    nTrials = size(eeg, 3);
    if length(labels) ~= nTrials
        warning('Mismatch between EEG trials and labels for participant %s', participantID);
        continue;
    end

    % --- Initialize output ---
    X = zeros(nTrials, nElectrodes * (nBins - 1)); % feature vector [trials x (10 x (13-1))] = [425 x 120]
    y = labels(:);  

    % --- Feature extraction --- %t:trial/e:electrode/b:bin
    for t = 1:nTrials
        featureVec = [];

        for e = 1:nElectrodes
            ch = electrodes(e);

            for b = 2:nBins %skip over the first bin 
                startIdx = (b - 1) * binSize + 1; % first loop: (2-1) * 25+1 = timepoint index:26
                endIdx = b * binSize;% first loop: 2 * 25 = timepoint index:50
                binData = eeg(ch, startIdx:endIdx, t); %eeg = [electrode; start:end; trial]
                
                %here is the parameter testing part
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
                        error('Unknown voltage transformation type: %s', transformation);
                end
                featureVec(end+1) = featureVal;
            end 

        end
        X(t, :) = featureVec; % store full 120-D feature vector for trial t, the : basically means for all cols, so from 1:120
        % here x [ numtrials x 120 ], so for each trial we get 120 features

    end

    % --- Save features and labels ---
save(fullfile(outputDir, sprintf('features_labels_%s_%s.mat', transformation, participantID)), 'X', 'y');
fprintf('Saved %s features and labels for participant %s\n', transformation, participantID);
end

fprintf('\nFeature extraction complete for all participants.\n');
