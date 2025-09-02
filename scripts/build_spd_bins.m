function build_spd_bins(varargin)

inputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs";
inputDir_label = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\data";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


% ----- PARAMETERS ---------
samplerate = 250;                        % Hz
binSize     = round(0.10*samplerate);    % 25 samples = 100 ms
binStep     = round(0.04*samplerate);    % 10 samples =  40 ms
baselineEnd = round(0.10*samplerate);    % 25 samples = 100 ms
startAfterBaseline = baselineEnd + 1;    % first sample after baseline
demeanPerBin = true;                     % remove per-channel mean inside each bin,
lambda = 1e-2;                           % diagonal ridge (scaled) to guarantee SPD


eegFiles = dir(fullfile(inputDir, 'test_*.mat'));      % list test files
if isempty(eegFiles)                                   % if none,
    eegFiles = dir(fullfile(inputDir, 'finalcorrected_EEG_*.mat')); % fallback
end
if isempty(eegFiles); error('No EEG files found in: %s', inputDir); end 

for i = 1:length(eegFiles)
    % Extract participant ID and load EEG
    eegFile = fullfile(eegFiles(i).folder, eegFiles(i).name);
    eegStruct = load(eegFile);

    % Try common variable names 
    if     isfield(eegStruct, 'data')
        eeg = eegStruct.data;              % test_XX.mat usual var
    elseif isfield(eegStruct, 'correctedEEG')
        eeg = eegStruct.correctedEEG;      % finalcorrected_EEG_XX.mat
    elseif isfield(eegStruct, 'EEG')
        eeg = eegStruct.EEG;
    else
        error('No EEG array found in %s (looked for data/correctedEEG/EEG).', eegFile);
    end
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once');

    if ndims(eeg) ~= 3
        error('EEG must be [channels x time x trials] in %s', eegFile);
    end
    [nCh, nTime, nTrials] = size(eeg);

    eventFile = fullfile(inputDir_label, sprintf('events_%s.mat', participantID));
    L = load(eventFile)

    if isfield(L,'test') && size(L.test,2) >= 2
        labels = L.test(:,2);        % codes: 1=hit, 2=miss, 3=CR, 4=FA  (should be 450 long)
    elseif isfield(L,'events') && size(L.events,2) >= 2
        labels = L.events(:,2);      % fallback if variable is named 'events'
    elseif isfield(L,'correctedEvents')
        labels = L.correctedEvents(:); % only if your EEG were study-phase (usually 225 long)
    else
        error('No usable labels in %s (expected test(:,2)/events(:,2)/correctedEvents).', eventFile);
    end