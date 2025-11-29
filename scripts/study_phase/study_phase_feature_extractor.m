function outFiles = study_phase_feature_extractor(inputDir, outputDir, inputDir_label, electrodes, transformation)
% FEATURE_LABEL_MOVING_BIN_STUDY_V3  Moving time-bin EEG features for STUDY phase
%
% Usage (example):
%   outFiles = feature_label_moving_bin_study_v3('/path/to/data', ...
%               fullfile(pwd,'output_moving_bins_study'), [], [], 'raw');
%
% Inputs (all optional):
%   inputDir        : folder containing study EEG .mat files (e.g. study_XX.mat or finalcorrected_EEG_XX.mat)
%   outputDir       : where to write features
%   inputDir_label  : folder containing events_XX.mat
%   electrodes      : vector of channel indices to use
%   transformation  : 'raw' | 'abs' | 'square' | 'cube'
%
% Output:
%   outFiles        : cellstr of saved .mat files (one per participant)

% ---------------- Defaults ----------------
if nargin < 1 || isempty(inputDir)
    inputDir = fullfile(pwd, 'data');
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(pwd, 'feature_label_moving_study_v3');
end
if nargin < 3 || isempty(inputDir_label)
    inputDir_label = inputDir;
end
if nargin < 4 || isempty(electrodes)
    electrodes = [64 194 21 41 214 8 101 87 153 137];  % 10 electrodes
end
if nargin < 5 || isempty(transformation)
    transformation = 'raw';
end

% Ensure output directory exists and is writable (fallback to tempdir if not)
outputDir = ensureWritableDir(outputDir);

% ---------------- Parameters ----------------
samplerate = 250;           % Hz
binDurSec  = 0.100;         % 100 ms windows
stepDurSec = 0.040;         % 40 ms step between windows
baselineDurSec = 0.100;     % skip first 100 ms as baseline

binSize  = max(1, round(binDurSec  * samplerate));
binStep  = max(1, round(stepDurSec * samplerate));
baselineEnd = max(0, round(baselineDurSec * samplerate));
startAfterBaseline = baselineEnd + 1;

% ---------------- Gather EEG files ----------------
% Prefer study_*.mat, then finalcorrected_EEG_*.mat, then test_*.mat as last fallback.
eegFiles = dir(fullfile(inputDir, 'study_*.mat'));
if isempty(eegFiles)
    eegFiles = dir(fullfile(inputDir, 'finalcorrected_EEG_*.mat'));
end
if isempty(eegFiles)
    eegFiles = dir(fullfile(inputDir, 'test_*.mat'));
end
if isempty(eegFiles)
    error('No EEG files found at: %s', inputDir);
end

outFiles = {};

for i = 1:numel(eegFiles)
    % Participant ID from filename (digits)
    participantID = regexp(eegFiles(i).name, '\d+', 'match', 'once');

    % ---- Load EEG ----
    eegFile = fullfile(inputDir, eegFiles(i).name);
    S = load(eegFile);

    % Try common variable names
    if isfield(S, 'correctedEEG')
        eeg = S.correctedEEG;
    elseif isfield(S, 'EEG')
        eeg = S.EEG;
    elseif isfield(S, 'data')
        eeg = S.data;      % matches your test_XX.mat structure
    else
        error('Could not find EEG array in %s (looked for correctedEEG/EEG/data).', eegFile);
    end

    eeg = double(eeg);   % ensure numeric
    if ndims(eeg) ~= 3
        error('EEG must be 3D (channels x time x trials). Got size %s.', mat2str(size(eeg)));
    end

    nCh     = size(eeg,1);
    nTime   = size(eeg,2);
    nTrials = size(eeg,3);

    % Channel sanity check
    if any(electrodes > nCh)
        error('Electrode index exceeds available channels. Max channel = %d', nCh);
    end

    % ---- Load STUDY labels from events_X.mat ----
    eventFile = fullfile(inputDir_label, sprintf('events_%s.mat', participantID));
    if ~exist(eventFile, 'file')
        warning('No events file %s for participant %s. Skipping.', eventFile, participantID);
        continue;
    end

    E = load(eventFile);

    if isfield(E, 'study')
        ev = E.study;   % expected [nTrials x 2]: [time, label]
    else
        error('Could not find ''study'' variable in %s.', eventFile);
    end

    if size(ev,2) < 2
        error('study variable in %s must be N x 2 (time, label). Got %s.', eventFile, mat2str(size(ev)));
    end

    labels = ev(:,2);       % 0 = forgotten, 1 = remembered
    labels = labels(:);     % column vector

    % Trial count check
    if numel(labels) ~= nTrials
        warning('Mismatch EEG(%d trials) vs study labels(%d) for %s. Skipping.', ...
            nTrials, numel(labels), participantID);
        continue;
    end

    % ---- Moving windows ----
    maxIdx = nTime; % number of time samples
    if startAfterBaseline > maxIdx - binSize + 1
        warning('Not enough post-baseline samples for %s. Skipping.', participantID);
        continue;
    end
    startBins   = startAfterBaseline:binStep:(maxIdx - binSize + 1);
    nMovingBins = numel(startBins);

    % Preallocate features
    nElectrodes = numel(electrodes);
    X = zeros(nTrials, nElectrodes * nMovingBins, 'double');
    y = labels;  % 0/1 remembered vs forgotten; binarize in downstream code if needed

    % ---- Feature extraction ----
    for t = 1:nTrials
        featureVec = zeros(1, nElectrodes * nMovingBins);
        idx = 0;
        for e = 1:nElectrodes
            ch = electrodes(e);
            for b = 1:nMovingBins
                startIdx = startBins(b);
                endIdx   = startIdx + binSize - 1;
                binData  = eeg(ch, startIdx:endIdx, t);
                switch lower(transformation)
                    case 'raw'
                        featureVal = mean(binData);
                    case 'abs'
                        featureVal = mean(abs(binData));
                    case 'square'
                        featureVal = mean(binData.^2);
                    case 'cube'
                        featureVal = mean(binData.^3);
                    otherwise
                        error('Unknown transformation: %s', transformation);
                end
                idx = idx + 1;
                featureVec(idx) = featureVal;
            end
        end
        X(t, :) = featureVec;
    end

    % ---- Save result ----
    params = struct('samplerate', samplerate, 'binDurSec', binDurSec, ...
                    'stepDurSec', stepDurSec, 'baselineDurSec', baselineDurSec, ...
                    'binSizeSamples', binSize, 'binStepSamples', binStep, ...
                    'baselineEndSamples', baselineEnd, 'startBins', startBins, ...
                    'electrodes', electrodes, 'transformation', transformation, ...
                    'phase', 'study');

    outFile = fullfile(outputDir, sprintf('features_labels_study_%s_%s.mat', transformation, participantID));
    save(outFile, 'X', 'y', 'params', '-v7');
    fprintf('Saved STUDY %s (%d trials × %d features) for participant %s\n', ...
            transformation, size(X,1), size(X,2), participantID);
    outFiles{end+1} = outFile; %#ok<AGROW>
end

fprintf('\nStudy feature extraction complete for %d participant(s). Output: %s\n', numel(outFiles), outputDir);

end

% ---------------- Helpers ----------------
function outDir = ensureWritableDir(outDir)
% Create outDir if needed; fall back to tempdir if creation or write test fails
    if exist(outDir, 'file') == 2 && exist(outDir, 'dir') ~= 7
        error('Output path exists as a FILE, not a folder: %s', outDir);
    end
    [ok, msg, msgid] = mkdir(outDir);
    if ~ok
        warning('mkdir failed for %s (%s: %s). Falling back to tempdir.', outDir, msgid, msg);
        outDir = fullfile(tempdir, ['feature_label_moving_study_v3_' datestr(now,'yyyymmdd_HHMMSS')]);
        [ok2, msg2, msgid2] = mkdir(outDir);
        assert(ok2, 'Failed to create fallback output folder (%s: %s).', msgid2, msg2);
    end
    % Write test
    probe = fullfile(outDir, '.write_test');
    [fid, errmsg] = fopen(probe, 'w');
    if fid < 0
        warning('No write permission to %s (%s). Using tempdir fallback.', outDir, errmsg);
        outDir = fullfile(tempdir, ['feature_label_moving_study_v3_' datestr(now,'yyyymmdd_HHMMSS')]);
        [ok3, msg3, msgid3] = mkdir(outDir);
        assert(ok3, 'Failed to create fallback folder (%s: %s).', msgid3, msg3);
        fid = fopen(fullfile(outDir, '.write_test'), 'w');
        assert(fid >= 0, 'Still cannot write to fallback directory: %s', outDir);
    end
    fclose(fid);
    if exist(probe, 'file') == 2, delete(probe); end
end
