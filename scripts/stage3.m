%====================================================================
% Stage 3 — FA/CR Classification Participant Filtering (X,y preserved)
%
% PURPOSE
%   This script filters participants for the FA/CR classification task
%   using the *existing* moving-bin outputs (files named
%   features_labels_raw_<ID>.mat that contain variables X and y).
%
%   *** IMPORTANT GUARANTEES ***
%   - Uses ONLY the specified criteria:
%       (1) False Alarms (FA) >= 10
%       (2) False Alarm rate < 50%   [strictly less than 50]
%   - Does NOT add, change, or remove any other filters/criteria.
%   - Does NOT change the shape or content of X or y for included
%     participants; it simply re-saves X and y as-is to the output folder.
%
% INPUT (per participant)
%   features_labels_raw_<ID>.mat
%       X : [trials x features]   double   (unchanged by this script)
%       y : [trials x 1]          double   (labels; 1=Hit, 2=Miss, 3=CR, 4=FA)
%
% OUTPUT (per participant that passes the gate)
%   features_labels_raw_<ID>.mat  (saved to outputDir with the SAME X,y)
%
% ADDITIONAL OUTPUTS
%   filtered_FA_CR_participants.mat : cell array 'included' of IDs that passed
%   excluded_FA_CR_participants.mat : cell array 'excluded' of IDs that failed
%
% NOTES
%   - This script does not subset trials or alter variables.
%   - It only gates participants and re-saves X,y for those who pass.
%====================================================================

%----------------------------
% Directory configuration
%----------------------------
inputDir  = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_after_filter";
outputDir = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\stage3";

% Ensure the output directory exists; create if it does not.
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%---------------------------------------
% Find all per-participant X,y files
%---------------------------------------
% We only look for files named 'features_labels_raw_*.mat' in inputDir.
files = dir(fullfile(inputDir, "features_labels_raw_*.mat"));

% Prepare containers to record which participant IDs are included/excluded.
included = {};   % cell array of strings (participant IDs that pass)
excluded = {};   % cell array of strings (participant IDs that fail)

%---------------------------------------
% Constants for label codes (from labels.txt)
%---------------------------------------
% TEST phase codes:
%   1 = Hit, 2 = Miss, 3 = Correct Rejection (CR), 4 = False Alarm (FA)
HIT_CODE = 1;  %#ok<NASGU>  % not directly used here, kept for clarity
MISS_CODE = 2; %#ok<NASGU>  % not directly used here, kept for clarity
CR_CODE = 3;
FA_CODE = 4;

%---------------------------------------
% Loop over each participant file
%---------------------------------------
for i = 1:numel(files)
    % Full filename and path of the current participant file
    fname = files(i).name;
    fpath = fullfile(files(i).folder, fname);

    % Load the participant's X and y.
    % EXPECTED: file contains variables X (trials x features) and y (trials x 1).
    S = load(fpath);

    % Validate presence of X and y; error if missing (do not proceed silently).
    assert(isfield(S,'X') && isfield(S,'y'), ...
        'File %s must contain variables X and y.', fname);

    % Ensure y is a column vector; X is trials x features.
    X = S.X;
    y = S.y(:);

    % Double-check trial alignment: number of trials must match.
    assert(size(X,1) == numel(y), ...
        'Trials mismatch in %s: size(X,1)=%d vs numel(y)=%d.', ...
        fname, size(X,1), numel(y));

    %--------------------------------------------------------
    % Compute FA/CR metrics used for participant-level gate
    %--------------------------------------------------------
    % Count False Alarms (FA == 4) and Correct Rejections (CR == 3)
    faCount = sum(y == FA_CODE);
    crCount = sum(y == CR_CODE);

    % "New" trials are CR + FA
    newTrials = faCount + crCount;

    % Compute FA rate = FA / (CR + FA) * 100
    % Guard against division by zero: if there are no new trials, define FA rate as 0.
    if newTrials == 0
        faRate = 0;
    else
        faRate = (faCount / newTrials) * 100;
    end

    % Extract participant ID digits from filename (e.g., '..._74.mat' -> '74')
    pid = regexp(fname, '\d+', 'match', 'once');

    %--------------------------------------------------------
    % Apply EXACT gate (no changes/additions):
    %   Include if FA >= 10 AND FA rate < 50%
    %--------------------------------------------------------
    pass = (faCount >= 10) && (faRate < 50);

    if pass
        %-----------------------------------------------
        % Participant passes:
        %   - Re-save the SAME X and y (unchanged)
        %   - Use same naming pattern in the output dir
        %-----------------------------------------------
        outName = sprintf('stage3_%s.mat', pid);
        save(fullfile(outputDir, outName), 'X', 'y');

        % Record inclusion and print a concise status line
        included{end+1} = pid; %#ok<SAGROW>
        fprintf('✅ ID %s INCLUDED | FA=%d, CR=%d, FA rate=%.2f%% | saved %s\n', ...
            pid, faCount, crCount, faRate, outName);
    else
        % Participant fails the gate; do not save X,y.
        excluded{end+1} = pid; %#ok<SAGROW>
        fprintf('❌ ID %s EXCLUDED | FA=%d, CR=%d, FA rate=%.2f%%\n', ...
            pid, faCount, crCount, faRate);
    end
end

%---------------------------------------
% Persist inclusion/exclusion lists
%---------------------------------------
save(fullfile(outputDir, 'filtered_FA_CR_participants.mat'), 'included');
save(fullfile(outputDir, 'excluded_FA_CR_participants.mat'), 'excluded');

% Final summary line
fprintf('\nSummary: %d files processed | %d included | %d excluded.\n', ...
    numel(files), numel(included), numel(excluded));
