% d' calculation and exclusion factor this will be inputted into the
% classifer
%inputs the output from feature-label extraction

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
%-------------------------------------------------------------------------
%Compute the d' 
    hitCount = sum(y==HIT_CODE);
    missCount = sum(y==MISS_CODE);
    crCount = sum(y==CR_CODE);
    faCount = sum(y==FA_CODE);

    %old trials are the hits +missses
    oldTrials = hitCount +missCount;
    %new trials are the CR+FA
    newTrials  = crCount +faCount;

% calculares the hit rate and fa rate and protect from 0-dviison
    if oldTrials ==0
        hitRate =0;
    else
        hitRate = hitCount / oldTrials;
    end
    
    if newTrials ==0
        faRate =0;
    else
        faRate = faCount/newTrials;
    end 
    
    %Computing edge cases lile if hit rate is 1 

    if hitRate ==1
        hitRate =1 -1/(2*oldTrials);
    elseif hitRate ==0
        hitRate =1/(2*oldTrials);
    end

    %if the fa rate is 1 or 0 edge cases
    if faRate == 1
        faRate = 1 - 1/(2*newTrials);
    elseif faRate == 0
        faRate = 1/(2*newTrials);
    end

    %d' calculations 

    d_prime_val = norminv(hitRate) - norminv(faRate);

    pid = regexp(fname, '\d+', 'match', 'once');

    %threshold adjustable d'>=threshold ( keep trial)

    D_PRIME_THRESHOLD = 0.5;

    pass(d_prime_val >= D_PRIME_THRESHOLD);

    if pass 
        outName= sprintf('dprime_exclusion_%s.mat', pid);
        save(fullfile(outputDir, outName), 'X', 'y');

        included{end+1} = pids;
        fprintf('✅ ID %s INCLUDED | d''=%.3f | Hits=%d, FA=%d, HR=%.2f%%, FAR=%.2f%% | saved %s\n', ...
        pid, d_prime, hitCount, faCount, hitRate*100, faRate*100, outName)

    else 
        excluded{end+1} = pid; %#ok<SAGROW>
    fprintf('❌ ID %s EXCLUDED | d''=%.3f | Hits=%d, FA=%d, HR=%.2f%%, FAR=%.2f%%\n', ...
        pid, d_prime, hitCount, faCount, hitRate*100, faRate*100);
end
