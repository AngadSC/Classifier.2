clear ; clc;


% example_file = one of the test mat files 
GH_FILE = 'GH_matrices_257elec.mat';
% SFP_FILE = my sfp file,

load(example_file, 'test_X');
load(GH_FILE, 'M');
sfp_montage = readlocs(SFP_FILE);

fprintf('=== CHANNEL ORDER VERIFICATION ===\n\n');
fprintf('Data matrix: %d channels\n', size(test_X, 1));
fprintf('G/H matrices: %d channels\n', length(M.lab));
fprintf('SFP file: %d channels\n\n', length(sfp_montage));


if size(test_X, 1) ~= length(M.lab)
    erorr ("Channel count mismatch");
end 

fprintf('Channel comparison (first 20):\n');
fprintf('%-5s %-15s %-15s %-10s\n', 'Index', 'G/H Label', 'SFP Label', 'Match?');
fprintf('%s\n', repmat('-', 1, 50));

all_match=true;
mismatches=[];

for i = 1:min(20, length(M.lab))
    gh_label = M.lab{i};
    sfp_label = sfp_montage(i).labels;
    match = strcmp(gh_label, sfp_label);

    if match
        fprintf('%-5d %-15s %-15s %-10s\n', i, gh_label, sfp_label, 'YES');
    else
        fprintf('%-5d %-15s %-15s %-10s\n', i, gh_label, sfp_label, '*** NO ***');
        all_match = false;
        mismatches = [mismatches i];
    end
end

fprintf('\n');
if all_match
    fprintf('✓ ALL CHANNELS MATCH!\n');
    fprintf('✓ Safe to proceed with batch processing\n');
else
    fprintf('✗ MISMATCHES DETECTED at indices: %s\n', mat2str(mismatches));
    fprintf('✗ DO NOT proceed - fix channel order first\n');

end 