clear; clc;

%CONFIG
INPUT_DIR = 'C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\moving_bin_raw';
OUTPUT_DIR = "C:\Users\Angad\OneDrive\Desktop\Comp Memory Lab\Classifier.2\outputs\laplacian";
% GH_FILE = location of the GH matrices file 

lambda = 1e-5; 
HEAD_RADIUS = 10;

CONVERT_TO_MICROVOLTS = false; 

fprintf('Loading G and H matrcies ...\n');
gh_data = load(GH_FILE);
G = gh_data.G;
H = gh_data.H;
M = gh_data.M;

fprintf('G and H matrices are for %d channels\n', length(M.lab));
fprintf('Expected channel order (first 5): %s, %s, %s, %s, %s\n', ...
        M.lab{1}, M.lab{2}, M.lab{3}, M.lab{4}, M.lab{5});

file_pattern = fullfile(INPUT_DIR, 'test_*.mat');
file_list = dir(file_pattern);

fprintf('Found %d files to process\n\n', length(file_list));


for f = 1:length(file_list)
    input_file = fullfile(INPUT_DIR, file_list(f).name);
    output_file = fullfile(OUTPUT_DIR, file_list(f).name);

    fprintf('=== Processing file %d/%d: %s ===\n', f, length(file_list), file_list(f).name);

    fprintf('Loading data...\n');
    load(input_file, 'test_X');
    
    fprintf('Data dimensions: %s (channels x timepoints x trials)\n', mat2str(size(test_X)));

    if size(test_X, 1) ~= length(M.lab)
        error(['Channel count mismatch!\n' ...
               'Data has %d channels but G/H have %d channels.\n' ...
               'Check that your data channel order matches the SFP file.'], ...
               size(test_X, 1), length(M.lab));
    end

    data_range = [min(test_X(:)) max(test_X(:))];
    fprintf('Data range: [%.6f, %.6f]\n', data_range(1), data_range(2));
    
    if abs(data_range(1)) < 0.001 && abs(data_range(2)) < 0.001 && ~CONVERT_TO_MICROVOLTS
        warning(['Data appears to be in volts (very small values).\n' ...
                 'Consider setting CONVERT_TO_MICROVOLTS = true']);
    end

    if CONVERT_TO_MICROVOLTS
        fprintf('Converting from volts to microvolts...\n');
        test_X = test_X * 1e6;
    end

    test_X_laplacian = apply_csd_transform(test_X, G, H, LAMBDA, HEAD_RADIUS);
    
    % Replace original data with Laplacian version
    test_X = test_X_laplacian;
    
    % Save to output directory
    fprintf('Saving to: %s\n', output_file);
    save(output_file, 'test_X');
    
    fprintf('File %d/%d complete.\n\n', f, length(file_list));
end

fprintf('=== All files processed successfully! ===\n');
fprintf('Output directory: %s\n', OUTPUT_DIR);
