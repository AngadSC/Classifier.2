function laplacian_data = apply_csd_transform(data,G,H,lambda,head_radius)

%Apply the CSD (laplacian) transform to EEG daata
%applites the transformation matrcies from the GH matrices file
%lamdba is a smoothing constant (default is 1e-5 for microvolts)
%head radius - the actual radius of the person skul;
%authors used this they (default 10cm) 
%output - laplaican data (CSD- trasnformed), dimensions will be the same as
%the input 

if nargin < 4, lambda = 1e-5; end 
if nargin < 5, head_radius = 10; end

[n_channels, n_timepoints, n_trials] = size(data);

%verify the G and H dimensions, we cannot have a mismatch\
if size(G,1) ~= n_channels || size(H,1) ~= n_channels
    error(['Channel count mismatch! Data has %d channels but G/H have %d channels.\n' ...
               'Your data channel order must match the order used to create G and H!'], ...
               n_channels, size(G, 1));
end

fprintf('Applying CSD transform to %d trials...\n', n_trials);

fprintf('Reshaping data for batch processing...\n');
data_2d = reshape(data, n_channels, n_timepoints * n_trials);


% Apply CSD transformation once to entire dataset
fprintf('Computing CSD (this may take a moment)...\n');
[csd_2d, ~] = CSD(data_2d, G, H, lambda, head_radius);

% Reshape back to original dimensions
    laplacian_data = reshape(csd_2d, n_channels, n_timepoints, n_trials);
    
    fprintf('CSD transform complete.\n');
end
    