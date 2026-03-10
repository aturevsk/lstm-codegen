%% generate_code_opt3.m
%  Option 3: importNetworkFromPyTorch -> dlnetwork -> MATLAB Coder
%  Generates library-free C code using Embedded Coder with DeepLearningConfig('none')
%
%  Workflow:
%    importNetworkFromPyTorch(.pt2) -> dlnetwork -> codegen -> C
%
%  Usage:
%    cd option3_matlab_dlnetwork
%    run('generate_code_opt3.m')

PROJECT  = fileparts(fileparts(mfilename('fullpath')));
OPT3_DIR = fullfile(PROJECT, 'option3_matlab_dlnetwork');
PT2_FILE = fullfile(PROJECT, 'LSTMSeqToSeqModel.pt2');
OUT_DIR  = fullfile(OPT3_DIR, 'codegen_opt3');

fprintf('=== Option 3: importNetworkFromPyTorch -> dlnetwork -> MATLAB Coder ===\n');
fprintf('  Model  : %s\n', PT2_FILE);
fprintf('  Output : %s\n', OUT_DIR);

%% Step 1: Import model from PyTorch .pt2
fprintf('\n  Importing PyTorch model...\n');
net = importNetworkFromPyTorch(PT2_FILE);
fprintf('  SUCCESS: %d layers\n', length(net.Layers));
for i = 1:length(net.Layers)
    fprintf('    Layer %d: %s (%s)\n', i, net.Layers(i).Name, class(net.Layers(i)));
end

%% Step 2: Check which layers are codegen-compatible
% The auto-generated custom argmax layer (LSTMModel_max_to_3) may not support codegen.
% We strip it and apply argmax in the entry-point function.
fprintf('\n  Checking layers for codegen compatibility...\n');
nLayers = length(net.Layers);
for i = 1:nLayers
    L = net.Layers(i);
    if isa(L, 'nnet.layer.Layer') && ~isa(L, 'nnet.cnn.layer.SequenceInputLayer') && ...
       ~isa(L, 'nnet.cnn.layer.LSTMLayer') && ~isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        fprintf('    Layer %d [%s] is custom — will be handled in entry-point\n', i, L.Name);
    end
end

%% Step 3: Build a codegen-compatible dlnetwork (standard layers only)
% Keep: SequenceInput -> LSTM -> FullyConnected (drop custom argmax)
% The MATLAB Embedded Coder can handle standard sequence/LSTM/FC layers.
fprintf('\n  Building codegen-compatible network (standard layers only)...\n');

% Find standard layers
stdLayers = {};
for i = 1:nLayers
    L = net.Layers(i);
    if isa(L, 'nnet.cnn.layer.SequenceInputLayer') || ...
       isa(L, 'nnet.cnn.layer.LSTMLayer') || ...
       isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        stdLayers{end+1} = L; %#ok<AGROW>
    end
end
fprintf('  Using %d standard layers (dropping %d custom)\n', ...
    length(stdLayers), nLayers - length(stdLayers));

% Reconstruct as lgraph and convert to dlnetwork
lg = layerGraph();
for i = 1:length(stdLayers)
    lg = addLayers(lg, stdLayers{i});
end
for i = 1:length(stdLayers)-1
    lg = connectLayers(lg, stdLayers{i}.Name, stdLayers{i+1}.Name);
end
netCodegen = dlnetwork(lg);

%% Step 4: Validate the trimmed network
fprintf('\n  Validating trimmed network...\n');
X_test = dlarray(single(rand(3,75,1)), 'CTB');
Y_test = predict(netCodegen, X_test);
fprintf('  Output size: [%s], class: %s\n', num2str(size(Y_test)), class(Y_test));

%% Step 5: Save the trimmed network for codegen
netFile = fullfile(OPT3_DIR, 'lstm_net3.mat');
save(netFile, 'netCodegen');
fprintf('  Saved trimmed network: %s\n', netFile);

%% Step 6: Write entry-point function
epFile = fullfile(OPT3_DIR, 'lstm_infer_opt3.m');
fid = fopen(epFile, 'w');
fprintf(fid, '%% lstm_infer_opt3.m  — codegen entry point for Option 3\n');
fprintf(fid, '%% importNetworkFromPyTorch -> dlnetwork -> MATLAB Coder\n');
fprintf(fid, '%%\n');
fprintf(fid, '%% Input : single [3 x 75]   (Channels x Time, one sample)\n');
fprintf(fid, '%% Output: int32  [1 x 75]   (per-step class index, 0-indexed)\n');
fprintf(fid, 'function pred = lstm_infer_opt3(X)\n');
fprintf(fid, '%%#codegen\n');
fprintf(fid, 'persistent net;\n');
fprintf(fid, 'if isempty(net)\n');
fprintf(fid, '    net = coder.loadDeepLearningNetwork(''lstm_net3.mat'', ''net3'');\n');
fprintf(fid, 'end\n');
fprintf(fid, '%% dlarray format: CT (Channels x Time), batch-less for codegen\n');
fprintf(fid, 'Xdl = dlarray(X, ''CT'');\n');
fprintf(fid, 'Y = predict(net, Xdl);      %% [5 x 75] float32 logits\n');
fprintf(fid, 'Ydata = extractdata(Y);     %% [5 x 75] float32\n');
fprintf(fid, '%% Argmax along dim 1 (class axis) -> 1-indexed, convert to 0-indexed\n');
fprintf(fid, '[~, idx] = max(Ydata, [], 1);\n');
fprintf(fid, 'pred = int32(idx) - int32(1);\n');
fprintf(fid, 'end\n');
fclose(fid);
fprintf('  Entry-point: %s\n', epFile);

%% Step 7: Configure MATLAB Coder (Embedded Coder, library-free)
cfg = coder.config('lib');
cfg.TargetLang         = 'C';
cfg.GenerateReport     = true;
cfg.EnableOpenMP       = false;
cfg.DeepLearningConfig = coder.DeepLearningConfig('none');


%% Step 8: Define input type [3 x 75] single, fixed size
inType = coder.typeof(single(zeros(3,75)), [3 75], [false false]);

%% Step 9: Run codegen
fprintf('\n  Running codegen (dlnetwork -> C, DeepLearningConfig=none)...\n');
[~,~] = mkdir(OUT_DIR);
prevDir = cd(OPT3_DIR);  % must be in dir containing net .mat file
cleanup = onCleanup(@() cd(prevDir));

try
    codegen('-config', cfg, ...
            '-d', OUT_DIR, ...
            epFile, ...
            '-args', {inType});
    fprintf('  SUCCESS\n');
catch ME
    cFiles = dir(fullfile(OUT_DIR, '*.c'));
    if ~isempty(cFiles)
        fprintf('  NOTE: Build step failed (%s)\n', ME.message);
        fprintf('  C source files generated (%d files)\n', length(cFiles));
    else
        fprintf('  FAILED: %s\n', ME.message);
        exit(1);
    end
end

%% Step 10: List generated files
fprintf('\n  Generated files:\n');
d = [dir(fullfile(OUT_DIR,'*.c')); dir(fullfile(OUT_DIR,'*.h'))];
for i = 1:length(d)
    fprintf('    %s  (%d bytes)\n', d(i).name, d(i).bytes);
end

fprintf('\n=== Option 3 codegen complete ===\n');
exit(0);
